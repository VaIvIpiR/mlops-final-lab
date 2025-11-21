import pandas as pd
import numpy as np
import torch
import os
import json
import argparse
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW 
from sklearn.metrics import accuracy_score, f1_score

# --- CUSTOM DATASET ---
class CustomerSupportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_dir):
    print(f"Loading data from {data_dir}...")
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    return train_df, val_df, test_df

def encode_labels(train_df, val_df, test_df):
    unique_labels = sorted(train_df['intent'].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    train_df['label'] = train_df['intent'].map(label2id)
    val_df['label'] = val_df['intent'].map(label2id)
    test_df['label'] = test_df['intent'].map(label2id)
    return train_df, val_df, test_df, label2id, id2label

def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / len(train_loader)

def evaluate(model, eval_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, predictions), f1_score(true_labels, predictions, average='macro')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    args = parser.parse_args()

    # Setup MLflow
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("sagemaker-bert-experiment")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with mlflow.start_run():
        train_df, val_df, test_df = load_data(args.data_dir)
        train_df, val_df, test_df, label2id, id2label = encode_labels(train_df, val_df, test_df)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label2id), id2label=id2label, label2id=label2id)
        model.to(device)

        train_dataset = CustomerSupportDataset(train_df['utterance'], train_df['label'], tokenizer)
        val_dataset = CustomerSupportDataset(val_df['utterance'], val_df['label'], tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * args.epochs)

        mlflow.log_params({"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.learning_rate})

        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
            val_acc, val_f1 = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1}: Loss {train_loss:.4f}, Val Acc {val_acc:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_acc": val_acc, "val_f1": val_f1}, step=epoch)

        # Save for SageMaker (Це обов'язково, щоб SageMaker зберіг model.tar.gz)
        print(f"Saving model to {args.model_dir}...")
        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
        with open(os.path.join(args.model_dir, 'label_mapping.json'), 'w') as f:
            json.dump({'id2label': id2label, 'label2id': label2id}, f)

        # --- ДОДАЙ ЦЕЙ БЛОК ДЛЯ PIPELINE ---
        # Зберігаємо метрики у JSON файл, щоб Pipeline міг їх прочитати для ConditionStep
        metrics_data = {
            "classification_metrics": {
                "validation:accuracy": {"value": val_acc},
                "validation:f1": {"value": val_f1}
            }
        }
        
        # Зберігаємо в папку output (SageMaker автоматично забере це)
        output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
        # Переконуємося, що папка існує
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
            json.dump(metrics_data, f)
        
        print(f"📝 Metrics saved to {output_dir}/evaluation.json for Pipeline Condition")
        # -----------------------------------

    print("✅ Training complete!")