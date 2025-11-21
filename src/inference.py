import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def model_fn(model_dir):
    """Завантажує модель"""
    print(f"Loading model from: {model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    labels_path = os.path.join(model_dir, 'label_mapping.json')
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            mappings = json.load(f)
            id2label = mappings['id2label']
            label2id = mappings['label2id']
    else:
        id2label, label2id = None, None

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=len(label2id) if label2id else 2,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    model.eval()
    return {'model': model, 'tokenizer': tokenizer}

def input_fn(request_body, request_content_type):
    """
    Обробляє вхідні дані.
    ЗМІНА: Більш м'яка перевірка content_type.
    """
    print(f"DEBUG: Input received. Content-Type: {request_content_type}")
    
    ct = str(request_content_type).lower()
    

    if 'json' in ct:
        try:
            data = json.loads(request_body)
            
            if isinstance(data, str): return data
            elif 'inputs' in data: return data['inputs']
            elif 'text' in data: return data['text']
            elif isinstance(data, list): return data[0]
            return str(data)
            
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {e}")


    return str(request_body)

def predict_fn(input_data, model_dict):
    """Робить прогноз"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    
    if not isinstance(input_data, str):
        input_data = str(input_data)

    inputs = tokenizer(input_data, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        
    return {"label_id": pred_id, "confidence": probs[0][pred_id].item()}

def output_fn(prediction, response_content_type):
    """Форматує вихід"""
    return json.dumps(prediction)