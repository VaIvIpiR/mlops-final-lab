import os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, CreateModelStep, ProcessingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

TERRAFORM_BUCKET = "mlops-lab-terraform-vaivipir-data"
TERRAFORM_ROLE = "arn:aws:iam::584360834542:role/mlops-lab-terraform-role"

def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    pipeline_name="NewsClassifierPipeline",
    model_package_group_name="NewsClassifierGroup",
    base_job_prefix="NewsClassifier"
):
    """
    Функція, яка повертає об'єкт Pipeline.
    """
    sess = sagemaker.Session()
    
    if default_bucket is None:
        default_bucket = TERRAFORM_BUCKET

    # Параметри
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.large")
    input_data = ParameterString(name="InputData", default_value=f"s3://{default_bucket}/data/raw/")
    epochs = ParameterInteger(name="Epochs", default_value=3)
    batch_size = ParameterInteger(name="BatchSize", default_value=8)

    # --- Крок 1: Тренування ---
    estimator = PyTorch(
        entry_point='train_sagemaker.py',
        source_dir='./src', 
        role=role,
        framework_version='1.13',
        py_version='py39',
        instance_count=1,
        instance_type=training_instance_type,
        output_path=f"s3://{default_bucket}/NewsClassifier/training_jobs",
        hyperparameters={'epochs': epochs, 'batch-size': batch_size, 'learning-rate': 2e-5},
        environment={
            'MLFLOW_TRACKING_URI': 'https://dagshub.com/vaivipir/news-classifier.mlflow',
            'MLFLOW_TRACKING_USERNAME': 'vaivipir',
            'MLFLOW_TRACKING_PASSWORD': 'a38c18a9ac6a76a6d22f38feae7cc1e984dff094' 
        }
    )

    step_train = TrainingStep(
        name="BertTrainingStep",
        estimator=estimator,
        inputs={"training": TrainingInput(s3_data=input_data)}
    )

    # --- Крок 2: Створення моделі ---
    model = PyTorchModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        entry_point='inference.py',
        source_dir='./src',
        framework_version='1.13',
        py_version='py39',
        sagemaker_session=sess
    )

    step_create_model = CreateModelStep(
        name="BertCreateModelStep",
        model=model,
        inputs=sagemaker.inputs.CreateModelInput(instance_type="ml.m5.large")
    )

    # --- Крок 2.5: Перевірка Дрифту (Drift Check) ---
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        base_job_name="drift-check"
    )

    step_drift = ProcessingStep(
        name="DriftCheckStep",
        processor=sklearn_processor,
        code="./src/check_drift.py",
        # ФІКС: Додаємо вхідні дані (щоб скрипт мав що перевіряти)
        inputs=[
            ProcessingInput(
                source=input_data, # Беремо ті самі дані, що і для навчання
                destination="/opt/ml/processing/input" 
            )
        ],
        outputs=[
            ProcessingOutput(output_name="report", source="/opt/ml/processing/output")
        ]
    )

    # --- Крок 3: Реєстрація ---
    step_register = RegisterModel(
        name="BertRegisterStep",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status="PendingManualApproval"
    )
    
    # Додаємо залежність: Реєстрація чекає завершення Drift Check
    step_register.add_depends_on([step_drift])

    # --- ЗБИРАЄМО ПАЙПЛАЙН (Один раз!) ---
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[training_instance_type, input_data, epochs, batch_size],
        # Включаємо ВСІ кроки
        steps=[step_train, step_create_model, step_drift, step_register] 
    )
    
    return pipeline

if __name__ == "__main__":
    print("Building pipeline using Terraform resources...")
    
    import sys
    
    # Беремо роль з енвайронменту (GitHub) АБО використовуємо нову з Terraform як дефолт
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN", TERRAFORM_ROLE)
    
    print(f"Using Role: {role_arn}")
    print(f"Using Bucket: {TERRAFORM_BUCKET}")

    pipeline = get_pipeline(
        region=os.environ.get("AWS_REGION", "eu-north-1"),
        role=role_arn,
        default_bucket=TERRAFORM_BUCKET
    )
    
    print(f"Pipeline definition: {pipeline.name}")
    
    pipeline.upsert(role_arn=role_arn)
    print("Pipeline submitted/updated in SageMaker.")
    
    execution = pipeline.start()
    print(f"Pipeline execution started. Execution ARN: {execution.arn}")