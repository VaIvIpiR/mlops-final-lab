import boto3
import sagemaker
import time
import os
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


ROLE_ARN = "arn:aws:iam::584360834542:role/mlops-lab-terraform-role" 
BUCKET_NAME = "mlops-lab-terraform-vaivipir-data"
REGION = "eu-north-1"

def get_latest_model_artifact():
    """Шукає останню модель, створену в S3"""
    s3 = boto3.client('s3', region_name=REGION)
    prefix = "NewsClassifier/training_jobs" 
    
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    
    if 'Contents' not in response:
        raise Exception("Не знайдено жодної моделі! Чи запускався пайплайн?")
        

    latest_obj = sorted(response['Contents'], key=lambda x: x['LastModified'])[-1]
    key = latest_obj['Key']
    

    all_models = [obj for obj in response['Contents'] if obj['Key'].endswith('model.tar.gz')]
    if not all_models:
         raise Exception("Архівів model.tar.gz не знайдено.")
         
    latest_model = sorted(all_models, key=lambda x: x['LastModified'])[-1]
    
    return f"s3://{BUCKET_NAME}/{latest_model['Key']}"

if __name__ == "__main__":
    print("🔍 Searching for latest model artifact...")
    model_data = get_latest_model_artifact()
    print(f"📦 Found: {model_data}")
    
    endpoint_name = f"load-test-{int(time.time())}"
    print(f"🚀 Deploying to endpoint: {endpoint_name}...")
    

    model = PyTorchModel(
        model_data=model_data,
        role=ROLE_ARN,
        entry_point='inference.py',
        source_dir='./src', 
        framework_version='1.13',
        py_version='py39',
        env={
            "TS_RESPONSE_TIMEOUT": "600", 
            "MMS_DEFAULT_RESPONSE_TIMEOUT": "600",
            "SAGEMAKER_MODEL_SERVER_TIMEOUT": "600"
        }
    )
    
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )
    
    print("\n" + "="*50)
    print(f"✅ Endpoint READY: {endpoint_name}")
    print("Now run locust!")
    print("="*50)