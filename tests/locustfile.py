import boto3
import json
import time
import os
from locust import User, task, between, events

# !!! СЮДИ ТРЕБА БУДЕ ВСТАВИТИ ІМ'Я ПІСЛЯ ЗАПУСКУ КРОКУ 1 !!!
# Або передати через змінну середовища: os.environ.get("ENDPOINT_NAME")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "REPLACE_ME_WITH_ENDPOINT_NAME") 
REGION = "eu-north-1"

class SageMakerUser(User):
    abstract = True
    wait_time = between(0.5, 2) # Пауза між запитами юзера

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = boto3.client("sagemaker-runtime", region_name=REGION)

    @task
    def predict(self):
        # Тестовий текст
        payload = "I want to reset my password immediately because it was stolen"
        
        start_time = time.time()
        try:
            response = self.client.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps(payload) # Відправляємо як JSON рядок
            )
            
            # Читаємо відповідь
            response_body = response['Body'].read().decode('utf-8')
            
            # Рахуємо час
            total_time = int((time.time() - start_time) * 1000)
            
            # Успіх!
            events.request.fire(
                request_type="sagemaker",
                name="invoke_endpoint",
                response_time=total_time,
                response_length=len(response_body),
            )
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="sagemaker",
                name="invoke_endpoint",
                response_time=total_time,
                exception=e,
            )

class WebsiteUser(SageMakerUser):
    pass