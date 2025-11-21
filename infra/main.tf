provider "aws" {
  region = "eu-north-1" # Твій регіон (Стокгольм)
}

# --- 1. ЗМІННІ (Щоб було гарно) ---
variable "project_name" {
  default = "mlops-lab-terraform"
}

# --- 2. S3 BUCKET (Сховище) ---
resource "aws_s3_bucket" "ml_bucket" {
  # Ім'я має бути унікальним у всьому світі! Додаємо рандомні цифри або свій нік.
  bucket = "${var.project_name}-vaivipir-data" 
  
  tags = {
    Name        = "MLOps Lab Bucket"
    Environment = "Dev"
  }
}

# --- 3. IAM ROLE (Права доступу) ---
# Спочатку описуємо, ХТО може брати цю роль (Trust Policy)
data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

# Створюємо саму роль
resource "aws_iam_role" "sagemaker_execution_role" {
  name               = "${var.project_name}-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json
}

# Додаємо до ролі права "SageMakerFullAccess"
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Додаємо права "S3FullAccess" (або краще - тільки до нашого бакета, але для лаби S3Full ок)
resource "aws_iam_role_policy_attachment" "s3_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# --- 4. OUTPUTS (Що ми отримали) ---
# Це виведеться в консоль після створення
output "s3_bucket_name" {
  value = aws_s3_bucket.ml_bucket.bucket
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker_execution_role.arn
}


# --- 5. MONITORING (CloudWatch Alarm) ---
# Цей ресурс створює моніторинг для Endpoint'а.
# Він спрацює, якщо сервер видасть більше 1 помилки (Inference Error) за 5 хвилин.

resource "aws_cloudwatch_metric_alarm" "sagemaker_errors" {
  alarm_name          = "${var.project_name}-error-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Invocation5XXErrors" # Помилки сервера
  namespace           = "AWS/SageMaker"
  period              = 300 # 5 хвилин
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Trigger if SageMaker endpoint generates 5XX errors"
  treat_missing_data  = "notBreaching"

  # Щоб прив'язати до конкретного ендпоінта, нам треба знати його ім'я.
  # Оскільки ім'я динамічне (генерується пайплайном), ми ставимо тут
  # узагальнений моніторинг або припускаємо фіксоване ім'я варіанту.
  dimensions = {
    EndpointName = "NewsClassifierEndpoint" # Це ім'я має збігатися з тим, що створить пайплайн
    VariantName  = "AllTraffic"
  }
}


# --- 6. SCHEDULING (EventBridge) ---

# 1. Отримуємо ID твого акаунту (потрібно для формування ARN)
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# 2. Роль для EventBridge (Будильника), щоб він міг запускати пайплайн
resource "aws_iam_role" "eventbridge_role" {
  name = "${var.project_name}-eventbridge-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "events.amazonaws.com" }
    }]
  })
}

# Додаємо право "StartPipelineExecution"
resource "aws_iam_policy" "eventbridge_policy" {
  name = "${var.project_name}-eventbridge-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sagemaker:StartPipelineExecution"
      # ФІКС: Додали регіон у ARN
      Resource = "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/NewsClassifierPipeline"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eventbridge_attach" {
  role       = aws_iam_role.eventbridge_role.name
  policy_arn = aws_iam_policy.eventbridge_policy.arn
}

# 3. Сама подія (Розклад)
resource "aws_cloudwatch_event_rule" "pipeline_schedule" {
  name        = "${var.project_name}-weekly-trigger"
  description = "Triggers SageMaker Pipeline every Monday at 8:00 AM UTC"
  
  # CRON вираз: (Хвилини Години ДеньМісяця Місяць ДеньТижня Рік)
  # 0 8 ? * MON * -> Кожного понеділка о 8:00
  schedule_expression = "cron(0 8 ? * MON *)"
}


# 4. Ціль (Target) - Спрощена версія
resource "aws_cloudwatch_event_target" "sagemaker_target" {
  rule      = aws_cloudwatch_event_rule.pipeline_schedule.name
  target_id = "TriggerSageMakerPipeline"
  # ФІКС: Додали регіон у ARN
  arn       = "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/NewsClassifierPipeline"
  role_arn  = aws_iam_role.eventbridge_role.arn
  
  # Параметри прибрали, як і домовлялися
}