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