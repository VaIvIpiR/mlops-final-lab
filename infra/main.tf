provider "aws" {
  region = "eu-north-1" 
}


variable "project_name" {
  default = "mlops-lab-terraform"
}


resource "aws_s3_bucket" "ml_bucket" {
  bucket = "${var.project_name}-vaivipir-data" 
  
  tags = {
    Name        = "MLOps Lab Bucket"
    Environment = "Dev"
  }
}


data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution_role" {
  name               = "${var.project_name}-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "s3_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}


output "s3_bucket_name" {
  value = aws_s3_bucket.ml_bucket.bucket
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker_execution_role.arn
}



resource "aws_cloudwatch_metric_alarm" "sagemaker_errors" {
  alarm_name          = "${var.project_name}-error-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Invocation5XXErrors" 
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Trigger if SageMaker endpoint generates 5XX errors"
  treat_missing_data  = "notBreaching"


  dimensions = {
    EndpointName = "NewsClassifierEndpoint" 
    VariantName  = "AllTraffic"
  }
}




data "aws_caller_identity" "current" {}
data "aws_region" "current" {}


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


resource "aws_iam_policy" "eventbridge_policy" {
  name = "${var.project_name}-eventbridge-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sagemaker:StartPipelineExecution"
      Resource = "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/NewsClassifierPipeline"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eventbridge_attach" {
  role       = aws_iam_role.eventbridge_role.name
  policy_arn = aws_iam_policy.eventbridge_policy.arn
}


resource "aws_cloudwatch_event_rule" "pipeline_schedule" {
  name        = "${var.project_name}-weekly-trigger"
  description = "Triggers SageMaker Pipeline every Monday at 8:00 AM UTC"
  

  schedule_expression = "cron(0 8 ? * MON *)"
}



resource "aws_cloudwatch_event_target" "sagemaker_target" {
  rule      = aws_cloudwatch_event_rule.pipeline_schedule.name
  target_id = "TriggerSageMakerPipeline"
  arn       = "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:pipeline/NewsClassifierPipeline"
  role_arn  = aws_iam_role.eventbridge_role.arn
  
}