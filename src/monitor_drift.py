import pandas as pd
import os
from sklearn import datasets

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

def run_monitoring():
    print("🕵️‍♂️ Starting Drift Detection...")

    # 1. Імітуємо завантаження даних
    # У реальності ти б тягнув це з S3: train.csv (Reference) і logs.csv (Current)
    print("📦 Loading data...")
    
    # Для прикладу створимо штучні дані на основі твого завдання (текстова класифікація)
    # Reference (на чому вчили)
    reference_data = pd.DataFrame({
        'text_length': [10, 15, 12, 10, 50, 45, 12, 11],
        'confidence': [0.9, 0.8, 0.95, 0.85, 0.9, 0.88, 0.92, 0.87],
        'label': ['billing', 'support', 'billing', 'billing', 'tech', 'tech', 'billing', 'billing']
    })

    # Current (що приходить зараз - імітуємо проблему/дрифт)
    # Наприклад, тексти стали дуже довгими, а впевненість впала
    current_data = pd.DataFrame({
        'text_length': [100, 120, 90, 110, 50, 45, 12, 11], # Дрифт довжини!
        'confidence': [0.5, 0.4, 0.45, 0.55, 0.9, 0.88, 0.92, 0.87], # Падіння впевненості!
        'label': ['unknown', 'unknown', 'unknown', 'unknown', 'tech', 'tech', 'billing', 'billing']
    })

    print("📊 Generating Evidently Report...")
    
    # 2. Створюємо звіт
    data_drift_report = Report(metrics=[
        DataDriftPreset(),   # Перевірка зміни розподілу даних
        DataQualityPreset()  # Перевірка якості (пропуски, типи)
    ])

    data_drift_report.run(reference_data=reference_data, current_data=current_data)

    # 3. Зберігаємо результат
    output_path = "drift_report.html"
    data_drift_report.save_html(output_path)
    
    print(f"✅ Report saved to {output_path}")
    print("   Open this file in browser to see the magic!")

if __name__ == "__main__":
    run_monitoring()