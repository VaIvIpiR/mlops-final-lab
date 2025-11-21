import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import datetime

def generate_simple_html_report(drift_results, output_file="drift_report.html"):
    """Генерує простий HTML звіт без важких бібліотек"""
    html = f"""
    <html>
    <head><title>Simple Drift Report</title></head>
    <body style="font-family: sans-serif; padding: 20px;">
        <h1>📉 Data Drift Report</h1>
        <p>Generated at: {datetime.datetime.now()}</p>
        <table border="1" style="border-collapse: collapse; width: 50%;">
            <tr style="background-color: #f2f2f2;">
                <th style="padding: 8px;">Feature</th>
                <th style="padding: 8px;">Drift Detected?</th>
                <th style="padding: 8px;">P-Value</th>
            </tr>
    """
    
    for feature, result in drift_results.items():
        color = "#ffcccc" if result['drift'] else "#ccffcc" # Червоний якщо дрифт, зелений якщо ні
        status = "YES 🚨" if result['drift'] else "NO ✅"
        html += f"""
            <tr style="background-color: {color};">
                <td style="padding: 8px;">{feature}</td>
                <td style="padding: 8px;">{status}</td>
                <td style="padding: 8px;">{result['p_value']:.5f}</td>
            </tr>
        """
    
    html += "</table></body></html>"
    
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(html)
    print(f"✅ Report saved to {output_file}")

def check_drift():
    print("🚀 Starting Custom Drift Detection...")
    
    # 1. Імітація даних (Reference vs Current)
    # Reference: Дані навчання
    ref_data = pd.DataFrame({
        'confidence': np.random.normal(0.9, 0.05, 1000), # Нормальний розподіл
        'text_length': np.random.randint(10, 50, 1000)
    })
    
    # Current: Нові дані (З ДРИФТОМ)
    curr_data = pd.DataFrame({
        'confidence': np.random.normal(0.6, 0.1, 1000),  # Впевненість впала -> ДРИФТ!
        'text_length': np.random.randint(10, 50, 1000)   # Довжина така сама -> НЕМАЄ ДРИФТУ
    })
    
    results = {}
    
    # 2. Математика (KS Test)
    # Якщо p_value < 0.05, значить розподіли різні (дрифт є)
    for col in ref_data.columns:
        stat, p_value = ks_2samp(ref_data[col], curr_data[col])
        is_drift = p_value < 0.05
        
        results[col] = {
            'p_value': p_value,
            'drift': is_drift
        }
        print(f"Feature '{col}': P-value={p_value:.5f} -> Drift: {is_drift}")

    # 3. Генерація звіту
    generate_simple_html_report(results)

if __name__ == "__main__":
    check_drift()