import pandas as pd
import numpy as np
import os
import json
from scipy.stats import ks_2samp


if __name__ == "__main__":
    print("Starting Data Drift Check in the Cloud...")
    
    ref_data = np.random.normal(0.9, 0.05, 1000)
    curr_data = np.random.normal(0.6, 0.1, 1000) 
    
    stat, p_value = ks_2samp(ref_data, curr_data)
    is_drift = p_value < 0.05
    
    print(f"Statistical Result: P-value={p_value:.10f}")
    print(f"Drift Detected: {is_drift}")
    
    output_dir = "/opt/ml/processing/output"
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        "metric": "confidence_score",
        "p_value": p_value,
        "drift_detected": bool(is_drift),
        "status": "FAILED" if is_drift else "PASSED"
    }
    
    with open(os.path.join(output_dir, "drift_report.json"), "w") as f:
        json.dump(report, f)
        
    print(f"Report saved to {output_dir}/drift_report.json")