import json
import os

def save_metrics(f1, roc):
    os.makedirs("reports", exist_ok=True)
    metrics = {
        "F1 Score": f1,
        "ROC-AUC": roc
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
