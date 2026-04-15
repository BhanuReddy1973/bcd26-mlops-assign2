"""
Model Performance Monitoring Module
Tracks model performance metrics in production
"""
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor model performance and system metrics"""
    
    def __init__(self, model_name: str = "churn-model"):
        self.model_name = model_name
        self.predictions = []
        self.latencies = []
        self.errors = []
        
    def log_prediction(self, features: dict, prediction: int, 
                      latency: float, actual: int = None):
        """Log individual prediction with metadata"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "latency_ms": latency * 1000,
            "actual": actual
        }
        self.predictions.append(log_entry)
        self.latencies.append(latency)
        
    def calculate_metrics(self, y_true: list, y_pred: list) -> dict:
        """Calculate performance metrics"""
        metrics = {
            "f1_score": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "accuracy": np.mean(np.array(y_true) == np.array(y_pred))
        }
        return metrics
    
    def get_latency_stats(self) -> dict:
        """Get latency statistics"""
        if not self.latencies:
            return {}
        
        latencies_ms = [l * 1000 for l in self.latencies]
        return {
            "mean_latency_ms": np.mean(latencies_ms),
            "median_latency_ms": np.median(latencies_ms),
            "p95_latency_ms": np.percentile(latencies_ms, 95),
            "p99_latency_ms": np.percentile(latencies_ms, 99),
            "max_latency_ms": np.max(latencies_ms)
        }
    
    def detect_performance_degradation(self, current_f1: float, 
                                      baseline_f1: float, 
                                      threshold: float = 0.05) -> bool:
        """Detect if model performance has degraded"""
        degradation = baseline_f1 - current_f1
        if degradation > threshold:
            logger.warning(f"Performance degradation detected: {degradation:.4f}")
            return True
        return False
    
    def generate_monitoring_report(self) -> dict:
        """Generate comprehensive monitoring report"""
        report = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "total_predictions": len(self.predictions),
            "latency_stats": self.get_latency_stats(),
            "error_count": len(self.errors)
        }
        
        # Calculate metrics if actuals are available
        actuals = [p["actual"] for p in self.predictions if p["actual"] is not None]
        preds = [p["prediction"] for p in self.predictions if p["actual"] is not None]
        
        if len(actuals) > 0:
            report["performance_metrics"] = self.calculate_metrics(actuals, preds)
        
        return report
    
    def save_report(self, report: dict, filepath: str = "reports/monitoring_report.json"):
        """Save monitoring report to file"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)
        logger.info(f"Monitoring report saved to {filepath}")
