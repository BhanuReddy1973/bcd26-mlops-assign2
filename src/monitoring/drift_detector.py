"""
Data Drift Detection Module
Monitors feature and concept drift in production
"""
import pandas as pd
import numpy as np
import logging
from scipy import stats
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetector:
    """Detect data drift using statistical tests"""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        self.reference_data = reference_data
        self.threshold = threshold
        self.drift_report = {}
        
    def detect_feature_drift(self, current_data: pd.DataFrame) -> dict:
        """
        Detect drift in numerical features using KS test
        """
        drift_detected = {}
        
        numerical_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in self.reference_data.columns:
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                
                drift_detected[col] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "drift": p_value < self.threshold
                }
                
                if p_value < self.threshold:
                    logger.warning(f"Drift detected in {col}: p-value={p_value:.4f}")
        
        return drift_detected
    
    def detect_categorical_drift(self, current_data: pd.DataFrame) -> dict:
        """
        Detect drift in categorical features using Chi-square test
        """
        drift_detected = {}
        
        categorical_cols = current_data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in self.reference_data.columns:
                # Get value counts
                ref_counts = self.reference_data[col].value_counts()
                curr_counts = current_data[col].value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]
                
                # Chi-square test
                if sum(ref_freq) > 0 and sum(curr_freq) > 0:
                    statistic, p_value = stats.chisquare(curr_freq, ref_freq)
                    
                    drift_detected[col] = {
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "drift": p_value < self.threshold
                    }
                    
                    if p_value < self.threshold:
                        logger.warning(f"Drift detected in {col}: p-value={p_value:.4f}")
        
        return drift_detected
    
    def generate_drift_report(self, current_data: pd.DataFrame) -> dict:
        """Generate comprehensive drift report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "reference_size": len(self.reference_data),
            "current_size": len(current_data),
            "numerical_drift": self.detect_feature_drift(current_data),
            "categorical_drift": self.detect_categorical_drift(current_data),
        }
        
        # Count drifted features
        num_drifted = sum(1 for v in report["numerical_drift"].values() if v["drift"])
        cat_drifted = sum(1 for v in report["categorical_drift"].values() if v["drift"])
        
        report["summary"] = {
            "total_numerical_features": len(report["numerical_drift"]),
            "drifted_numerical_features": num_drifted,
            "total_categorical_features": len(report["categorical_drift"]),
            "drifted_categorical_features": cat_drifted,
            "overall_drift_detected": (num_drifted + cat_drifted) > 0
        }
        
        logger.info(f"Drift report: {num_drifted + cat_drifted} features drifted")
        return report
    
    def save_report(self, report: dict, filepath: str = "reports/drift_report.json"):
        """Save drift report to file"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)
        logger.info(f"Drift report saved to {filepath}")
