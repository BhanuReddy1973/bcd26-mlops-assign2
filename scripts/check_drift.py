"""
Data Drift Checking Script
Run this to check for drift in new data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.monitoring.drift_detector import DriftDetector
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data

def main():
    print("=" * 60)
    print("DATA DRIFT DETECTION")
    print("=" * 60)
    
    # Load reference data
    df_reference = pd.read_csv("data/processed/data.csv")
    print(f"Reference data loaded: {df_reference.shape}")
    
    # Load new data
    df_new = load_data("data/raw/customer_churn.csv")
    df_new = preprocess_data(df_new)
    print(f"New data loaded: {df_new.shape}")
    
    # Initialize detector
    detector = DriftDetector(df_reference, threshold=0.05)
    
    # Generate report
    report = detector.generate_drift_report(df_new)
    
    # Save report
    detector.save_report(report)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DRIFT DETECTION SUMMARY")
    print("=" * 60)
    print(f"Numerical features drifted: {report['summary']['drifted_numerical_features']}/{report['summary']['total_numerical_features']}")
    print(f"Categorical features drifted: {report['summary']['drifted_categorical_features']}/{report['summary']['total_categorical_features']}")
    print(f"Overall drift detected: {report['summary']['overall_drift_detected']}")
    print("=" * 60)
    
    if report['summary']['overall_drift_detected']:
        print("\n⚠️  DRIFT DETECTED - Consider retraining the model")
    else:
        print("\n✅ No significant drift detected")

if __name__ == "__main__":
    main()
