"""
Automated Model Retraining Script
Triggers retraining based on drift detection or scheduled intervals
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import logging
from datetime import datetime
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.evaluate_model import save_metrics
from src.monitoring.drift_detector import DriftDetector
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def should_retrain(drift_report: dict, performance_threshold: float = 0.05) -> bool:
    """Determine if retraining is needed based on drift"""
    if drift_report["summary"]["overall_drift_detected"]:
        logger.info("Drift detected - retraining recommended")
        return True
    return False

def retrain_pipeline(data_path: str = "data/raw/customer_churn.csv"):
    """Execute automated retraining pipeline"""
    logger.info("=" * 60)
    logger.info("AUTOMATED RETRAINING PIPELINE STARTED")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Load new data
    df_new = load_data(data_path)
    
    # Check for drift
    try:
        df_reference = pd.read_csv("data/processed/data.csv")
        detector = DriftDetector(df_reference)
        drift_report = detector.generate_drift_report(df_new)
        detector.save_report(drift_report)
        
        if not should_retrain(drift_report):
            logger.info("No significant drift detected - skipping retraining")
            return
    except FileNotFoundError:
        logger.info("No reference data found - proceeding with training")
    
    # Preprocess
    df_processed = preprocess_data(df_new)
    
    # Feature engineering
    df_features = build_features(df_processed)
    
    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df_features.to_csv("data/processed/data.csv", index=False)
    logger.info("Processed data saved")
    
    # Train model
    model, f1, roc = train_model(df_features)
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/model_{timestamp}.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Also save as latest
    joblib.dump(model, "models/model.pkl")
    
    # Save metrics
    save_metrics(f1, roc)
    
    logger.info("=" * 60)
    logger.info("RETRAINING COMPLETED SUCCESSFULLY")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC-AUC: {roc:.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    retrain_pipeline()
