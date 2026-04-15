from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
import time
import logging
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.model_monitor import ModelMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="ML-powered customer churn prediction with monitoring",
    version="2.0.0"
)

# Initialize monitoring
monitor = ModelMonitor(model_name="churn-model")

# Load model from MLflow registry (production alias)
try:
    model = mlflow.pyfunc.load_model("models:/churn-model/Production")
    logger.info("Model loaded from MLflow registry")
except Exception as e:
    logger.warning(f"Could not load from registry: {e}. Loading from file.")
    import joblib
    model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {
        "message": "Churn Prediction API is running",
        "version": "2.0.0",
        "model": "churn-model",
        "status": "healthy"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(data: dict):
    """Predict churn with monitoring"""
    start_time = time.time()
    
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        latency = time.time() - start_time
        
        # Log prediction for monitoring
        monitor.log_prediction(
            features=data,
            prediction=int(prediction[0]),
            latency=latency
        )
        
        result = {
            "prediction": int(prediction[0]),
            "churn": "Yes" if prediction[0] == 1 else "No",
            "latency_ms": round(latency * 1000, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: {result['churn']} (latency: {result['latency_ms']}ms)")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """Get monitoring metrics"""
    report = monitor.generate_monitoring_report()
    return report
