"""
Inference Pipeline Module
Handles prediction preprocessing and postprocessing
"""
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferencePipeline:
    def __init__(self, model):
        self.model = model
        
    def preprocess_input(self, data: dict) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        df = pd.DataFrame([data])
        return df
    
    def predict(self, data: dict) -> dict:
        """Make prediction on input data"""
        df = self.preprocess_input(data)
        prediction = self.model.predict(df)
        
        result = {
            "prediction": int(prediction[0]),
            "churn": "Yes" if prediction[0] == 1 else "No",
            "confidence": "High" if prediction[0] in [0, 1] else "Low"
        }
        
        logger.info(f"Prediction made: {result}")
        return result
