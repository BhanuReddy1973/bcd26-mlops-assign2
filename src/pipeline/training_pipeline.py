"""
Training Pipeline Module
Orchestrates the complete training workflow
"""
import logging
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def run(self):
        """Execute the complete training pipeline"""
        logger.info("Starting training pipeline...")
        
        # Load data
        df = load_data(self.data_path)
        
        # Preprocess
        df = preprocess_data(df)
        
        # Feature engineering
        df = build_features(df)
        
        # Train model
        model, f1, roc = train_model(df)
        
        logger.info("Training pipeline completed successfully!")
        return model, f1, roc
