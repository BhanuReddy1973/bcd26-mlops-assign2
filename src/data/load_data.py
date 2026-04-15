import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
