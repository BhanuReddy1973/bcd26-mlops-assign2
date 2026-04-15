import os
import pandas as pd
import joblib
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.evaluate_model import save_metrics

DATA_PATH = "data/raw/customer_churn.csv"
OUTPUT_PATH = "data/processed/data.csv"

def main():
    # Load
    df = load_data(DATA_PATH)

    # Preprocess
    df = preprocess_data(df)

    # Feature Engineering
    df = build_features(df)

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Processed + feature-engineered data saved!")

    # Train model
    model, f1, roc = train_model(df)

    # Save model artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # Save evaluation metrics
    save_metrics(f1, roc)

if __name__ == "__main__":
    main()
