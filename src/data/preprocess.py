import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess dataset
    """
    df = df.copy()

    # Drop customerID (not useful)
    df.drop(columns=["customerID"], inplace=True)

    # Fix TotalCharges (it is string sometimes)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Handle missing values
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert target to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    logging.info("Preprocessing completed")
    return df
