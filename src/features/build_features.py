import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced feature engineering for churn prediction
    Includes ML-based features for better model performance
    """
    df = df.copy()

    # 1. Number of services subscribed
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum([1 for val in row if val in ["Yes", "DSL", "Fiber optic"]]),
        axis=1
    )

    # 2. Contract type flag
    df["is_monthly_contract"] = df["Contract"].apply(
        lambda x: 1 if x == "Month-to-month" else 0
    )

    # 3. Average charge per tenure month
    df["avg_charge_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

    # 4. Tenure groups (categorical bucketing)
    def tenure_group(x):
        if x < 12:
            return "0-1yr"
        elif x < 24:
            return "1-2yr"
        elif x < 48:
            return "2-4yr"
        else:
            return "4+yr"

    df["tenure_group"] = df["tenure"].apply(tenure_group)

    # 5. Has support services
    df["has_support"] = df[["TechSupport", "OnlineSecurity"]].apply(
        lambda row: 1 if "Yes" in row.values else 0,
        axis=1
    )

    # 6. Charge increase rate (simulated)
    df["charge_increase_rate"] = (df["MonthlyCharges"] * df["tenure"]) / (df["TotalCharges"] + 1)

    # 7. Service density (services per dollar)
    df["service_density"] = df["num_services"] / (df["MonthlyCharges"] + 1)

    # 8. High value customer flag
    df["is_high_value"] = (df["TotalCharges"] > df["TotalCharges"].quantile(0.75)).astype(int)

    # 9. Payment method risk score
    payment_risk = {
        "Electronic check": 3,
        "Mailed check": 2,
        "Bank transfer (automatic)": 1,
        "Credit card (automatic)": 1
    }
    df["payment_risk_score"] = df["PaymentMethod"].map(payment_risk).fillna(2)

    # 10. Customer lifetime value estimate
    df["estimated_clv"] = df["MonthlyCharges"] * df["tenure"]

    logging.info("Advanced feature engineering completed with 10 features")
    return df
