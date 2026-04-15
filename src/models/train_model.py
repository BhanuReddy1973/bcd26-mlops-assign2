import pandas as pd
import logging
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)

def train_model(df: pd.DataFrame):
    df = df.copy()
    target = "Churn"
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])

        model = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        # Log metrics to MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="churn-model"
        )

        logging.info(f"F1 Score: {f1}")
        logging.info(f"ROC-AUC: {roc}")

    return model, f1, roc
