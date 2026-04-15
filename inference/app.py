from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load model from MLflow registry (production alias)
model = mlflow.pyfunc.load_model("models:/churn-model/Production")

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {
        "prediction": int(prediction[0]),
        "churn": "Yes" if prediction[0] == 1 else "No"
    }
