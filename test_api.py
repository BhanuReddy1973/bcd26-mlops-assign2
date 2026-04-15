"""
API Testing Script
Test the FastAPI inference endpoint
"""
import requests
import json

# API endpoint
BASE_URL = "http://127.0.0.1:8000"

# Sample customer data
sample_data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 840.0
}

def test_home():
    """Test home endpoint"""
    response = requests.get(f"{BASE_URL}/")
    print("Home Endpoint:")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_predict():
    """Test prediction endpoint"""
    response = requests.post(
        f"{BASE_URL}/predict",
        json=sample_data
    )
    print("Predict Endpoint:")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

if __name__ == "__main__":
    print("Testing Churn Prediction API\n")
    print("="*50)
    test_home()
    test_predict()
    print("="*50)
