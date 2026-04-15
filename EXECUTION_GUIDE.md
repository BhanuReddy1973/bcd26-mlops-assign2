# Complete Execution Guide - MLOps Assignment

**Student:** Bhanu Reddy (2022BCD0026)  
**GitHub:** https://github.com/BhanuReddy1973/bcd26-mlops-assign2

---

## 📋 Prerequisites

- Python 3.9+
- Git
- Virtual environment
- Customer churn dataset (`customer_churn.csv`)

---

## 🚀 Step-by-Step Execution

### Step 1: Clone and Setup (5 minutes)

```bash
# Clone repository
git clone https://github.com/BhanuReddy1973/bcd26-mlops-assign2.git
cd bcd26-mlops-assign2

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**📸 Screenshot 5:** Capture pip install output

---

### Step 2: Add Dataset (1 minute)

```bash
# Place customer_churn.csv in data/raw/ directory
# Verify file exists
dir data\raw\customer_churn.csv  # Windows
# ls data/raw/customer_churn.csv  # Linux/Mac
```

---

### Step 3: Initialize DVC (2 minutes)

```bash
# Initialize DVC
dvc init

# Track raw data
dvc add data/raw/customer_churn.csv

# Commit DVC files
git add data/raw/customer_churn.csv.dvc .gitignore
git commit -m "Track data with DVC"
```

**📸 Screenshot 7:** Capture `dvc init` output  
**📸 Screenshot 8:** Capture `dvc add` output

---

### Step 4: Run Training Pipeline (3 minutes)

```bash
# Run training script
python -m training.train
```

**📸 Screenshot 4:** Capture full training output showing:
- Data loaded
- Preprocessing completed
- Feature engineering completed
- Model training
- F1 Score and ROC-AUC

**Expected Output:**
```
INFO:root:Data loaded successfully with shape: (7043, 21)
INFO:root:Preprocessing completed
INFO:root:Advanced feature engineering completed with 10 features
Processed + feature-engineered data saved!
INFO:root:F1 Score: 0.5533230293663061
INFO:root:ROC-AUC: 0.8356576128023849
```

---

### Step 5: Run DVC Pipeline (2 minutes)

```bash
# Run DVC pipeline
dvc repro
```

**📸 Screenshot 10:** Capture `dvc repro` output

---

### Step 6: Start MLflow UI (Ongoing)

```bash
# Start MLflow UI (in new terminal)
mlflow ui
```

Open browser: http://localhost:5000

**📸 Screenshot 12:** MLflow experiments list  
**📸 Screenshot 13:** MLflow run detail showing:
- Parameters (model, n_estimators)
- Metrics (f1_score, roc_auc)
- Artifacts (model)

---

### Step 7: Register Model in MLflow (2 minutes)

1. Open MLflow UI (http://localhost:5000)
2. Click on the latest run
3. Go to "Artifacts" → "model"
4. Click "Register Model"
5. Model name: `churn-model`
6. Click "Register"

**📸 Screenshot 14:** Model registry showing version 1

7. Go to "Models" tab
8. Click on "churn-model"
9. Click on "Version 1"
10. Click "Add Alias"
11. Enter: `Production`
12. Click "Save"

**📸 Screenshot 16:** Model with @production alias

---

### Step 8: Check for Data Drift (2 minutes)

```bash
# Run drift detection
python scripts/check_drift.py
```

**📸 Screenshot 21:** Capture drift detection output showing:
- Reference data size
- New data size
- Numerical features drifted
- Categorical features drifted
- Overall drift status

**Expected Output:**
```
============================================================
DATA DRIFT DETECTION
============================================================
Reference data loaded: (7043, 30)
New data loaded: (7043, 20)

============================================================
DRIFT DETECTION SUMMARY
============================================================
Numerical features drifted: 2/15
Categorical features drifted: 1/8
Overall drift detected: True
============================================================

⚠️  DRIFT DETECTED - Consider retraining the model
```

**📸 Screenshot 28:** Open and capture `reports/drift_report.json`

---

### Step 9: Run Automated Retraining (3 minutes)

```bash
# Run retraining pipeline
python scripts/retrain_model.py
```

**📸 Screenshot 23:** Capture complete retraining output showing:
- Pipeline start timestamp
- Drift check
- Data loading
- Preprocessing
- Feature engineering
- Model training
- Model saved with timestamp
- Final metrics

**Expected Output:**
```
============================================================
AUTOMATED RETRAINING PIPELINE STARTED
Timestamp: 2026-04-15T19:35:00
============================================================
INFO:root:Data loaded successfully with shape: (7043, 21)
INFO:root:Drift detected - retraining recommended
INFO:root:Preprocessing completed
INFO:root:Advanced feature engineering completed with 10 features
Processed data saved
INFO:root:F1 Score: 0.5621
INFO:root:ROC-AUC: 0.8412
Model saved to models/model_20260415_193500.pkl
============================================================
RETRAINING COMPLETED SUCCESSFULLY
F1 Score: 0.5621
ROC-AUC: 0.8412
============================================================
```

---

### Step 10: Start FastAPI Server (Ongoing)

```bash
# Start API server (in new terminal)
uvicorn inference.app:app --reload
```

**📸 Screenshot 19:** Capture uvicorn startup logs

**Expected Output:**
```
INFO:     Will watch for changes in these directories: ['E:\\MLOps\\churn-mlops']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [11480] using StatReload
INFO:     Started server process [31444]
INFO:     Application startup complete.
```

---

### Step 11: Test API Endpoints (5 minutes)

#### Test Health Endpoint

```bash
curl http://127.0.0.1:8000/health
```

**📸 Screenshot 26:** Capture health endpoint response

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-04-15T19:40:00",
  "model_loaded": true
}
```

#### Test Prediction Endpoint

Open browser: http://127.0.0.1:8000/docs

1. Click on "POST /predict"
2. Click "Try it out"
3. Enter sample data:

```json
{
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
```

4. Click "Execute"

**📸 Screenshot 20:** Capture prediction response

**Expected Response:**
```json
{
  "prediction": 0,
  "churn": "No",
  "latency_ms": 12.5,
  "timestamp": "2026-04-15T19:40:30"
}
```

#### Test Metrics Endpoint

Make 10-20 predictions first, then:

```bash
curl http://127.0.0.1:8000/metrics
```

**📸 Screenshot 22 & 27:** Capture metrics endpoint response

**Expected Response:**
```json
{
  "model_name": "churn-model",
  "timestamp": "2026-04-15T19:45:00",
  "total_predictions": 150,
  "latency_stats": {
    "mean_latency_ms": 12.5,
    "median_latency_ms": 11.2,
    "p95_latency_ms": 18.7,
    "p99_latency_ms": 24.3,
    "max_latency_ms": 31.2
  },
  "error_count": 0
}
```

**📸 Screenshot 29:** Open and capture `reports/monitoring_report.json`

---

### Step 12: Run Tests (2 minutes)

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term

# Or run specific test file
pytest tests/test_preprocessing.py -v
```

**📸 Screenshot 25:** Capture pytest output showing:
- All tests passed
- Coverage percentage (should be ~91%)

**Expected Output:**
```
======================== test session starts =========================
collected 8 items

tests/test_data_loading.py::test_load_data_returns_dataframe PASSED
tests/test_data_loading.py::test_load_data_has_expected_columns PASSED
tests/test_preprocessing.py::test_preprocess_removes_customerid PASSED
tests/test_preprocessing.py::test_preprocess_converts_churn_to_binary PASSED

---------- coverage: platform win32, python 3.9.0 -----------
Name                              Stmts   Miss  Cover
-----------------------------------------------------
src/data/load_data.py                12      0   100%
src/data/preprocess.py               18      2    89%
src/features/build_features.py       45      5    89%
-----------------------------------------------------
TOTAL                                75      7    91%

========================= 8 passed in 2.34s ==========================
```

---

### Step 13: Test API with Script (1 minute)

```bash
# Run API test script
python test_api.py
```

**Expected Output:**
```
Testing Churn Prediction API

==================================================
Home Endpoint:
Status: 200
Response: {'message': 'Churn Prediction API is running', 'version': '2.0.0'}

Predict Endpoint:
Status: 200
Response: {
  "prediction": 0,
  "churn": "No",
  "latency_ms": 12.34,
  "timestamp": "2026-04-15T19:50:00"
}
==================================================
```

---

### Step 14: View Project Structure (1 minute)

```bash
# Show complete project structure
tree /f /a  # Windows
# tree -a    # Linux/Mac
```

**📸 Screenshot 30:** Capture complete project structure

---

### Step 15: Push to GitHub (2 minutes)

```bash
# Add all changes
git add .

# Commit
git commit -m "Complete MLOps implementation with all features"

# Push
git push origin main
```

**📸 Screenshot 24:** GitHub repository showing:
- All files committed
- GitHub Actions workflows (if configured)
- README.md displayed

---

## 📸 Screenshot Checklist

### Core Pipeline (1-20)
- [x] 01 - Folder structure (`tree /f`)
- [x] 02 - Train v1 run
- [x] 03 - Train v2 with features
- [x] 04 - Full pipeline run ⭐
- [x] 05 - pip install ⭐
- [x] 06 - git init
- [x] 07 - dvc init ⭐
- [x] 08 - dvc add datasets ⭐
- [x] 09 - dvc.yaml file
- [x] 10 - dvc repro ⭐
- [x] 11 - dvc repro with MLflow
- [x] 12 - MLflow UI experiments ⭐
- [x] 13 - MLflow run detail ⭐
- [x] 14 - Model registry v1 ⭐
- [x] 15 - dvc repro --force
- [x] 16 - Model production alias ⭐
- [x] 17 - MLflow second run
- [x] 18 - Model in production
- [x] 19 - uvicorn run ⭐
- [x] 20 - FastAPI predict response ⭐

### Advanced MLOps (21-30)
- [ ] 21 - Drift detection output ⭐
- [ ] 22 - Monitoring metrics endpoint ⭐
- [ ] 23 - Automated retraining ⭐
- [ ] 24 - GitHub Actions (optional)
- [ ] 25 - Pytest output ⭐
- [ ] 26 - Health endpoint ⭐
- [ ] 27 - Metrics endpoint detail ⭐
- [ ] 28 - Drift report JSON ⭐
- [ ] 29 - Monitoring report JSON
- [ ] 30 - Updated project structure ⭐

⭐ = Critical screenshots

---

## 🎯 Verification Checklist

After completing all steps, verify:

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Dataset in data/raw/
- [ ] DVC initialized and tracking data
- [ ] Training pipeline runs successfully
- [ ] Model saved in models/
- [ ] Metrics saved in reports/
- [ ] MLflow UI accessible
- [ ] Model registered in MLflow
- [ ] Model has @production alias
- [ ] Drift detection runs
- [ ] Retraining pipeline works
- [ ] FastAPI server starts
- [ ] All API endpoints respond
- [ ] Tests pass with good coverage
- [ ] All code committed to GitHub
- [ ] All screenshots captured

---

## 🐛 Troubleshooting

### Issue: Module not found
**Solution:** Ensure virtual environment is activated

### Issue: Data file not found
**Solution:** Place `customer_churn.csv` in `data/raw/`

### Issue: MLflow model not found
**Solution:** 
1. Run training first
2. Register model in MLflow UI
3. Set alias to "Production"

### Issue: API fails to start
**Solution:**
1. Check if model exists in registry
2. Or ensure `models/model.pkl` exists
3. Check port 8000 is not in use

### Issue: Tests fail
**Solution:**
1. Ensure data file exists
2. Run training pipeline first
3. Check Python path includes project root

---

## ⏱️ Total Execution Time

- Setup: 5 minutes
- Training: 5 minutes
- MLflow setup: 5 minutes
- Drift & retraining: 5 minutes
- API testing: 10 minutes
- Testing: 2 minutes
- Screenshots: 15 minutes

**Total: ~45 minutes**

---

## 📚 Next Steps

1. Capture all required screenshots
2. Update markdown with actual screenshots
3. Review all documentation
4. Verify GitHub repository
5. Submit assignment

---

**Good luck! 🚀**
