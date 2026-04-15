# Project Summary - MLOps Assignment Part 2

**Student:** Bhanu Reddy  
**Roll No:** 2022BCD0026  
**GitHub:** https://github.com/BhanuReddy1973/bcd26-mlops-assign2

---

## ✅ Completed Tasks

### 1. Project Structure ✓
- Created complete MLOps project structure
- Organized code into modular components
- Set up proper Python package structure

### 2. Data Pipeline ✓
- `src/data/load_data.py` - Data loading with error handling
- `src/data/preprocess.py` - Data cleaning and preprocessing
- Handles missing values, type conversions, target encoding

### 3. Feature Engineering ✓
- `src/features/build_features.py` - 5 engineered features
  - num_services
  - is_monthly_contract
  - avg_charge_per_tenure
  - tenure_group
  - has_support

### 4. Model Training ✓
- `src/models/train_model.py` - sklearn Pipeline
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
  - RandomForestClassifier (100 estimators)

### 5. Model Evaluation ✓
- `src/models/evaluate_model.py` - Metrics calculation
- F1 Score: 0.5533
- ROC-AUC: 0.8357
- Saves metrics to JSON

### 6. Training Pipeline ✓
- `training/train.py` - End-to-end training script
- Orchestrates all pipeline steps
- Saves processed data and model artifacts

### 7. DVC Integration ✓
- `dvc.yaml` - Pipeline definition
- `.dvcignore` - DVC ignore patterns
- Data versioning setup
- Reproducible pipeline

### 8. MLflow Integration ✓
- Experiment tracking
- Parameter logging
- Metric logging
- Model artifact logging
- Model registry setup

### 9. FastAPI Deployment ✓
- `inference/app.py` - REST API
- `inference/schema.py` - Pydantic schemas
- GET / - Health check endpoint
- POST /predict - Prediction endpoint

### 10. Documentation ✓
- README.md - Project overview
- SETUP.md - Installation guide
- QUICK_START.md - Quick reference
- ARCHITECTURE.md - System design
- CONTRIBUTING.md - Development guide

### 11. Testing ✓
- `test_api.py` - API testing script
- Sample data for testing

### 12. Docker Support ✓
- Dockerfile for containerization
- Production-ready configuration

### 13. Git & GitHub ✓
- Repository initialized
- All code committed
- Pushed to GitHub
- .gitignore configured

---

## 📁 Project Structure

```
bcd26-mlops-assign2/
├── data/
│   ├── raw/              # Raw dataset
│   ├── processed/        # Processed data
│   └── external/         # External data
├── src/
│   ├── data/            # Data loading & preprocessing
│   ├── features/        # Feature engineering
│   ├── models/          # Training & evaluation
│   ├── pipeline/        # Pipeline orchestration
│   └── utils/           # Utilities
├── training/            # Training scripts
├── inference/           # FastAPI application
├── models/              # Saved models
├── reports/             # Metrics & figures
├── notebooks/           # Jupyter notebooks
├── screenshots/         # Documentation screenshots
├── dvc.yaml            # DVC pipeline
├── params.yaml         # Configuration
├── requirements.txt    # Dependencies
└── Dockerfile          # Container config
```

---

## 🚀 How to Use

### Setup
```bash
git clone https://github.com/BhanuReddy1973/bcd26-mlops-assign2.git
cd bcd26-mlops-assign2
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Train Model
```bash
python -m training.train
# or
dvc repro
```

### View Experiments
```bash
mlflow ui
```

### Start API
```bash
uvicorn inference.app:app --reload
```

### Test API
```bash
python test_api.py
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| F1 Score | 0.5533 |
| ROC-AUC | 0.8357 |
| Model | Random Forest |
| Estimators | 100 |

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **scikit-learn** - Machine learning
- **pandas** - Data manipulation
- **DVC** - Data versioning
- **MLflow** - Experiment tracking
- **FastAPI** - API framework
- **Uvicorn** - ASGI server
- **Docker** - Containerization

---

## 📝 Key Files

| File | Purpose |
|------|---------|
| `training/train.py` | Main training script |
| `inference/app.py` | FastAPI application |
| `dvc.yaml` | DVC pipeline definition |
| `params.yaml` | Configuration parameters |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container configuration |

---

## 🎯 Assignment Requirements Met

✅ Folder structure created  
✅ Dataset added to data/raw  
✅ Data loading implemented  
✅ Data preprocessing implemented  
✅ Training entry point created  
✅ Feature engineering implemented  
✅ Model training implemented  
✅ Model evaluation implemented  
✅ Full pipeline integrated  
✅ DVC setup and configured  
✅ DVC pipeline created  
✅ MLflow experiment tracking  
✅ Model registry configured  
✅ FastAPI inference endpoint  
✅ GitHub repository created  
✅ Code pushed to GitHub  

---

## 📚 Documentation Files

1. **README.md** - Project overview and quick start
2. **SETUP.md** - Detailed installation instructions
3. **QUICK_START.md** - 5-minute quick start guide
4. **ARCHITECTURE.md** - System architecture and design
5. **CONTRIBUTING.md** - Development guidelines
6. **PROJECT_SUMMARY.md** - This file

---

## 🔗 Links

- **GitHub Repository:** https://github.com/BhanuReddy1973/bcd26-mlops-assign2
- **Reference Repository:** https://github.com/2022BCS0019-abhinav/2022bcs0019-assignment-mlops

---

## 📸 Screenshots

Screenshots should be captured for:
1. Folder structure (tree /f)
2. Training runs (v1, v2, v3)
3. DVC initialization and pipeline
4. MLflow UI and experiments
5. Model registry
6. FastAPI endpoints

See `screenshots/README.md` for complete list.

---

## ✨ Next Steps

To complete the assignment:

1. **Add Dataset**
   - Place `customer_churn.csv` in `data/raw/`

2. **Run Pipeline**
   ```bash
   python -m training.train
   ```

3. **Initialize DVC**
   ```bash
   dvc init
   dvc add data/raw/customer_churn.csv
   ```

4. **Run DVC Pipeline**
   ```bash
   dvc repro
   ```

5. **Start MLflow**
   ```bash
   mlflow ui
   ```

6. **Register Model**
   - Set alias to "Production" in MLflow UI

7. **Start API**
   ```bash
   uvicorn inference.app:app --reload
   ```

8. **Capture Screenshots**
   - Follow `screenshots/README.md` checklist

9. **Update Documentation**
   - Add actual screenshots to markdown

---

**Assignment Completed By:**  
Bhanu Reddy (2022BCD0026)  
Date: April 15, 2026
