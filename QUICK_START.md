# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Clone and Setup (2 min)

```bash
git clone https://github.com/BhanuReddy1973/bcd26-mlops-assign2.git
cd bcd26-mlops-assign2
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Add Dataset (1 min)

Place `customer_churn.csv` in `data/raw/` directory.

### Step 3: Train Model (1 min)

```bash
python -m training.train
```

### Step 4: Start API (1 min)

```bash
uvicorn inference.app:app --reload
```

Visit: http://127.0.0.1:8000/docs

## 📊 View Experiments

```bash
mlflow ui
```

Visit: http://localhost:5000

## 🧪 Test API

```bash
python test_api.py
```

## 📝 Key Commands

| Task | Command |
|------|---------|
| Train model | `python -m training.train` |
| Run DVC pipeline | `dvc repro` |
| Start MLflow UI | `mlflow ui` |
| Start API | `uvicorn inference.app:app --reload` |
| Test API | `python test_api.py` |

## 🎯 Expected Results

- F1 Score: ~0.55
- ROC-AUC: ~0.84
- Model: Random Forest (100 estimators)

## 📁 Important Files

- `training/train.py` - Main training script
- `inference/app.py` - FastAPI application
- `dvc.yaml` - DVC pipeline definition
- `params.yaml` - Configuration parameters

## 🐛 Common Issues

**Issue:** Module not found  
**Fix:** Activate virtual environment

**Issue:** Data file not found  
**Fix:** Place CSV in `data/raw/`

**Issue:** MLflow model not found  
**Fix:** Train model first, set alias to "Production"

## 📚 Full Documentation

See `SETUP.md` for detailed instructions.
