# MLOps Assignment Part 2 - Customer Churn Prediction

**Student:** Bhanu Reddy  
**Roll No:** 2022BCD0026

## Project Overview

This project implements an end-to-end MLOps pipeline for customer churn prediction using:
- DVC for data versioning and pipeline management
- MLflow for experiment tracking and model registry
- FastAPI for model serving
- scikit-learn for model training

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw dataset
│   ├── processed/        # Processed data
│   └── external/         # External data sources
├── src/
│   ├── data/            # Data loading and preprocessing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training and evaluation
│   └── utils/           # Utility functions
├── training/            # Training scripts
├── inference/           # FastAPI application
├── models/              # Saved model artifacts
├── reports/             # Metrics and figures
├── notebooks/           # Jupyter notebooks
├── dvc.yaml            # DVC pipeline definition
├── params.yaml         # Pipeline parameters
└── requirements.txt    # Python dependencies
```

## Setup

1. Create virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize DVC:
```bash
dvc init
```

## Usage

### Run Training Pipeline
```bash
dvc repro
```

### Start MLflow UI
```bash
mlflow ui
```

### Start API Server
```bash
uvicorn inference.app:app --reload
```

## Model Performance

- F1 Score: 0.5533
- ROC-AUC: 0.8357

## GitHub Repository

[https://github.com/BhanuReddy1973/bcd26-mlops-assign2](https://github.com/BhanuReddy1973/bcd26-mlops-assign2)
