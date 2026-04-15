# Project Architecture

## Overview

This project implements an end-to-end MLOps pipeline for customer churn prediction with the following components:

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                               │
├─────────────────────────────────────────────────────────────┤
│  Raw Data (CSV) → DVC Tracking → Processed Data             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Processing Layer                            │
├─────────────────────────────────────────────────────────────┤
│  Load → Preprocess → Feature Engineering                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Training Layer                             │
├─────────────────────────────────────────────────────────────┤
│  sklearn Pipeline (Preprocessing + RandomForest)             │
│  ├── StandardScaler (numerical)                              │
│  ├── OneHotEncoder (categorical)                             │
│  └── RandomForestClassifier                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 Experiment Tracking                          │
├─────────────────────────────────────────────────────────────┤
│  MLflow                                                      │
│  ├── Parameters (model type, hyperparameters)               │
│  ├── Metrics (F1, ROC-AUC)                                   │
│  └── Artifacts (model.pkl)                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Model Registry                             │
├─────────────────────────────────────────────────────────────┤
│  MLflow Model Registry                                       │
│  └── churn-model@production                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Serving Layer                               │
├─────────────────────────────────────────────────────────────┤
│  FastAPI REST API                                            │
│  ├── GET  /         (health check)                           │
│  └── POST /predict  (inference)                              │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Management (DVC)

- **Purpose:** Version control for datasets
- **Files:** `dvc.yaml`, `.dvc` files
- **Benefits:** Reproducibility, data lineage

### 2. Data Processing

#### Load Data (`src/data/load_data.py`)
- Reads CSV into pandas DataFrame
- Validates data shape
- Error handling

#### Preprocessing (`src/data/preprocess.py`)
- Drops non-predictive columns
- Handles missing values
- Type conversions
- Target encoding

#### Feature Engineering (`src/features/build_features.py`)
- Creates 5 new features:
  - `num_services`: Service count
  - `is_monthly_contract`: Contract flag
  - `avg_charge_per_tenure`: Average charge
  - `tenure_group`: Tenure buckets
  - `has_support`: Support flag

### 3. Model Training

#### Pipeline (`src/models/train_model.py`)
```python
Pipeline([
    ("preprocessing", ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(), categorical_cols)
    ])),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])
```

#### Evaluation (`src/models/evaluate_model.py`)
- F1 Score
- ROC-AUC
- Saves to JSON

### 4. Experiment Tracking (MLflow)

**Logged Information:**
- Parameters: model type, hyperparameters
- Metrics: F1, ROC-AUC
- Artifacts: serialized model
- Model registry: versioned models

### 5. Model Serving (FastAPI)

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Churn prediction |

**Request Format:**
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  ...
}
```

**Response Format:**
```json
{
  "prediction": 0,
  "churn": "No"
}
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| ML Framework | scikit-learn |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| API Framework | FastAPI |
| Server | Uvicorn |
| Serialization | joblib |

## Data Flow

1. **Raw Data** → CSV file in `data/raw/`
2. **Load** → pandas DataFrame
3. **Preprocess** → Clean, transform, encode
4. **Feature Engineering** → Create new features
5. **Save Processed** → `data/processed/data.csv`
6. **Train** → sklearn Pipeline
7. **Evaluate** → Metrics calculation
8. **Log** → MLflow tracking
9. **Register** → MLflow Model Registry
10. **Serve** → FastAPI endpoint

## Deployment Options

### Local Development
```bash
uvicorn inference.app:app --reload
```

### Docker Container
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### Production Considerations
- Load balancing
- Model versioning
- A/B testing
- Monitoring & logging
- Auto-scaling

## Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| F1 Score | 0.5533 | Moderate precision-recall balance |
| ROC-AUC | 0.8357 | Good discrimination ability |

## Future Enhancements

1. **Model Improvements**
   - Hyperparameter tuning
   - Ensemble methods
   - Deep learning models

2. **Pipeline Enhancements**
   - Automated retraining
   - Data drift detection
   - Model monitoring

3. **API Features**
   - Batch predictions
   - Authentication
   - Rate limiting
   - Caching

4. **Infrastructure**
   - CI/CD pipeline
   - Kubernetes deployment
   - Cloud integration
