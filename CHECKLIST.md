# Assignment Completion Checklist

**Student:** Bhanu Reddy  
**Roll No:** 2022BCD0026  
**Date:** April 15, 2026

---

## ✅ Implementation Checklist

### Part 1: Project Setup
- [x] Create folder structure
- [x] Initialize Git repository
- [x] Create .gitignore
- [x] Create requirements.txt
- [x] Create README.md
- [x] Create Dockerfile
- [x] Create params.yaml
- [x] Create dvc.yaml

### Part 2: Data Management
- [x] Create data/raw directory
- [x] Create data/processed directory
- [x] Create data/external directory
- [ ] Add customer_churn.csv to data/raw (USER ACTION REQUIRED)

### Part 3: Source Code - Data
- [x] Implement src/data/load_data.py
- [x] Implement src/data/preprocess.py
- [x] Add logging to data modules
- [x] Add error handling

### Part 4: Source Code - Features
- [x] Implement src/features/build_features.py
- [x] Create num_services feature
- [x] Create is_monthly_contract feature
- [x] Create avg_charge_per_tenure feature
- [x] Create tenure_group feature
- [x] Create has_support feature

### Part 5: Source Code - Models
- [x] Implement src/models/train_model.py
- [x] Create sklearn Pipeline
- [x] Add StandardScaler for numerical features
- [x] Add OneHotEncoder for categorical features
- [x] Add RandomForestClassifier
- [x] Implement src/models/evaluate_model.py
- [x] Calculate F1 Score
- [x] Calculate ROC-AUC
- [x] Save metrics to JSON

### Part 6: Training Pipeline
- [x] Implement training/train.py (v1 - load & preprocess)
- [x] Update training/train.py (v2 - add feature engineering)
- [x] Update training/train.py (v3 - full pipeline)
- [x] Save processed data
- [x] Save model artifact
- [x] Save evaluation metrics

### Part 7: DVC Setup
- [ ] Run: dvc init (USER ACTION REQUIRED)
- [ ] Run: dvc add data/raw/customer_churn.csv (USER ACTION REQUIRED)
- [ ] Run: git add data/raw/customer_churn.csv.dvc (USER ACTION REQUIRED)
- [ ] Run: git commit -m "Track data with DVC" (USER ACTION REQUIRED)
- [x] Create dvc.yaml pipeline definition
- [x] Define pipeline stages
- [x] Define dependencies
- [x] Define outputs

### Part 8: MLflow Integration
- [x] Add MLflow to train_model.py
- [x] Log parameters (model, n_estimators)
- [x] Log metrics (f1_score, roc_auc)
- [x] Log model artifact
- [x] Set registered_model_name
- [ ] Run: mlflow ui (USER ACTION REQUIRED)
- [ ] Register model in MLflow UI (USER ACTION REQUIRED)
- [ ] Set alias to "Production" (USER ACTION REQUIRED)

### Part 9: FastAPI Deployment
- [x] Implement inference/app.py
- [x] Create GET / endpoint (health check)
- [x] Create POST /predict endpoint
- [x] Load model from MLflow registry
- [x] Implement inference/schema.py
- [x] Create Pydantic models
- [ ] Run: uvicorn inference.app:app --reload (USER ACTION REQUIRED)
- [ ] Test API endpoints (USER ACTION REQUIRED)

### Part 10: Testing
- [x] Create test_api.py
- [x] Implement test_home()
- [x] Implement test_predict()
- [x] Add sample data
- [ ] Run: python test_api.py (USER ACTION REQUIRED)

### Part 11: Documentation
- [x] Create README.md
- [x] Create SETUP.md
- [x] Create QUICK_START.md
- [x] Create ARCHITECTURE.md
- [x] Create CONTRIBUTING.md
- [x] Create PROJECT_SUMMARY.md
- [x] Create CHECKLIST.md
- [x] Create data/raw/README.md
- [x] Create screenshots/README.md

### Part 12: Additional Features
- [x] Create src/pipeline/training_pipeline.py
- [x] Create src/pipeline/inference_pipeline.py
- [x] Create src/utils/config.py
- [x] Create src/utils/logger.py
- [x] Add all __init__.py files
- [x] Add .gitkeep files for empty directories

### Part 13: Git & GitHub
- [x] Initialize Git repository
- [x] Create .gitignore
- [x] Add all files to Git
- [x] Commit initial structure
- [x] Add remote repository
- [x] Push to GitHub
- [x] Verify repository online

---

## 🎯 User Actions Required

To complete the assignment, you need to:

### 1. Add Dataset
```bash
# Place customer_churn.csv in data/raw/ directory
```

### 2. Setup Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Initialize DVC
```bash
dvc init
dvc add data/raw/customer_churn.csv
git add data/raw/customer_churn.csv.dvc .gitignore
git commit -m "Track data with DVC"
```

### 4. Run Training Pipeline
```bash
python -m training.train
```

### 5. Run DVC Pipeline
```bash
dvc repro
```

### 6. Start MLflow UI
```bash
mlflow ui
# Open http://localhost:5000
# Set model alias to "Production"
```

### 7. Start FastAPI Server
```bash
uvicorn inference.app:app --reload
# Open http://127.0.0.1:8000/docs
```

### 8. Test API
```bash
python test_api.py
```

### 9. Capture Screenshots
Follow the list in `screenshots/README.md`:
- [ ] 01_folder_structure.png
- [ ] 02_train_v1_run.png
- [ ] 03_train_v2_feature_engineering.png
- [ ] 04_full_pipeline_run.png
- [ ] 05_pip_install.png
- [ ] 06_git_init.png
- [ ] 07_dvc_init.png
- [ ] 08_dvc_add_datasets.png
- [ ] 09_dvc_yaml.png
- [ ] 10_dvc_repro.png
- [ ] 11_dvc_repro_mlflow.png
- [ ] 12_mlflow_ui_experiments.png
- [ ] 13_mlflow_run_detail.png
- [ ] 14_model_registry_v1.png
- [ ] 15_dvc_repro_force.png
- [ ] 16_model_production_alias.png
- [ ] 17_mlflow_second_run.png
- [ ] 18_model_in_production.png
- [ ] 19_uvicorn_run.png
- [ ] 20_fastapi_predict_response.png

### 10. Update Documentation
- [ ] Add actual screenshots to markdown files
- [ ] Update README.md with final results
- [ ] Commit and push changes

---

## 📊 Expected Results

After completing all steps, you should have:

✅ Working training pipeline  
✅ F1 Score: ~0.55  
✅ ROC-AUC: ~0.84  
✅ Model saved in models/  
✅ Metrics saved in reports/  
✅ MLflow experiments tracked  
✅ Model registered in MLflow  
✅ FastAPI server running  
✅ API responding to requests  
✅ All code on GitHub  

---

## 🔗 Important Links

- **GitHub Repository:** https://github.com/BhanuReddy1973/bcd26-mlops-assign2
- **MLflow UI:** http://localhost:5000
- **FastAPI Docs:** http://127.0.0.1:8000/docs
- **Reference Repo:** https://github.com/2022BCS0019-abhinav/2022bcs0019-assignment-mlops

---

## 📝 Notes

- All code is committed and pushed to GitHub ✅
- Project structure is complete ✅
- All modules are implemented ✅
- Documentation is comprehensive ✅
- Ready for execution once dataset is added ✅

---

**Status:** Implementation Complete - Ready for Execution  
**Next Step:** Add dataset and run pipeline
