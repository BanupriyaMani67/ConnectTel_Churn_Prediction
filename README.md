
# ConnectTel – Predictive Customer Churn Intelligence

An end-to-end machine-learning project that analyzes telecom customer churn, trains Logistic Regression, Random Forest and XGBoost models, evaluates performance, explains predictions with SHAP, and provides a Streamlit dashboard.

## 1. Dataset

Download the IBM Telco Customer Churn dataset from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/`.

The dataset is not bundled in this repository. This avoids redistributing a third-party dataset and ensures the project uses the actual source data.

## 2. Windows PowerShell setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## 3. Train models

From the project root:
```powershell
python scripts/train.py
```

Training performs data cleaning, feature engineering, a stratified 80/20 split, 5-fold stratified cross-validation, randomized hyperparameter search and held-out test evaluation. It saves model artifacts under `models/`.

The final hold-out test set is not used for tuning. Model choice is based on test ROC-AUC after the tuning process; inspect cross-validation results and business metrics before deployment.

## 4. Run the dashboard

```powershell
streamlit run app/streamlit_app.py
```

Pages include executive overview, exploratory analysis, model performance, single-customer prediction, batch CSV prediction and retention recommendations.

## 5. Jupyter Notebook

```powershell
jupyter notebook
```
Open `notebooks/ConnectTel_Churn_Analysis.ipynb` and run cells from top to bottom. Run it from the project root (or use the notebook's path handling).

## 6. Batch prediction

The input CSV must contain the expected customer feature columns:
```powershell
python scripts/predict.py path\to\customers.csv --output churn_predictions.csv
```

## 7. Tests

```powershell
pytest -q
```

## 8. Project structure

- `data/`: raw and processed dataset locations
- `notebooks/`: executable analysis report
- `src/`: reusable data, feature, modeling, evaluation and prediction modules
- `app/`: Streamlit dashboard
- `models/`: saved model artifacts and metadata
- `reports/`: business and evaluation reports
- `scripts/`: training and batch-prediction entry points
- `tests/`: automated tests

## 9. Metrics and interpretation

Metrics include accuracy, precision, recall, F1, ROC-AUC, PR-AUC and specificity. The 0.5 classification threshold is a default, not necessarily the best business threshold. Select an operational threshold using retention capacity and the relative costs of false positives and false negatives.

SHAP explanations describe model behavior, not causal effects. Retention recommendations are hypotheses to validate with controlled experiments.

## 10. Limitations

The public dataset is historical and may not represent current customer behavior. The model does not know the actual outcome of a retention offer. Estimated revenue impact requires business-supplied financial assumptions and experimental validation.

## 11. Security and responsible use

Do not upload personal customer data to public repositories. Restrict access to customer-level predictions, minimize exported identifiers, and follow applicable privacy and data-protection requirements.
