
import pandas as pd
from config import RAW_DATA

EXPECTED_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
}

def load_data(path=RAW_DATA):
    path = str(path)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the IBM Telco Churn CSV "
            "from Kaggle and place it at data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        ) from exc
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df
