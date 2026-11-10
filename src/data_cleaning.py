
import pandas as pd

def clean_data(df):
    data = df.copy()
    data = data.drop_duplicates()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    # Blank TotalCharges commonly occur for zero-tenure customers; keep the row
    # and fill from MonthlyCharges * tenure (which is zero for these records).
    data["TotalCharges"] = data["TotalCharges"].fillna(
        data["MonthlyCharges"] * data["tenure"]
    )
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype(str).str.strip()
    if not set(data["Churn"].dropna().unique()).issubset({"Yes", "No"}):
        raise ValueError("Churn must contain only 'Yes' and 'No'.")
    data = data[data["Churn"].isin(["Yes", "No"])].copy()
    return data
