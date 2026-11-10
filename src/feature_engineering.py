
import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

def add_features(df):
    data = df.copy()
    data["TotalChargesPerTenure"] = np.where(
        data["tenure"].fillna(0) > 0,
        data["TotalCharges"] / data["tenure"].replace(0, np.nan),
        0.0
    )
    # Count actual subscribed services; "No internet service" and "No phone service"
    # are deliberately treated as not subscribed.
    data["ServiceCount"] = sum(
        (data[c].astype(str).str.lower() == "yes").astype(int)
        for c in SERVICE_COLUMNS if c in data.columns
    )
    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
        include_lowest=True
    ).astype(str)
    data["IsNewCustomer"] = (data["tenure"] <= 12).astype(int)
    data["IsMonthToMonth"] = (data["Contract"] == "Month-to-month").astype(int)
    data["HasInternetService"] = (data["InternetService"] != "No").astype(int)
    return data.replace([np.inf, -np.inf], np.nan)
