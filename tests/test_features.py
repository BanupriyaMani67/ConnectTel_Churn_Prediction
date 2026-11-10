
import pandas as pd
from src.feature_engineering import add_features

def test_feature_engineering_creates_expected_columns():
    row = {
        "tenure":[0], "TotalCharges":[0.0], "MonthlyCharges":[50.0],
        "PhoneService":["Yes"], "MultipleLines":["No"],
        "OnlineSecurity":["No"], "OnlineBackup":["No"], "DeviceProtection":["No"],
        "TechSupport":["No"], "StreamingTV":["No"], "StreamingMovies":["No"],
        "Contract":["Month-to-month"], "InternetService":["DSL"]
    }
    result = add_features(pd.DataFrame(row))
    assert {"TotalChargesPerTenure", "ServiceCount", "TenureGroup",
            "IsNewCustomer", "IsMonthToMonth", "HasInternetService"}.issubset(result.columns)
    assert result.loc[0, "TotalChargesPerTenure"] == 0
