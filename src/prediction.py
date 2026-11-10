
import joblib
import pandas as pd
from config import MODEL_DIR
from src.feature_engineering import add_features

def load_model():
    path = MODEL_DIR / "best_model.joblib"
    if not path.exists():
        raise FileNotFoundError("Trained model not found. Run: python scripts/train.py")
    return joblib.load(path)

def predict_customers(df, model=None):
    model = model or load_model()
    data = add_features(df.copy())
    ids = data["customerID"] if "customerID" in data else pd.Series(range(1, len(data)+1))
    X = data.drop(columns=["Churn", "customerID"], errors="ignore")
    probabilities = model.predict_proba(X)[:, 1]
    result = pd.DataFrame({
        "customerID": ids,
        "churn_probability": probabilities,
        "risk_category": pd.cut(
            probabilities, bins=[-0.001, 0.30, 0.60, 1.0],
            labels=["Low", "Medium", "High"], include_lowest=True
        ).astype(str),
        "predicted_churn": (probabilities >= 0.5).astype(int)
    })
    return result
