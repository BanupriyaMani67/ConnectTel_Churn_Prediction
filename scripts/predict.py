
import argparse
import pandas as pd
from src.prediction import predict_customers

parser = argparse.ArgumentParser(description="Predict churn for a customer CSV.")
parser.add_argument("input_csv")
parser.add_argument("--output", default="churn_predictions.csv")
args = parser.parse_args()

df = pd.read_csv(args.input_csv)
predictions = predict_customers(df)
predictions.to_csv(args.output, index=False)
print(f"Saved predictions to {args.output}")
