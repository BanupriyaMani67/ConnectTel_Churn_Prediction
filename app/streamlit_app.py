
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd
import streamlit as st
import plotly.express as px
from config import RAW_DATA, MODEL_DIR
from src.data_loader import load_data
from src.data_cleaning import clean_data
from src.prediction import load_model, predict_customers
from src.recommendations import recommend

st.set_page_config(page_title="ConnectTel Churn Intelligence", page_icon="📡", layout="wide")
st.markdown("""
<style>
.stApp {background:#F8F7FF;}
[data-testid="stMetric"] {background:white;padding:16px;border-radius:14px;border:1px solid #E8E0FF;}
h1,h2,h3 {color:#55418A;}
</style>
""", unsafe_allow_html=True)

st.title("📡 ConnectTel | Customer Churn Intelligence")
st.caption("AI-powered churn prediction, explainability and retention decision support")

@st.cache_data
def get_data():
    return clean_data(load_data())

try:
    df = get_data()
except Exception as e:
    st.error(str(e))
    st.info("Place the Kaggle Telco Churn CSV in data/raw/ and restart the app.")
    st.stop()

metadata_path = MODEL_DIR / "model_metadata.json"
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text())
else:
    metadata = {}

page = st.sidebar.radio("Navigate", [
    "Executive Dashboard", "Exploratory Analysis", "Model Performance",
    "Customer Prediction", "Batch CSV Prediction", "Business Recommendations"
])

if page == "Executive Dashboard":
    st.subheader("Business overview")
    churn_rate = (df["Churn"] == "Yes").mean()
    a, b, c, d = st.columns(4)
    a.metric("Total customers", f"{len(df):,}")
    b.metric("Actual churn rate", f"{churn_rate:.1%}")
    c.metric("Average monthly charge", f"${df['MonthlyCharges'].mean():.2f}")
    c2 = metadata.get("best_model", "Not trained")
    d.metric("Selected model", c2)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Churn", color="Churn", title="Customer churn distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        contract = df.groupby("Contract", observed=False)["Churn"].apply(lambda x: (x == "Yes").mean()).reset_index(name="Churn rate")
        fig = px.bar(contract, x="Contract", y="Churn rate", title="Churn rate by contract", text_auto=".1%")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Predicted risk metrics become available after model training.")

elif page == "Exploratory Analysis":
    st.subheader("Explore customer churn")
    contract_values = ["All"] + sorted(df["Contract"].dropna().unique().tolist())
    chosen = st.selectbox("Contract type", contract_values)
    view = df if chosen == "All" else df[df["Contract"] == chosen]
    internet = view.groupby("InternetService", dropna=False)["Churn"].apply(lambda x: (x == "Yes").mean()).reset_index(name="Churn rate")
    st.plotly_chart(px.bar(internet, x="InternetService", y="Churn rate", title="Churn rate by internet service", text_auto=".1%"), use_container_width=True)
    st.plotly_chart(px.histogram(view, x="tenure", color="Churn", nbins=24, title="Tenure distribution by churn"), use_container_width=True)
    st.plotly_chart(px.box(view, x="Churn", y="MonthlyCharges", color="Churn", title="Monthly charges by churn"), use_container_width=True)

elif page == "Model Performance":
    st.subheader("Model evaluation")
    if not metadata:
        st.warning("Train the models first: python scripts/train.py")
    else:
        st.write("Selected model:", metadata.get("best_model"))
        perf = pd.DataFrame(metadata.get("test_results", {})).T
        st.dataframe(perf, use_container_width=True)
        st.caption("Metrics are calculated on the held-out test set by the training pipeline.")

elif page == "Customer Prediction":
    st.subheader("Predict churn for one customer")
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        gender = col1.selectbox("Gender", ["Female", "Male"])
        senior = col1.selectbox("Senior citizen", [0, 1])
        partner = col1.selectbox("Partner", ["Yes", "No"])
        dependents = col1.selectbox("Dependents", ["Yes", "No"])
        tenure = col2.slider("Tenure (months)", 0, 72, 12)
        phone = col2.selectbox("Phone service", ["Yes", "No"])
        lines = col2.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
        internet = col3.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        security = col3.selectbox("Online security", ["Yes", "No", "No internet service"])
        backup = col3.selectbox("Online backup", ["Yes", "No", "No internet service"])
        protection = col3.selectbox("Device protection", ["Yes", "No", "No internet service"])
        support = col3.selectbox("Tech support", ["Yes", "No", "No internet service"])
        tv = col1.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = col1.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
        contract = col2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = col2.selectbox("Paperless billing", ["Yes", "No"])
        payment = col3.selectbox("Payment method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly charges", min_value=0.0, max_value=500.0, value=70.0)
        total = st.number_input("Total charges", min_value=0.0, max_value=100000.0, value=monthly*tenure)
        submitted = st.form_submit_button("Predict churn")
    if submitted:
        row = pd.DataFrame([{
            "customerID":"NEW_CUSTOMER", "gender":gender, "SeniorCitizen":senior,
            "Partner":partner, "Dependents":dependents, "tenure":tenure,
            "PhoneService":phone, "MultipleLines":lines, "InternetService":internet,
            "OnlineSecurity":security, "OnlineBackup":backup, "DeviceProtection":protection,
            "TechSupport":support, "StreamingTV":tv, "StreamingMovies":movies,
            "Contract":contract, "PaperlessBilling":paperless, "PaymentMethod":payment,
            "MonthlyCharges":monthly, "TotalCharges":total
        }])
        try:
            result = predict_customers(row)
            p = float(result.iloc[0]["churn_probability"])
            st.metric("Predicted churn probability", f"{p:.1%}")
            st.progress(p)
            st.subheader(f"Risk category: {result.iloc[0]['risk_category']}")
            for action in recommend(row.iloc[0].to_dict(), p):
                st.write("• " + action)
        except Exception as e:
            st.error(f"Prediction unavailable: {e}")

elif page == "Batch CSV Prediction":
    st.subheader("Batch customer prediction")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded:
        batch = pd.read_csv(uploaded)
        required = set(df.columns) - {"Churn"}
        missing = required - set(batch.columns)
        if missing:
            st.error(f"Missing required columns: {sorted(missing)}")
        else:
            try:
                output = predict_customers(batch)
                st.dataframe(output, use_container_width=True)
                st.download_button("Download predictions CSV", output.to_csv(index=False), "connecttel_predictions.csv", "text/csv")
            except Exception as e:
                st.error(str(e))

elif page == "Business Recommendations":
    st.subheader("Retention strategy guide")
    st.write("Use model explanations and customer context to prioritize offers. Validate offers through controlled experiments.")
    st.markdown("""
    - **Month-to-month contract:** test annual-contract incentives.
    - **High monthly charges:** review plan fit and offer value-based bundles.
    - **Short tenure:** improve onboarding and early-life engagement.
    - **No technical support:** offer a relevant support trial.
    - **High predicted risk:** prioritize personal outreach.

    These are candidate interventions, not proven causal effects.
    """)
