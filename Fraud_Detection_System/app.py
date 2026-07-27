from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_data
from model import train_model

# ---------------- Base Directory ----------------
BASE_DIR = Path(__file__).resolve().parent

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Digital Payment Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------- Load CSS ----------------
css_file = BASE_DIR / "style.css"

if css_file.exists():
    with open(css_file, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("⚠ style.css not found.")

# ---------------- Load Dataset ----------------
try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Unable to load dataset.\n\n{e}")
    st.stop()

# ---------------- Train Model ----------------
try:
    model = train_model(df)
except Exception as e:
    st.error(f"❌ Model training failed.\n\n{e}")
    st.stop()

# ---------------- Dashboard ----------------
st.title("💳 Digital Payment Fraud Detection using Anomaly Detection")

st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Transactions", len(df))
col2.metric("Average Amount", f"₹{df['Amount'].mean():.0f}")
col3.metric("Highest Amount", f"₹{df['Amount'].max():.0f}")
col4.metric("Safe Transactions", len(df[df["LocationRisk"] == 0]))

# ---------------- Charts ----------------
st.subheader("📊 Transaction Analysis")

c1, c2 = st.columns(2)

with c1:
    fig = px.histogram(
        df,
        x="Amount",
        color="TransactionType",
        nbins=10,
        title="Transaction Amount Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig2 = px.scatter(
        df,
        x="Time",
        y="Amount",
        color="LocationRisk",
        size="Amount",
        title="Transaction Time vs Amount"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------- Prediction ----------------
st.header("🔍 Fraud Prediction")

left, right = st.columns(2)

with left:
    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=100,
        max_value=100000,
        value=5000
    )

    hour = st.slider(
        "Transaction Hour",
        0,
        23,
        12
    )

with right:
    ttype = st.selectbox(
        "Transaction Type",
        ["UPI", "Card"]
    )

    location = st.selectbox(
        "Location",
        ["Safe", "High Risk"]
    )

transaction = 0 if ttype == "UPI" else 1
risk = 0 if location == "Safe" else 1

if st.button("🚀 Predict Fraud", use_container_width=True):

    sample = pd.DataFrame({
        "Amount": [amount],
        "Time": [hour],
        "TransactionType": [transaction],
        "LocationRisk": [risk]
    })

    prediction = model.predict(sample)

    if prediction[0] == -1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Genuine Transaction")

st.divider()

# ---------------- Dataset ----------------
st.subheader("📄 Dataset")

st.dataframe(df, use_container_width=True)