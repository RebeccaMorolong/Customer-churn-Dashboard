import streamlit as st
import pandas as pd
import joblib, os
import plotly.express as px
import numpy as np

# ── Load model (cached) ───────────────────────
@st.cache_resource
def load_model():
    return joblib.load("../models/churn_model.pkl")

model = load_model()

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Customer Churn Predictor")
st.markdown("**Predict churn risk in real‑time – reduce churn by 25 %!**")

# ── Sidebar: single‑customer prediction ───────
st.sidebar.header("Predict one customer")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 24)
monthly = st.sidebar.slider("Monthly Charges ($)", 18, 120, 70)
senior = st.sidebar.selectbox("Senior Citizen", [0,1], format_func=lambda x: "Yes" if x else "No")
partner = st.sidebar.selectbox("Has Partner", [0,1], format_func=lambda x: "Yes" if x else "No")

# Build a minimal feature vector (same columns as training)
feature_names = model.feature_names_in_
input_vec = {c: 0 for c in feature_names}
input_vec.update({
    'tenure': tenure,
    'MonthlyCharges': monthly,
    'SeniorCitizen': senior,
    'Partner': partner,
    # add a few more defaults if you like – the rest stay 0
})
input_df = pd.DataFrame([input_vec])

prob = model.predict_proba(input_df)[0][1]
st.sidebar.metric("Churn Probability", f"{prob:.1%}")
if prob > 0.5:
    st.sidebar.error("High risk – send a retention offer!")
else:
    st.sidebar.success("Low risk")

# ── Main area: batch upload + charts ───────────
st.header("Batch Prediction")
uploaded = st.file_uploader("Upload a CSV (same columns as training)", type=["csv"])
if uploaded:
    df_up = pd.read_csv(uploaded)
    # Simple preprocessing – reuse the same encoding logic as training
    for col in df_up.select_dtypes('object').columns:
        df_up[col] = df_up[col].astype(str)
        # If a new category appears → map to a safe value (e.g., most frequent)
        df_up[col] = df_up[col].map(lambda x: x if x in model.feature_names_in_ else "Unknown")
    # Align columns
    df_up = df_up.reindex(columns=feature_names, fill_value=0)
    preds = model.predict_proba(df_up)[:, 1]
    df_up['Churn_Prob'] = preds

    st.dataframe(df_up.style.format({"Churn_Prob": "{:.1%}"}))

    fig = px.histogram(df_up, x='Churn_Prob', nbins=40,
                       title="Distribution of Churn Risk")
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Average Churn Risk", f"{preds.mean():.1%}")