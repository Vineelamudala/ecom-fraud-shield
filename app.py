import streamlit as st
import pandas as pd
import numpy as np
from fraud import predict, load_model

st.set_page_config(page_title="Ecom Fraud Shield", page_icon="🛡️", layout="centered")

@st.cache_resource
def get_model():
    return load_model()

try:
    model = get_model()
except Exception:
    st.error("Model not found. Run the notebook first to train the model.")
    st.stop()

st.title("🛡️ Ecom Fraud Shield")
st.caption("Real-time e-commerce payment fraud detection — Demo")

tab1, tab2 = st.tabs(["Single Transaction", "Batch Scoring"])


# ── TAB 1: SINGLE TRANSACTION ─────────────────────────────────

with tab1:
    st.subheader("Score a Transaction")

    col1, col2 = st.columns(2)

    with col1:
        amount           = st.number_input("Amount (₹)",          min_value=1.0,   value=52000.0, step=500.0)
        payment_method   = st.selectbox("Payment Method",         ["upi", "credit_card", "debit_card", "wallet"])
        merchant_category= st.selectbox("Merchant Category",      ["electronics", "fashion", "grocery", "food", "travel", "pharmacy"])
        device_os        = st.selectbox("Device OS",              ["android", "ios", "web"])
        hour_of_day      = st.slider("Hour of Day",               0, 23, 14)
        is_new_account   = st.selectbox("New Account?",           [0, 1])

    with col2:
        device_changed           = st.selectbox("Device Changed?",    [0, 1])
        location_changed         = st.selectbox("Location Changed?",  [0, 1])
        failed_attempts          = st.slider("Failed Attempts",       0, 5, 0)
        txn_count_24hr           = st.slider("Txns in Last 24hr",     0, 30, 2)
        amount_vs_avg            = st.number_input("Amount vs User Avg (ratio)", min_value=0.1, value=1.0, step=0.5)
        ip_risk_score            = st.slider("IP Risk Score",         0.0, 1.0, 0.1)
        is_international         = st.selectbox("International?",     [0, 1])
        time_since_last_txn_mins = st.number_input("Mins Since Last Txn", min_value=1.0, value=120.0)

    # Quick scenario buttons
    st.caption("Quick scenarios:")
    s1, s2, s3 = st.columns(3)

    if s1.button("✅ Normal ₹1,200"):
        amount, payment_method, merchant_category = 1200, "upi", "grocery"
        device_os, hour_of_day, is_new_account = "android", 11, 0
        device_changed, location_changed, failed_attempts = 0, 0, 0
        txn_count_24hr, amount_vs_avg, ip_risk_score = 2, 1.0, 0.05
        is_international, time_since_last_txn_mins = 0, 300.0

    if s2.button("⚠️ Suspicious ₹90K"):
        amount, payment_method, merchant_category = 90000, "credit_card", "electronics"
        device_os, hour_of_day, is_new_account = "android", 2, 0
        device_changed, location_changed, failed_attempts = 1, 1, 1
        txn_count_24hr, amount_vs_avg, ip_risk_score = 6, 6.5, 0.65
        is_international, time_since_last_txn_mins = 1, 15.0

    if s3.button("🚫 Block This"):
        amount, payment_method, merchant_category = 150000, "credit_card", "electronics"
        device_os, hour_of_day, is_new_account = "web", 3, 1
        device_changed, location_changed, failed_attempts = 1, 1, 2
        txn_count_24hr, amount_vs_avg, ip_risk_score = 10, 12.0, 0.92
        is_international, time_since_last_txn_mins = 1, 5.0

    if st.button("Analyse Transaction", use_container_width=True, type="primary"):
        result = predict({
            "amount":                   amount,
            "payment_method":           payment_method,
            "merchant_category":        merchant_category,
            "device_os":                device_os,
            "hour_of_day":              hour_of_day,
            "is_new_account":           is_new_account,
            "device_changed":           device_changed,
            "location_changed":         location_changed,
            "failed_attempts":          failed_attempts,
            "txn_count_24hr":           txn_count_24hr,
            "amount_vs_avg":            amount_vs_avg,
            "ip_risk_score":            ip_risk_score,
            "is_international":         is_international,
            "time_since_last_txn_mins": time_since_last_txn_mins
        }, model=model)

        prob     = result["fraud_probability"]
        decision = result["decision"]

        st.divider()
        if decision == "APPROVE":
            st.success(f"✅ APPROVED  —  Fraud Score: {prob:.2%}")
        elif decision == "OTP_REQUIRED":
            st.warning(f"⚠️ OTP REQUIRED  —  Fraud Score: {prob:.2%}")
        else:
            st.error(f"🚫 BLOCKED  —  Fraud Score: {prob:.2%}")

        st.progress(prob, text=f"Fraud Probability: {prob:.2%}")

        st.subheader("Top Risk Factors")
        for factor in result["risk_factors"]:
            name, val = factor.split(":")
            val = float(val.strip())
            st.write(f"{'🔴' if val > 0 else '🟢'} **{name.strip()}**: `{val:+.3f}`")


# ── TAB 2: BATCH SCORING ──────────────────────────────────────

with tab2:
    st.subheader("Batch Score Transactions")
    st.caption("CSV columns needed: amount, payment_method, merchant_category, device_os, hour_of_day, is_new_account, device_changed, location_changed, failed_attempts, txn_count_24hr, amount_vs_avg, ip_risk_score, is_international, time_since_last_txn_mins")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df):,} transactions")

        if st.button("Score All", type="primary"):
            with st.spinner("Scoring..."):
                df["fraud_probability"] = [
                    predict(row.to_dict(), model=model)["fraud_probability"]
                    for _, row in df.iterrows()
                ]
                df["decision"] = df["fraud_probability"].apply(
                    lambda p: "BLOCK" if p >= 0.75 else ("OTP_REQUIRED" if p >= 0.40 else "APPROVE")
                )

            st.dataframe(df[["amount", "payment_method", "merchant_category", "fraud_probability", "decision"]], use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Approved",     (df["decision"] == "APPROVE").sum())
            c2.metric("OTP Required", (df["decision"] == "OTP_REQUIRED").sum())
            c3.metric("Blocked",      (df["decision"] == "BLOCK").sum())

            st.download_button("Download Results", df.to_csv(index=False), "results.csv", "text/csv")
