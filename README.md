--- 
title: Ecom Fraud Shield
emoji: 🚀
colorFrom: red
colorTo: red
sdk: streamlit
app_port: 8501
tags:
  - streamlit 
pinned: false
short_description: Real-time e-commerce payment fraud detection
---


# 🛡️ Ecom Fraud Shield — Real-Time Payment Fraud Detection

Real-time e-commerce payment fraud detection using XGBoost + SHAP explainability.  
Score transactions instantly — **APPROVE**, **OTP REQUIRED**, or **BLOCK** with risk factor explanations.

🔗 **Live Demo** → [HuggingFace Space](https://huggingface.co/spaces/Vineel0508/Ecom-fraud-shield)  
🔗 **GitHub** → [github.com/Vineelamudala/ecom-fraud-shield](https://github.com/Vineelamudala/Ecom-fraud-shield)

---

## 📌 Problem Statement

An e-commerce platform processes **8 million transactions/month** with a **1.3% fraud rate** — causing significant revenue loss. Manual review is too slow. The goal is a real-time ML system that scores every transaction in **under 200ms** and returns an actionable decision.

---

## 🏗️ Architecture

```
Customer clicks Pay
        ↓
E-commerce Backend enriches payload
(queries user history — txn count, avg spend, device info)
        ↓
POST /v1/predict  →  FastAPI + XGBoost
        ↓
build_features()  →  20 engineered features
        ↓
model.predict_proba()  →  fraud probability
        ↓
Decision Engine  →  APPROVE / OTP_REQUIRED / BLOCK
        ↓
SHAP  →  top 3 risk factors explained
        ↓
JSON response in ~15ms
```

---

## 🎯 Decision Engine

| Fraud Score | Decision | Action |
|---|---|---|
| < 0.40 | ✅ APPROVE | Transaction goes through |
| 0.40 – 0.75 | ⚠️ OTP REQUIRED | Friction added, user verified |
| > 0.75 | 🚫 BLOCK | Transaction declined |

---

## 🔧 Features (20 Total)

**Raw features** — from transaction payload:
`amount`, `payment_method`, `merchant_category`, `device_os`, `hour_of_day`, `is_new_account`, `device_changed`, `location_changed`, `failed_attempts`, `txn_count_24hr`, `amount_vs_avg`, `ip_risk_score`, `is_international`, `time_since_last_txn_mins`

**Engineered features** — computed in `build_features()`:
`amount_log`, `is_late_night`, `is_high_value`, `is_card_payment`, `is_high_risk_merchant`, `high_velocity`, `combined_risk`

---

## 🤖 Model Training

- **6 candidates**: LogisticRegression, DecisionTree, RandomForest, GradientBoosting, XGBoost, LightGBM
- **Selection**: RandomizedSearchCV with StratifiedKFold (5 folds, 10 iterations)
- **Metric**: AUC-PR (not AUC-ROC — imbalanced data at 1.3% fraud rate)
- **Imbalance**: SMOTE applied on training data (`sampling_strategy=0.1`)
- **Tracking**: MLflow logs all runs — params, metrics, best model
- **Explainability**: SHAP TreeExplainer for top risk factors per prediction

---

## 📊 Key Design Decisions

**Why AUC-PR over AUC-ROC?**  
With only 1.3% fraud rate, AUC-ROC is misleading — a model predicting all legitimate would score 0.987. AUC-PR focuses on the minority class performance.

**Why SMOTE inside CV folds?**  
Applying SMOTE before splitting causes data leakage — synthetic fraud samples from training appear in validation. SMOTE must be applied only on training folds.

**Why separate decision engine?**  
Thresholds are stored in `.env` — the risk team can tune BLOCK/OTP thresholds without retraining the model.

**Why same `build_features()` at train and serve?**  
One function called in both `train()` and `predict()` — prevents train/serve skew.

---

## 🗂️ Project Structure

```
ecom-fraud-shield/
├── fraud.py          # Core: data generation, features, training, prediction
├── app.py            # Streamlit demo — single transaction + batch scoring
├── main.py           # FastAPI production REST API
├── models/
│   └── fraud_model.pkl
├── requirements.txt
└── .github/
    └── workflows/
        └── deploy.yml
```

---

## 🚀 Run Locally

**Install dependencies:**
```bash
conda create -p ./fraud-detect python=3.11 -y
conda activate ./fraud-detect
pip install -r requirements.txt
```

**Train model:**
```bash
jupyter notebook notebook.ipynb
# Run all cells top to bottom
```

**Run Streamlit demo:**
```bash
streamlit run app.py
```

**Run production API:**
```bash
uvicorn main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| ML Framework | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Imbalance | imbalanced-learn (SMOTE) |
| Experiment Tracking | MLflow |
| API | FastAPI + Pydantic |
| Demo | Streamlit |
| CI/CD | GitHub Actions |
| Deployment | Hugging Face Spaces |

---

## 📡 API Usage

```python
import requests

response = requests.post(
    url     = "https://your-api.com/v1/predict",
    headers = {"x-api-key": "your-key"},
    json    = {
        "transaction_id":           "TXN_001",
        "amount":                   85000,
        "payment_method":           "credit_card",
        "merchant_category":        "electronics",
        "device_os":                "android",
        "hour_of_day":              2,
        "is_new_account":           1,
        "device_changed":           1,
        "location_changed":         1,
        "failed_attempts":          2,
        "txn_count_24hr":           6,
        "amount_vs_avg":            14.2,
        "ip_risk_score":            0.91,
        "is_international":         0,
        "time_since_last_txn_mins": 5
    }
)

print(response.json())
# {
#   "decision": "BLOCK",
#   "fraud_probability": 0.9412,
#   "risk_factors": ["combined_risk: +0.41", "ip_risk_score: +0.38", ...]
# }
```

---

## 👤 Author

**Vineel Amudala**

🔗 [GitHub](https://github.com/Vineel0508) · [LinkedIn](https://linkedin.com/in/vineel-amudala)
