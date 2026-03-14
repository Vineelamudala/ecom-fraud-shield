import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
import random

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import precision_recall_curve, confusion_matrix, auc, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH      = os.getenv("MODEL_PATH", "fraud_model.pkl")
BLOCK_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD_BLOCK", 0.75))
OTP_THRESHOLD   = float(os.getenv("FRAUD_THRESHOLD_OTP",   0.40))


# ── 1. SYNTHETIC DATA GENERATOR ──────────────────────────────
# Generates realistic e-commerce transaction data
# matching what a payment gateway actually sends

def generate_data(n_samples: int = 100000, fraud_rate: float = 0.013) -> pd.DataFrame:
    """
    Generates synthetic e-commerce transaction data.
    Fraud rate default = 1.3% (realistic for e-commerce)
    """
    log.info(f"Generating {n_samples:,} synthetic transactions...")
    np.random.seed(42)
    random.seed(42)

    n_fraud  = int(n_samples * fraud_rate)
    n_legit  = n_samples - n_fraud

    # ── Legitimate transactions ───────────────────────────────
    legit = {
        "amount":           np.random.lognormal(mean=7.5, sigma=1.2, size=n_legit).clip(10, 50000),
        "payment_method":   np.random.choice(["upi", "credit_card", "debit_card", "wallet"], size=n_legit, p=[0.40, 0.25, 0.25, 0.10]),
        "merchant_category":np.random.choice(["grocery", "fashion", "electronics", "food", "travel", "pharmacy"], size=n_legit, p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05]),
        "device_os":        np.random.choice(["android", "ios", "web"], size=n_legit, p=[0.55, 0.30, 0.15]),
        "hour_of_day":      np.random.choice(range(24), size=n_legit, p=_hour_weights(fraud=False)),
        "is_new_account":   np.random.choice([0, 1], size=n_legit, p=[0.92, 0.08]),
        "device_changed":   np.random.choice([0, 1], size=n_legit, p=[0.95, 0.05]),
        "location_changed": np.random.choice([0, 1], size=n_legit, p=[0.96, 0.04]),
        "failed_attempts":  np.random.choice([0, 1, 2, 3], size=n_legit, p=[0.90, 0.07, 0.02, 0.01]),
        "txn_count_24hr":   np.random.poisson(lam=2, size=n_legit).clip(0, 20),
        "amount_vs_avg":    np.random.normal(loc=1.0, scale=0.3, size=n_legit).clip(0.1, 5),
        "ip_risk_score":    np.random.beta(a=1, b=10, size=n_legit),       # mostly low risk
        "is_international": np.random.choice([0, 1], size=n_legit, p=[0.93, 0.07]),
        "time_since_last_txn_mins": np.random.exponential(scale=300, size=n_legit).clip(1, 10000),
        "is_fraud":         np.zeros(n_legit, dtype=int)
    }

    # ── Fraudulent transactions ───────────────────────────────
    fraud = {
        "amount":           np.random.lognormal(mean=9.5, sigma=1.0, size=n_fraud).clip(5000, 200000),  # higher amounts
        "payment_method":   np.random.choice(["upi", "credit_card", "debit_card", "wallet"], size=n_fraud, p=[0.15, 0.55, 0.25, 0.05]),  # card-heavy
        "merchant_category":np.random.choice(["grocery", "fashion", "electronics", "food", "travel", "pharmacy"], size=n_fraud, p=[0.05, 0.10, 0.50, 0.05, 0.25, 0.05]),  # electronics + travel
        "device_os":        np.random.choice(["android", "ios", "web"], size=n_fraud, p=[0.60, 0.20, 0.20]),
        "hour_of_day":      np.random.choice(range(24), size=n_fraud, p=_hour_weights(fraud=True)),     # late night heavy
        "is_new_account":   np.random.choice([0, 1], size=n_fraud, p=[0.40, 0.60]),                     # mostly new accounts
        "device_changed":   np.random.choice([0, 1], size=n_fraud, p=[0.25, 0.75]),                     # device often changed
        "location_changed": np.random.choice([0, 1], size=n_fraud, p=[0.20, 0.80]),                     # location often changed
        "failed_attempts":  np.random.choice([0, 1, 2, 3], size=n_fraud, p=[0.30, 0.30, 0.25, 0.15]),  # more failed attempts
        "txn_count_24hr":   np.random.poisson(lam=6, size=n_fraud).clip(0, 30),                         # high velocity
        "amount_vs_avg":    np.random.normal(loc=4.5, scale=1.5, size=n_fraud).clip(1, 20),             # much higher than user avg
        "ip_risk_score":    np.random.beta(a=5, b=2, size=n_fraud),                                     # high risk IPs
        "is_international": np.random.choice([0, 1], size=n_fraud, p=[0.30, 0.70]),                     # mostly international
        "time_since_last_txn_mins": np.random.exponential(scale=20, size=n_fraud).clip(1, 200),         # rapid succession
        "is_fraud":         np.ones(n_fraud, dtype=int)
    }

    df = pd.concat([pd.DataFrame(legit), pd.DataFrame(fraud)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    log.info(f"Generated {len(df):,} transactions | Fraud rate: {df.is_fraud.mean():.2%}")
    return df


def _hour_weights(fraud: bool) -> list:
    """Legitimate txns peak during day. Fraud peaks late night."""
    if fraud:
        # Fraud spikes 1am–4am
        weights = [0.08, 0.10, 0.10, 0.09, 0.04, 0.02,
                   0.02, 0.02, 0.02, 0.02, 0.03, 0.03,
                   0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                   0.04, 0.04, 0.04, 0.04, 0.05, 0.07]
    else:
        # Legit peaks 10am–8pm
        weights = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
                   0.03, 0.04, 0.05, 0.06, 0.07, 0.07,
                   0.07, 0.07, 0.07, 0.07, 0.07, 0.07,
                   0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    total = sum(weights)
    return [w / total for w in weights]


def save_data(df: pd.DataFrame, path: str = "data/raw/ecom_transactions.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Saved to {path}")


def load_data(path: str) -> pd.DataFrame:
    log.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} rows | Fraud rate: {df.is_fraud.mean():.2%}")
    return df


# ── 2. FEATURE ENGINEERING ───────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode payment method
    df["payment_method_encoded"] = df["payment_method"].map({
        "upi": 0, "credit_card": 1, "debit_card": 2, "wallet": 3
    }).fillna(0)

    # Encode merchant category
    df["merchant_category_encoded"] = df["merchant_category"].map({
        "grocery": 0, "fashion": 1, "electronics": 2,
        "food": 3, "travel": 4, "pharmacy": 5
    }).fillna(0)

    # Encode device
    df["device_os_encoded"] = df["device_os"].map({
        "android": 0, "ios": 1, "web": 2
    }).fillna(0)

    # High risk merchant — electronics and travel have higher fraud
    df["is_high_risk_merchant"] = df["merchant_category"].isin(["electronics", "travel"]).astype(int)

    # High risk payment — credit card fraud is more common
    df["is_card_payment"] = df["payment_method"].isin(["credit_card", "debit_card"]).astype(int)

    # Late night transaction — 11pm to 4am
    df["is_late_night"] = df["hour_of_day"].isin([23, 0, 1, 2, 3, 4]).astype(int)

    # Log transform of amount
    df["amount_log"] = np.log1p(df["amount"])

    # High value flag
    df["is_high_value"] = (df["amount"] > 10000).astype(int)

    # Velocity risk — many txns in 24hr is suspicious
    df["high_velocity"] = (df["txn_count_24hr"] > 5).astype(int)

    # Combined risk signal — new account + high value + device changed
    df["combined_risk"] = (
        df["is_new_account"] +
        df["device_changed"] +
        df["location_changed"] +
        df["is_late_night"]
    )

    return df


FEATURE_COLS = [
    "amount_log",
    "payment_method_encoded",
    "merchant_category_encoded",
    "device_os_encoded",
    "hour_of_day",
    "is_new_account",
    "device_changed",
    "location_changed",
    "failed_attempts",
    "txn_count_24hr",
    "amount_vs_avg",
    "ip_risk_score",
    "is_international",
    "time_since_last_txn_mins",
    "is_high_risk_merchant",
    "is_card_payment",
    "is_late_night",
    "is_high_value",
    "high_velocity",
    "combined_risk"
]


# ── 3. TRAIN ALL MODELS WITH RANDOMIZED SEARCH CV ────────────

def train(df: pd.DataFrame, n_splits: int = 2, n_iter: int = 10):
    """
    For each candidate model:
      1. RandomizedSearchCV tries n_iter random hyperparameter combos
         each evaluated with StratifiedKFold (no data leakage)
      2. Best params per model are selected by AUC-PR
      3. Best model across all candidates saved to disk
    """
    log.info("Starting RandomizedSearchCV training pipeline...")

    df = build_features(df)
    X  = df[FEATURE_COLS]
    y  = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # SMOTE on full training set — applied once before search
    sm = SMOTE(random_state=42, sampling_strategy=0.1)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    log.info(f"After SMOTE: {len(X_train_res):,} rows")

    scale = (y_train_res == 0).sum() / (y_train_res == 1).sum()

    # ── Candidates + their hyperparameter search spaces ──────
    candidates = {

        "LogisticRegression": {
            "model": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            "params": {
                "C":       [0.01, 0.1, 1, 10, 100],
                "solver":  ["lbfgs", "saga"],
            }
        },

        "DecisionTree": {
            "model": DecisionTreeClassifier(class_weight="balanced", random_state=42),
            "params": {
                "max_depth":        [4, 6, 8, 10, None],
                "min_samples_leaf": [1, 5, 10, 20],
                "criterion":        ["gini", "entropy"],
            }
        },

        "RandomForest": {
            "model": RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth":    [6, 8, 10, None],
                "max_features": ["sqrt", "log2"],
            }
        },

        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators":  [100, 200, 300],
                "max_depth":     [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample":     [0.7, 0.8, 1.0],
            }
        },

        "XGBoost": {
            "model": xgb.XGBClassifier(scale_pos_weight=scale, random_state=42, n_jobs=-1, eval_metric="aucpr"),
            "params": {
                "n_estimators":     [100, 200, 300],
                "max_depth":        [4, 6, 8],
                "learning_rate":    [0.01, 0.05, 0.1],
                "subsample":        [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
            }
        },

        "LightGBM": {
            "model": LGBMClassifier(scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1),
            "params": {
                "n_estimators":  [100, 200, 300],
                "max_depth":     [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves":    [31, 63, 127],
                "subsample":     [0.7, 0.8, 1.0],
            }
        },
    }


    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    mlflow.set_experiment("fraud-detection")

    results, best_name, best_score = {}, None, -1

    for name, cfg in candidates.items():
        log.info(f"\nRandomizedSearchCV → {name}  ({n_iter} iters × {n_splits} folds)")

        search = RandomizedSearchCV(
            estimator  = cfg["model"],
            param_distributions = cfg["params"],
            n_iter     = n_iter,
            scoring    = "average_precision",   # AUC-PR equivalent, fast
            cv         = skf,
            refit      = True,                  # refit best params on full train set
            n_jobs     = -1,
            random_state = 42,
            verbose    = 0
        )

        search.fit(X_train_res, y_train_res)

        best_params = search.best_params_
        cv_mean     = search.best_score_
        cv_std      = search.cv_results_["std_test_score"][search.best_index_]
        best_model  = search.best_estimator_

        log.info(f"  Best params: {best_params}")
        log.info(f"  CV AUC-PR:   {cv_mean:.4f} ± {cv_std:.4f}")

        # Evaluate on held-out test set
        y_prob         = best_model.predict_proba(X_test)[:, 1]
        y_pred         = (y_prob >= BLOCK_THRESHOLD).astype(int)
        prec, rec, _   = precision_recall_curve(y_test, y_prob)
        test_auc       = auc(rec, prec)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        recall         = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr            = fp / (fp + tn) if (fp + tn) > 0 else 0

        log.info(f"  Test AUC-PR: {test_auc:.4f} | Recall: {recall:.4f} | FPR: {fpr:.4f}")

        with mlflow.start_run(run_name=name):
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_auc_pr_mean", cv_mean)
            mlflow.log_metric("cv_auc_pr_std",  cv_std)
            mlflow.log_metric("test_auc_pr",    test_auc)
            mlflow.log_metric("recall",         recall)
            mlflow.log_metric("fpr",            fpr)

        results[name] = {
            "cv_mean":   cv_mean,
            "cv_std":    cv_std,
            "test_auc":  test_auc,
            "recall":    recall,
            "fpr":       fpr,
            "model":     best_model,
            "y_prob":    y_prob,
            "params":    best_params
        }

        if cv_mean > best_score:
            best_score = cv_mean
            best_name  = name

    # ── Comparison table ──────────────────────────────────────
    log.info(f"\n{'Model':<22} {'CV AUC-PR':>10} {'±Std':>7} {'Test AUC':>10} {'Recall':>8} {'FPR':>8}")
    for n, r in sorted(results.items(), key=lambda x: x[1]["cv_mean"], reverse=True):
        tag = " ← BEST" if n == best_name else ""
        log.info(f"{n:<22} {r['cv_mean']:>10.4f} {r['cv_std']:>7.4f} {r['test_auc']:>10.4f} {r['recall']:>8.4f} {r['fpr']:>8.4f}{tag}")


    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(results[best_name]["model"], f)

    log.info(f"\n✅ Best: {best_name} | CV AUC-PR: {best_score:.4f} | Saved → {MODEL_PATH}")
    return results[best_name]["model"], best_name, X_test, y_test, results[best_name]["y_prob"], results


# ── 4. PREDICT ────────────────────────────────────────────────

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(transaction: dict, model=None) -> dict:
    """
    transaction: dict with real e-commerce fields
    Example:
    {
        "amount": 52000,
        "payment_method": "credit_card",
        "merchant_category": "electronics",
        "device_os": "android",
        "hour_of_day": 2,
        "is_new_account": 1,
        "device_changed": 1,
        "location_changed": 1,
        "failed_attempts": 1,
        "txn_count_24hr": 4,
        "amount_vs_avg": 8.5,
        "ip_risk_score": 0.82,
        "is_international": 1,
        "time_since_last_txn_mins": 12
    }
    """
    if model is None:
        model = load_model()

    df = pd.DataFrame([transaction])
    df = build_features(df)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    X    = df[FEATURE_COLS]
    prob = float(model.predict_proba(X)[0][1])

    decision = (
        "BLOCK"        if prob >= BLOCK_THRESHOLD else
        "OTP_REQUIRED" if prob >= OTP_THRESHOLD   else
        "APPROVE"
    )

    # SHAP explanation
    try:
        explainer = shap.TreeExplainer(model)
        shap_arr  = explainer.shap_values(X)
        if isinstance(shap_arr, list):
            shap_arr = shap_arr[1]        # class 1 = fraud
        shap_arr = np.array(shap_arr).flatten()
    except Exception:
        explainer = shap.LinearExplainer(model, X)
        shap_arr  = explainer.shap_values(X)[0]

    top3         = sorted(zip(FEATURE_COLS, shap_arr), key=lambda x: abs(x[1]), reverse=True)[:3]
    risk_factors = [f"{k}: {v:+.3f}" for k, v in top3]

    return {
        "fraud_probability": round(prob, 4),
        "decision":          decision,
        "risk_factors":      risk_factors
    }


# ── 5. THRESHOLD SUMMARY ─────────────────────────────────────

def threshold_summary(y_test, y_prob):
    print(f"\n{'Threshold':>10} {'Recall':>10} {'Precision':>10} {'FPR':>10}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.75, 0.85]:
        y_pred         = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        recall         = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision      = tp / (tp + fp) if (tp + fp) > 0 else 0
        fpr            = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"{t:>10.2f} {recall:>10.3f} {precision:>10.3f} {fpr:>10.3f}")
