import os, pickle, logging
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import mlflow, mlflow.sklearn, mlflow.xgboost
import xgboost as xgb, shap

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import precision_recall_curve, confusion_matrix, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH      = os.getenv("MODEL_PATH", "models/fraud_model.pkl")
BLOCK_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD_BLOCK", 0.75))
OTP_THRESHOLD   = float(os.getenv("FRAUD_THRESHOLD_OTP",   0.40))


# ── 1. DATA GENERATION ───────────────────────────────────────

def _hour_weights(fraud):
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


def generate_data(n_samples=100000, fraud_rate=0.013):
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    legit = {
        "amount":           np.random.lognormal(7.5, 1.2, n_legit).clip(10, 50000),
        "payment_method":   np.random.choice(["upi","credit_card","debit_card","wallet"], n_legit, p=[0.40,0.25,0.25,0.10]),
        "merchant_category":np.random.choice(["grocery","fashion","electronics","food","travel","pharmacy"], n_legit, p=[0.30,0.25,0.15,0.15,0.10,0.05]),
        "device_os":        np.random.choice(["android","ios","web"], n_legit, p=[0.55,0.30,0.15]),
        "hour_of_day":      np.random.choice(range(24), n_legit, p=_hour_weights(False)),
        "is_new_account":   np.random.choice([0,1], n_legit, p=[0.92,0.08]),
        "device_changed":   np.random.choice([0,1], n_legit, p=[0.85,0.15]),
        "location_changed": np.random.choice([0,1], n_legit, p=[0.85,0.15]),
        "failed_attempts":  np.random.choice([0,1,2,3], n_legit, p=[0.80,0.12,0.05,0.03]),
        "txn_count_24hr":   np.random.poisson(3, n_legit).clip(0, 20),
        "amount_vs_avg":    np.random.normal(1.0, 0.5, n_legit).clip(0.1, 8),
        "ip_risk_score":    np.random.beta(1, 5, n_legit),
        "is_international": np.random.choice([0,1], n_legit, p=[0.88,0.12]),
        "time_since_last_txn_mins": np.random.exponential(200, n_legit).clip(1, 10000),
        "is_fraud":         np.zeros(n_legit, dtype=int)
    }

    fraud = {
        "amount":           np.random.lognormal(8.2, 1.3, n_fraud).clip(500, 100000),
        "payment_method":   np.random.choice(["upi","credit_card","debit_card","wallet"], n_fraud, p=[0.20,0.50,0.25,0.05]),
        "merchant_category":np.random.choice(["grocery","fashion","electronics","food","travel","pharmacy"], n_fraud, p=[0.08,0.12,0.40,0.08,0.27,0.05]),
        "device_os":        np.random.choice(["android","ios","web"], n_fraud, p=[0.55,0.25,0.20]),
        "hour_of_day":      np.random.choice(range(24), n_fraud, p=_hour_weights(True)),
        "is_new_account":   np.random.choice([0,1], n_fraud, p=[0.55,0.45]),
        "device_changed":   np.random.choice([0,1], n_fraud, p=[0.45,0.55]),
        "location_changed": np.random.choice([0,1], n_fraud, p=[0.40,0.60]),
        "failed_attempts":  np.random.choice([0,1,2,3], n_fraud, p=[0.45,0.30,0.15,0.10]),
        "txn_count_24hr":   np.random.poisson(5, n_fraud).clip(0, 25),
        "amount_vs_avg":    np.random.normal(3.0, 1.5, n_fraud).clip(0.5, 15),
        "ip_risk_score":    np.random.beta(3, 2, n_fraud),
        "is_international": np.random.choice([0,1], n_fraud, p=[0.50,0.50]),
        "time_since_last_txn_mins": np.random.exponential(60, n_fraud).clip(1, 500),
        "is_fraud":         np.ones(n_fraud, dtype=int)
    }

    df = pd.concat([pd.DataFrame(legit), pd.DataFrame(fraud)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    log.info(f"Generated {len(df):,} rows | Fraud: {df.is_fraud.mean():.2%}")
    return df

def save_data(df, path="data/raw/ecom_transactions.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Saved to {path}")

def load_data(path):
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} rows | Fraud: {df.is_fraud.mean():.2%}")
    return df

# ── 2. FEATURE ENGINEERING ───────────────────────────────────

def build_features(df):
    df = df.copy()
    df["payment_method_encoded"]    = df["payment_method"].map({"upi":0,"credit_card":1,"debit_card":2,"wallet":3}).fillna(0)
    df["merchant_category_encoded"] = df["merchant_category"].map({"grocery":0,"fashion":1,"electronics":2,"food":3,"travel":4,"pharmacy":5}).fillna(0)
    df["device_os_encoded"]         = df["device_os"].map({"android":0,"ios":1,"web":2}).fillna(0)
    df["is_high_risk_merchant"]     = df["merchant_category"].isin(["electronics","travel"]).astype(int)
    df["is_card_payment"]           = df["payment_method"].isin(["credit_card","debit_card"]).astype(int)
    df["is_late_night"]             = df["hour_of_day"].isin([23,0,1,2,3,4]).astype(int)
    df["amount_log"]                = np.log1p(df["amount"])
    df["is_high_value"]             = (df["amount"] > 10000).astype(int)
    df["high_velocity"]             = (df["txn_count_24hr"] > 5).astype(int)
    df["combined_risk"]             = df["is_new_account"] + df["device_changed"] + df["location_changed"] + df["is_late_night"]
    return df


FEATURE_COLS = [
    "amount_log", "payment_method_encoded", "merchant_category_encoded",
    "device_os_encoded", "hour_of_day", "is_new_account", "device_changed",
    "location_changed", "failed_attempts", "txn_count_24hr", "amount_vs_avg",
    "ip_risk_score", "is_international", "time_since_last_txn_mins",
    "is_high_risk_merchant", "is_card_payment", "is_late_night",
    "is_high_value", "high_velocity", "combined_risk"
]


# ── 3. TRAIN ─────────────────────────────────────────────────

def train(df, n_splits=5, n_iter=5):
    df = build_features(df)
    X, y = df[FEATURE_COLS], df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_train, y_train = SMOTE(random_state=42, sampling_strategy=0.1).fit_resample(X_train, y_train)
    scale = (y_train == 0).sum() / (y_train == 1).sum()

    candidates = {
        "LogisticRegression": (LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            {"C": [0.01,0.1,1,10], "solver": ["lbfgs","saga"]}),
        "DecisionTree":       (DecisionTreeClassifier(class_weight="balanced", random_state=42),
            {"max_depth": [4,6,8,None], "min_samples_leaf": [1,5,10]}),
        "RandomForest":       (RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42),
            {"n_estimators": [100,200], "max_depth": [6,8,None], "max_features": ["sqrt","log2"]}),
        "GradientBoosting":   (GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100,200], "max_depth": [3,5], "learning_rate": [0.05,0.1]}),
        "XGBoost":            (xgb.XGBClassifier(scale_pos_weight=scale, random_state=42, n_jobs=-1, eval_metric="aucpr"),
            {"n_estimators": [100,200], "max_depth": [4,6], "learning_rate": [0.05,0.1], "subsample": [0.8,1.0]}),
        "LightGBM":           (LGBMClassifier(scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1),
            {"n_estimators": [100,200], "max_depth": [4,6], "learning_rate": [0.05,0.1], "num_leaves": [31,63]}),
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    mlflow.set_experiment("fraud-detection")

    results, best_name, best_score = {}, None, -1

    for name, (model, params) in candidates.items():
        log.info(f"Training {name}...")

        search = RandomizedSearchCV(model, params, n_iter=n_iter, scoring="average_precision",
                                    cv=skf, refit=True, n_jobs=-1, random_state=42, verbose=0)
        search.fit(X_train, y_train)

        cv_mean    = search.best_score_
        cv_std     = search.cv_results_["std_test_score"][search.best_index_]
        best_model = search.best_estimator_

        y_prob         = best_model.predict_proba(X_test)[:, 1]
        y_pred         = (y_prob >= BLOCK_THRESHOLD).astype(int)
        prec, rec, _   = precision_recall_curve(y_test, y_prob)
        test_auc       = auc(rec, prec)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        recall         = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr            = fp / (fp + tn) if (fp + tn) > 0 else 0

        log.info(f"  CV: {cv_mean:.4f} ± {cv_std:.4f} | Test AUC: {test_auc:.4f} | Recall: {recall:.4f} | FPR: {fpr:.4f}")

        with mlflow.start_run(run_name=name):
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics({"cv_auc_pr": cv_mean, "test_auc_pr": test_auc, "recall": recall, "fpr": fpr})

        results[name] = {"cv_mean": cv_mean, "cv_std": cv_std, "test_auc": test_auc,
                         "recall": recall, "fpr": fpr, "model": best_model, "y_prob": y_prob}

        if cv_mean > best_score:
            best_score, best_name = cv_mean, name

    # Print comparison table
    log.info(f"\n{'Model':<22} {'CV AUC-PR':>10} {'±Std':>7} {'Test AUC':>10} {'Recall':>8} {'FPR':>8}")
    for n, r in sorted(results.items(), key=lambda x: x[1]["cv_mean"], reverse=True):
        tag = " ← BEST" if n == best_name else ""
        log.info(f"{n:<22} {r['cv_mean']:>10.4f} {r['cv_std']:>7.4f} {r['test_auc']:>10.4f} {r['recall']:>8.4f} {r['fpr']:>8.4f}{tag}")

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(results[best_name]["model"], f)

    log.info(f"\n✅ Best: {best_name} | AUC-PR: {best_score:.4f} | Saved → {MODEL_PATH}")
    return results[best_name]["model"], best_name, X_test, y_test, results[best_name]["y_prob"], results


# ── 4. PREDICT ────────────────────────────────────────────────

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(transaction, model=None):
    """transaction: dict with real e-commerce fields"""
    if model is None:
        model = load_model()

    df = build_features(pd.DataFrame([transaction]))
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

    try:
        explainer = shap.TreeExplainer(model)
        shap_arr  = explainer.shap_values(X)
        if isinstance(shap_arr, list):
            shap_arr = shap_arr[1]        # class 1 = fraud
        shap_arr = np.array(shap_arr).flatten()
    except Exception:
        explainer = shap.LinearExplainer(model, X)
        shap_arr  = explainer.shap_values(X)[0]
        
    top3 = sorted(zip(FEATURE_COLS, shap_arr), key=lambda x: abs(x[1]), reverse=True)[:3]
    return {
        "fraud_probability": round(prob, 4),
        "decision":          decision,
        "risk_factors":      [f"{k}: {v:+.3f}" for k, v in top3]
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
