import time
import uuid
import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from fraud import predict, load_model

load_dotenv()

app     = FastAPI(title="Ecom Fraud Shield API", version="1.0.0")
API_KEY = os.getenv("API_KEY", "dev-secret-key")
model   = None


@app.on_event("startup")
def startup():
    global model
    model = load_model()


# ── SCHEMAS ───────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    transaction_id:           str
    amount:                   float
    payment_method:           str       # upi / credit_card / debit_card / wallet
    merchant_category:        str       # electronics / fashion / grocery / food / travel / pharmacy
    device_os:                str       # android / ios / web
    hour_of_day:              int       # 0–23
    is_new_account:           int       # 0 or 1
    device_changed:           int       # 0 or 1
    location_changed:         int       # 0 or 1
    failed_attempts:          int       # number of failed payment attempts
    txn_count_24hr:           int       # transactions by this user in last 24hrs
    amount_vs_avg:            float     # current amount / user's average amount
    ip_risk_score:            float     # 0.0 (safe) to 1.0 (risky)
    is_international:         int       # 0 or 1
    time_since_last_txn_mins: float     # minutes since user's last transaction


class TransactionResponse(BaseModel):
    transaction_id:    str
    fraud_probability: float
    decision:          str              # APPROVE / OTP_REQUIRED / BLOCK
    risk_factors:      list[str]
    response_time_ms:  float
    request_id:        str


# ── ROUTES ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/v1/predict", response_model=TransactionResponse)
def predict_fraud(payload: TransactionRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    start   = time.time()
    result  = predict(payload.dict(), model=model)
    elapsed = round((time.time() - start) * 1000, 2)

    return TransactionResponse(
        transaction_id    = payload.transaction_id,
        fraud_probability = result["fraud_probability"],
        decision          = result["decision"],
        risk_factors      = result["risk_factors"],
        response_time_ms  = elapsed,
        request_id        = str(uuid.uuid4())
    )

# Run: uvicorn main:app --reload --port 8000
