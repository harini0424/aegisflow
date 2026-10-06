"""
AegisFlow - Stage 4: Real-Time Fraud Scoring Service
======================================================
This turns our trained model into a live web service (an API) - exactly like
what a real bank would run. Any transaction can be sent to it, and it replies
instantly with a decision and an explanation.

HOW TO RUN THIS:
    uvicorn src.scoring_service:app --reload

Then open your browser to:
    http://127.0.0.1:8000/docs
This gives you an interactive page where you can test it without writing any
extra code - just fill in a form and click "Execute".
"""

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------
# Load the trained model (built in Stage 2/3) once, when the service starts
# ----------------------------------------------------------------------
model = joblib.load("models/fraud_model.pkl")
FEATURE_COLUMNS = joblib.load("models/feature_columns.pkl")
explainer = shap.TreeExplainer(model)

# We also need some background info about accounts/devices to compute the
# same "clues" (features) we used during training - in a real bank this
# would come from a live database; here we reload our saved data.
transactions_history = pd.read_csv("data/transactions_with_features.csv", parse_dates=["timestamp"])

app = FastAPI(title="AegisFlow Fraud Scoring API")


# ----------------------------------------------------------------------
# Define what a request looks like (the shape of the data we accept)
# ----------------------------------------------------------------------
class TransactionRequest(BaseModel):
    account_id: str = Field(..., description="The account making the transaction")
    device_id: str = Field(..., description="The device used for this transaction")
    merchant_id: str = Field(..., description="The merchant receiving payment")
    amount: float = Field(..., description="Transaction amount")


# ----------------------------------------------------------------------
# Hard-coded RULES (on top of the AI model) - the spec calls for this:
# rules can force a decline regardless of what the AI thinks, e.g. a
# sanctions list hit. We simulate a tiny sanctions list here.
# ----------------------------------------------------------------------
SANCTIONED_ACCOUNTS = set()  # (empty for now - you could add fake IDs here to test)


def compute_features(req: TransactionRequest) -> dict:
    """Recreate the same 'clues' used during training, for this ONE new transaction."""

    acct_history = transactions_history[transactions_history["account_id"] == req.account_id]
    device_accounts = transactions_history[transactions_history["device_id"] == req.device_id]["account_id"].nunique()

    account_txn_number = len(acct_history) + 1

    recent_window = acct_history[
        acct_history["timestamp"] >= (pd.Timestamp.now() - pd.Timedelta(hours=1))
    ]
    txns_last_1h = len(recent_window) + 1  # +1 to include this new transaction

    if len(acct_history) > 0:
        avg_amount = acct_history["amount"].mean()
        std_amount = acct_history["amount"].std() or 1
    else:
        avg_amount = req.amount
        std_amount = 1

    amount_zscore = (req.amount - avg_amount) / (std_amount + 1e-6)
    is_high_amount = 1 if req.amount > 3000 else 0

    return {
        "amount": req.amount,
        "device_shared_by_n_accounts": device_accounts if device_accounts > 0 else 1,
        "account_txn_number": account_txn_number,
        "txns_last_1h": txns_last_1h,
        "amount_zscore": amount_zscore,
        "is_high_amount": is_high_amount,
    }


@app.get("/")
def home():
    return {"message": "AegisFlow Fraud Scoring API is running. Go to /docs to try it."}


@app.post("/score")
def score_transaction(req: TransactionRequest):
    # --- Step 1: Hard rules first (these can force a decline no matter what) ---
    rule_hits = []
    if req.account_id in SANCTIONED_ACCOUNTS:
        rule_hits.append("SANCTIONED_ACCOUNT")
        return {
            "decision": "decline",
            "fraud_probability": 1.0,
            "rule_hits": rule_hits,
            "reason_codes": ["Account is on the sanctions list - automatic decline."],
        }

    # --- Step 2: Compute the same features used in training ---
    features = compute_features(req)
    X = pd.DataFrame([features])[FEATURE_COLUMNS]

    # --- Step 3: Run the AI model ---
    fraud_probability = float(model.predict_proba(X)[0, 1])

    # --- Step 4: Turn the probability into a decision ---
    if fraud_probability >= 0.7:
        decision = "decline"
    elif fraud_probability >= 0.3:
        decision = "review"
    else:
        decision = "approve"

    # --- Step 5: Explain WHY, using SHAP (this is the "reason codes") ---
    shap_values = explainer.shap_values(X)[0]
    explanations = sorted(
        zip(FEATURE_COLUMNS, shap_values),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )
    reason_codes = [
        f"{name} = {features[name]:.2f} ({'raises' if val > 0 else 'lowers'} fraud risk)"
        for name, val in explanations[:3]  # top 3 most important reasons
    ]

    return {
        "decision": decision,
        "fraud_probability": round(fraud_probability, 4),
        "rule_hits": rule_hits,
        "reason_codes": reason_codes,
        "features_used": features,
    }