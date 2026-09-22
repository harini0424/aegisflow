"""
AegisFlow - Stage 2 & 3: Feature Engineering + Fraud Detection Model Training
================================================================================
This script does two things:

1. FEATURE ENGINEERING: turns raw transactions into "clues" (features) the AI
   can learn from. Examples of clues:
   - How many transactions has this account made recently? (velocity)
   - How many different accounts share this same device? (device sharing - a
     classic fraud-ring signal)
   - Is this amount unusually large compared to what this account usually spends?

2. MODEL TRAINING: trains an XGBoost model (a powerful, industry-standard
   method) to predict fraud probability using those clues, then measures how
   good it is, and adds SHAP so we can explain WHY it made each decision.

Run this file with:  python src/train_model.py
"""

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

print("Loading data...")
transactions = pd.read_csv("data/transactions.csv", parse_dates=["timestamp"])
transactions = transactions.sort_values("timestamp").reset_index(drop=True)

# ----------------------------------------------------------------------
# STEP 1: Feature Engineering (creating the "clues")
# ----------------------------------------------------------------------
print("Engineering features...")

# --- Clue 1: How many other accounts have used this same device? ---
# (a big number here is suspicious - could be a fraud ring sharing one phone)
device_account_counts = transactions.groupby("device_id")["account_id"].nunique()
transactions["device_shared_by_n_accounts"] = transactions["device_id"].map(device_account_counts)

# --- Clue 2: How many transactions has this account made so far? ---
transactions["account_txn_number"] = transactions.groupby("account_id").cumcount() + 1

# --- Clue 3: How many transactions has this account made in the last hour? ---
# (a burst of many transactions quickly is a classic fraud signal)
transactions = transactions.sort_values(["account_id", "timestamp"])
transactions = transactions.set_index("timestamp")
velocity = (
    transactions.groupby("account_id")["transaction_id"]
    .rolling("1h")
    .count()
    .droplevel(0)
)
transactions["txns_last_1h"] = velocity.values
transactions = transactions.reset_index()
transactions = transactions.sort_values("timestamp").reset_index(drop=True)

# --- Clue 4: How unusual is this amount compared to the account's normal spending? ---
account_avg_amount = transactions.groupby("account_id")["amount"].transform("mean")
account_std_amount = transactions.groupby("account_id")["amount"].transform("std").fillna(1)
transactions["amount_zscore"] = (transactions["amount"] - account_avg_amount) / (account_std_amount + 1e-6)

# --- Clue 5: How many different merchants has this account used recently? ---
transactions["is_high_amount"] = (transactions["amount"] > 3000).astype(int)

print("Feature engineering done. Sample of engineered features:")
print(transactions[[
    "amount", "device_shared_by_n_accounts", "account_txn_number",
    "txns_last_1h", "amount_zscore", "is_high_amount", "is_fraud"
]].head())

# Save the feature table (useful for the real-time scoring service later)
transactions.to_csv("data/transactions_with_features.csv", index=False)

# ----------------------------------------------------------------------
# STEP 2: Prepare data for training
# ----------------------------------------------------------------------

FEATURE_COLUMNS = [
    "amount",
    "device_shared_by_n_accounts",
    "account_txn_number",
    "txns_last_1h",
    "amount_zscore",
    "is_high_amount",
]

X = transactions[FEATURE_COLUMNS].fillna(0)
y = transactions["is_fraud"]

# Split into training data (80%) and testing data (20%) - we test on data
# the model has NEVER seen, so we get an honest measure of how good it is.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining on {len(X_train)} transactions, testing on {len(X_test)} transactions.")

# ----------------------------------------------------------------------
# STEP 3: Train the XGBoost fraud model
# ----------------------------------------------------------------------
print("Training XGBoost model...")

# scale_pos_weight helps the model pay more attention to the rare fraud cases
# (since only ~1% of transactions are fraud, without this the model could
# just always predict "not fraud" and still look "accurate")
fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=fraud_ratio,
    eval_metric="aucpr",
    random_state=42,
)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# STEP 4: Evaluate - how good is it, honestly?
# ----------------------------------------------------------------------
print("\nEvaluating model on unseen test data...")

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

print(f"\nROC-AUC score: {roc_auc:.3f}  (1.0 = perfect, 0.5 = random guessing)")
print(f"PR-AUC score:  {pr_auc:.3f}  (higher = better at catching rare fraud)")
print("\nDetailed report:")
print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))

# ----------------------------------------------------------------------
# STEP 5: SHAP - explain WHY the model makes its decisions
# ----------------------------------------------------------------------
print("Building SHAP explainer (for explaining individual predictions)...")

explainer = shap.TreeExplainer(model)

# Show an example: explain the model's reasoning for one fraud case
fraud_examples = X_test[y_test == 1]
if len(fraud_examples) > 0:
    example = fraud_examples.iloc[[0]]
    shap_values = explainer.shap_values(example)
    print("\nExample explanation for one real fraud case:")
    for feature, value, shap_val in zip(FEATURE_COLUMNS, example.values[0], shap_values[0]):
        direction = "pushes toward FRAUD" if shap_val > 0 else "pushes toward SAFE"
        print(f"  {feature:30s} = {value:>10.2f}   ({direction}, impact: {shap_val:+.3f})")

# ----------------------------------------------------------------------
# STEP 6: Save the trained model so the scoring service can use it later
# ----------------------------------------------------------------------
print("\nSaving trained model to models/fraud_model.pkl ...")

import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(FEATURE_COLUMNS, "models/feature_columns.pkl")

print("\n" + "=" * 50)
print("DONE! Model trained and saved.")
print("=" * 50)
print("Files created:")
print("  data/transactions_with_features.csv")
print("  models/fraud_model.pkl")
print("  models/feature_columns.pkl")