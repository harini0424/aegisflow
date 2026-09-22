"""
AegisFlow - Stage 1: Synthetic Transaction Generator
======================================================
This script creates a FAKE but realistic banking dataset:
- Accounts, devices, merchants (the "entities")
- Thousands of everyday transactions (normal behavior)
- A few hidden FRAUD RINGS (accounts sharing devices, doing suspicious things)
- A few hidden LAUNDERING PATTERNS (smurfing = many small deposits to hide a big amount)

Why we need this: real bank data is private/illegal to get for a class project,
so we generate our own realistic version, and we ALSO save the "ground truth"
(which transactions are actually fraud) so we can later check how good our AI is.

Run this file with:  python src/generate_transactions.py
"""

import random
import uuid
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)  # makes results repeatable every time you run it

# ----------------------------------------------------------------------
# STEP 1: Create a pool of normal accounts, devices, and merchants
# ----------------------------------------------------------------------

NUM_ACCOUNTS = 500
NUM_DEVICES = 400
NUM_MERCHANTS = 60

print("Creating accounts, devices, and merchants...")

accounts = [
    {
        "account_id": str(uuid.uuid4()),
        "holder_name": fake.name(),
        "risk_tier": random.choices(["low", "medium", "high"], weights=[0.8, 0.15, 0.05])[0],
        "opened_at": fake.date_between(start_date="-3y", end_date="-30d"),
    }
    for _ in range(NUM_ACCOUNTS)
]

devices = [
    {"device_id": str(uuid.uuid4()), "fingerprint": fake.sha1()[:12]}
    for _ in range(NUM_DEVICES)
]

merchants = [
    {"merchant_id": str(uuid.uuid4()), "name": fake.company(), "mcc": random.choice(
        ["5411", "5812", "5999", "4111", "5967", "7995"]  # grocery, restaurant, retail, transport, online, gambling
    )}
    for _ in range(NUM_MERCHANTS)
]

# Give each account 1-3 "usual" devices (their phone, laptop, etc.)
account_devices = {}
for acc in accounts:
    account_devices[acc["account_id"]] = random.sample(devices, k=random.randint(1, 3))

# ----------------------------------------------------------------------
# STEP 2: Generate NORMAL transactions (the majority - not fraud)
# ----------------------------------------------------------------------

print("Generating normal transactions...")

transactions = []
start_time = datetime.now() - timedelta(days=30)


def add_transaction(account_id, device_id, merchant_id, amount, ts, is_fraud, fraud_type=None):
    transactions.append({
        "transaction_id": str(uuid.uuid4()),
        "account_id": account_id,
        "device_id": device_id,
        "merchant_id": merchant_id,
        "amount": round(amount, 2),
        "timestamp": ts,
        "is_fraud": is_fraud,       # <-- our "ground truth" answer key
        "fraud_type": fraud_type,   # e.g. "smurfing", "ring", None for normal
    })


NUM_NORMAL_TRANSACTIONS = 8000

for _ in range(NUM_NORMAL_TRANSACTIONS):
    acc = random.choice(accounts)
    device = random.choice(account_devices[acc["account_id"]])  # their usual device
    merchant = random.choice(merchants)
    amount = round(random.lognormvariate(4, 1), 2)  # realistic spending pattern (mostly small, few large)
    ts = start_time + timedelta(seconds=random.randint(0, 30 * 24 * 3600))
    add_transaction(acc["account_id"], device["device_id"], merchant["merchant_id"], amount, ts, is_fraud=0)

# ----------------------------------------------------------------------
# STEP 3: Inject a FRAUD RING (accounts secretly sharing one device)
# ----------------------------------------------------------------------
# Real-world pattern: a fraudster controls several "mule" accounts from
# one phone/laptop, and fires off many transactions in a short burst.

print("Injecting fraud ring pattern...")

NUM_RINGS = 5
for _ in range(NUM_RINGS):
    ring_accounts = random.sample(accounts, k=random.randint(4, 7))
    shared_device = random.choice(devices)
    ring_merchant = random.choice(merchants)
    burst_start = start_time + timedelta(seconds=random.randint(0, 30 * 24 * 3600))

    for acc in ring_accounts:
        # several rapid transactions within a few minutes - a classic fraud signal
        for _ in range(random.randint(2, 4)):
            ts = burst_start + timedelta(seconds=random.randint(0, 600))
            amount = round(random.uniform(4000, 20000), 2)  # unusually large amounts
            add_transaction(
                acc["account_id"], shared_device["device_id"], ring_merchant["merchant_id"],
                amount, ts, is_fraud=1, fraud_type="ring",
            )

# ----------------------------------------------------------------------
# STEP 4: Inject SMURFING (money laundering: split a big amount into
# many small deposits across several accounts to avoid detection)
# ----------------------------------------------------------------------

print("Injecting smurfing (laundering) pattern...")

NUM_SMURFING_CASES = 4
for _ in range(NUM_SMURFING_CASES):
    smurf_accounts = random.sample(accounts, k=random.randint(5, 8))
    total_to_launder = random.uniform(80000, 150000)
    per_account = total_to_launder / len(smurf_accounts)
    launder_start = start_time + timedelta(seconds=random.randint(0, 30 * 24 * 3600))

    for acc in smurf_accounts:
        device = random.choice(account_devices[acc["account_id"]])
        merchant = random.choice(merchants)
        # each account gets one deposit just under a "suspicious amount" threshold
        amount = round(per_account * random.uniform(0.9, 1.0), 2)
        ts = launder_start + timedelta(hours=random.randint(0, 48))
        add_transaction(
            acc["account_id"], device["device_id"], merchant["merchant_id"],
            amount, ts, is_fraud=1, fraud_type="smurfing",
        )

# ----------------------------------------------------------------------
# STEP 5: Save everything to files
# ----------------------------------------------------------------------

print("Saving files to the data/ folder...")

df_accounts = pd.DataFrame(accounts)
df_devices = pd.DataFrame(devices)
df_merchants = pd.DataFrame(merchants)
df_transactions = pd.DataFrame(transactions).sort_values("timestamp").reset_index(drop=True)

df_accounts.to_csv("data/accounts.csv", index=False)
df_devices.to_csv("data/devices.csv", index=False)
df_merchants.to_csv("data/merchants.csv", index=False)
df_transactions.to_csv("data/transactions.csv", index=False)

# ----------------------------------------------------------------------
# Summary printout so you can see it worked
# ----------------------------------------------------------------------

print("\n" + "=" * 50)
print("DONE! Here's what was created:")
print("=" * 50)
print(f"Accounts:      {len(df_accounts)}")
print(f"Devices:       {len(df_devices)}")
print(f"Merchants:     {len(df_merchants)}")
print(f"Transactions:  {len(df_transactions)}")
print(f"  - Normal:    {(df_transactions['is_fraud'] == 0).sum()}")
print(f"  - Fraud:     {(df_transactions['is_fraud'] == 1).sum()}")
print(f"      - ring:      {(df_transactions['fraud_type'] == 'ring').sum()}")
print(f"      - smurfing:  {(df_transactions['fraud_type'] == 'smurfing').sum()}")
print("\nFiles saved in the data/ folder:")
print("  data/accounts.csv")
print("  data/devices.csv")
print("  data/merchants.csv")
print("  data/transactions.csv")