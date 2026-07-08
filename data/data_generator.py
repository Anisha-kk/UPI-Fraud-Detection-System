#Creates simulated UPI transaction data
'''
Steps:
User Profile
      ↓
Transaction Context
      ↓
Fraud Intent (hidden)
      ↓
Fraud Behaviour
      ↓
Observed Features
      ↓
Risk Score
      ↓
Fraud Label
'''
import numpy as np
import pandas as pd
import random
from datetime import timedelta
from config import DATASET
from data.data_config import (
    INTENT_THRESHOLD,
    N_TRANSACTIONS,
    N_USERS,
    RISK_THRESHOLD,
    START_DATE,
    END_DATE,
    CITIES,
    MERCHANT_CATEGORIES,
    UPI_DOMAIN_RISK,
    UPI_DOMAINS,
    FRAUD_TYPES,
    FESTIVAL_PERIODS
)

# -------------------------
#  USER PROFILES
# -------------------------

users = []
for user_id in range(N_USERS):
    users.append({
        "user_id": user_id,
        "home_city": random.choice(CITIES),
        "account_age_days": np.random.randint(30, 2000),
        "base_txn_amount": np.random.lognormal(6.5, 0.7),
        "base_txn_freq": np.random.randint(1, 8),
        "risk_profile": np.random.beta(2, 5),  # inherent risky user tendency
        "upi_domain": random.choice(UPI_DOMAINS)
    })

users_df = pd.DataFrame(users)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sample_time():
    return START_DATE + timedelta(
        seconds=random.randint(0, int((END_DATE - START_DATE).total_seconds()))
    )

# ------------------------
# TRANSACTION GENERATION
# ------------------------
records = []

for txn_id in range(N_TRANSACTIONS):
    sender = users_df.sample(1).iloc[0]
    receiver = users_df.sample(1).iloc[0]
    txn_time = sample_time()
    # -------------------------
    # BASE BEHAVIOR (NORMAL WORLD)
    # -------------------------
    amount = np.random.lognormal(np.log(sender["base_txn_amount"]), 0.6)
    device_changed = np.random.binomial(1, 0.05)
    ip_changed = np.random.binomial(1, 0.06)
    city_changed = np.random.binomial(1, 0.04)
    failed_logins = np.random.poisson(0.2)
    txn_count_last_1h = np.random.poisson(2)
    txn_count_last_24h = np.random.poisson(sender["base_txn_freq"])
    is_new_beneficiary = np.random.binomial(1, 0.2)
    receiver_risk_score = np.clip(np.random.normal(0.3, 0.15), 0, 1)
    merchant_category = random.choice(MERCHANT_CATEGORIES)
    fraud_type = "normal"

    #Adding payday spike in transactions: Salary days are either 1-5 or 25-31
    day = txn_time.day
    if day <= 5:
        payday_intensity = 1 - (day - 1) / 5
    elif day >= 25:
        payday_intensity = (day - 25) / 6
    else:
        payday_intensity = 0
    amount *= (1 + 0.5 * payday_intensity)
    # ---------------------------------
    # LATENT FRAUDSTER INTENT
    # ---------------------------------
    intent_score = (
        2.5 * sender["risk_profile"] +
        0.5 * receiver_risk_score +
        np.random.normal(0,0.7)
        + 0.5 * payday_intensity
    )
    #Calculating fraud/criminal intent from available data
    P_intent = sigmoid(intent_score - INTENT_THRESHOLD)
    fraud_intent = np.random.rand() < P_intent

    #IF there is fraud intent, modify the synthetic transaction to match fraudulent behaviour.
    if fraud_intent:
        fraud_type = random.choice(FRAUD_TYPES)
        #Modifying transaction to match fraud behaviour. After this, fraud label is assigned
        if fraud_type == "account_takeover":
            device_changed = 1
            ip_changed = 1
            city_changed = 1
            failed_logins += np.random.randint(4,10)
            txn_count_last_1h += np.random.randint(8,20)
            amount *= np.random.uniform(3,8)
            is_new_beneficiary = 1

        elif fraud_type == "phishing":
            amount *= np.random.uniform(2,5)
            is_new_beneficiary = 1
            ip_changed = np.random.binomial(1,0.8)

        elif fraud_type == "mule_account":
            receiver_risk_score = min(receiver_risk_score + 0.4,1)
            txn_count_last_24h += np.random.randint(20,70)
            amount *= np.random.uniform(1.5,4)

        elif fraud_type == "transaction_splitting":
            txn_count_last_1h += np.random.randint(15,35)
            amount = np.random.uniform(2000,9000)

        elif fraud_type == "merchant_fraud":
            receiver_risk_score = min(receiver_risk_score + 0.3,1)
            amount *= np.random.uniform(2,5)

        # stealth fraud
        if np.random.rand()<0.25:
            amount *= np.random.uniform(0.8,1.2)
    
    #Legitimate transactions that have anomalies
    if not fraud_intent:
        if np.random.rand()<0.03:
            device_changed=1
        if np.random.rand()<0.04:
            ip_changed=1
        if np.random.rand()<0.03:
            city_changed=1
        if np.random.rand()<0.05:
            amount*=np.random.uniform(1.5,3)      

    #Derived features
    amount_to_average_ratio = amount / sender["base_txn_amount"]
    distance_from_last_transaction = (
        int(np.random.gamma(5,50))
        if city_changed
        else int(np.random.gamma(2,15))
    )
    night_transaction = int(txn_time.hour < 6)

    # -------------------------
    # RISK MODEL
    # -------------------------
    #Younger the account, higher the risk factor
    sender_account_age_risk = np.exp(-sender["account_age_days"] / 365)
    receiver_account_age_risk = np.exp(-receiver["account_age_days"] / 365)

    #Calculating risk score and assigning the fraud label
    risk_score = (
    1.4 * device_changed +
    1.3 * ip_changed +
    1.1 * city_changed +
    1.0 * np.log1p(amount / sender["base_txn_amount"]) +
    0.9 * receiver_risk_score +
    0.8 * is_new_beneficiary +
    0.5 * failed_logins +
    0.7 * sender_account_age_risk +
    0.5 * receiver_account_age_risk +
    0.4 * txn_count_last_1h +
    0.2 * txn_count_last_24h +
    1.0 * sender["risk_profile"] +
    0.3 * UPI_DOMAIN_RISK.get(receiver["upi_domain"], 0.15) +
    0.5 * night_transaction +
    0.6 * payday_intensity +
    0.4 * np.log1p(distance_from_last_transaction) +
    np.random.normal(0, 0.8)
    )   

    '''
    #Fraud intent is the primary driver for fraud labeling. Risk score is a detection signal.
    intent → decides fraud attempt
    action → modifies behavior
    risk → detects anomaly
    detection → determines whether fraud is caught
    '''
    if fraud_intent: 
        risk_score += 2.5 
        fraud_prob = sigmoid(risk_score - RISK_THRESHOLD) 
    else: # normal transactions should rarely become fraud 
        fraud_prob = 0.01 * sigmoid(risk_score - RISK_THRESHOLD) 
    
    #LABELING 
    is_fraud = np.random.rand() < fraud_prob
    
    # -------------------------
    # STORE RECORD
    # -------------------------
    records.append({
        "txn_id": f"TXN{txn_id:08d}",
        "timestamp": txn_time,
        "sender_id": sender["user_id"],
        "receiver_id": receiver["user_id"],
        "sender_upi_domain":sender["upi_domain"],
        "receiver_upi_domain":receiver["upi_domain"],
        "sender_account_age_days": sender["account_age_days"],
        "receiver_account_age_days": receiver["account_age_days"],
        "amount": round(amount, 2),
        "sender_city": sender["home_city"],
        "receiver_city": receiver["home_city"],
        "device_changed": device_changed,
        "ip_changed": ip_changed,
        "city_changed": city_changed,
        "failed_logins_last_24h": failed_logins,
        "txn_count_last_1h": txn_count_last_1h,
        "txn_count_last_24h": txn_count_last_24h,
        "is_new_beneficiary": is_new_beneficiary,
        "merchant_category": merchant_category,
        "amount_to_average_ratio": round(amount_to_average_ratio, 2),
        "distance_from_last_transaction": distance_from_last_transaction,
        "payday_intensity": payday_intensity,
        "month": txn_time.month,
        "day_of_month": txn_time.day,#Day of month -1 to 31
        "hour_of_day": txn_time.hour,#Will include night transaction etc
        "day_of_week": txn_time.weekday(),
        "fraud_type": fraud_type,
        "fraud_label": int(is_fraud)
    })

# =========================
# SAVE
# =========================
df = pd.DataFrame(records)

#Adding festival seasons with respect to Indian Calendar
df["is_festival"] = 0
for start, end in FESTIVAL_PERIODS:
    mask = (
        (df["timestamp"] >= start) &
        (df["timestamp"] <= end)
    )
    df.loc[mask, "is_festival"] = 1

#Saving to csv file
df.to_csv(DATASET,index=False)

print(df["fraud_label"].value_counts())
