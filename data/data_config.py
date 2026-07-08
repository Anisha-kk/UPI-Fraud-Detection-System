from datetime import datetime

N_USERS = 10000
N_TRANSACTIONS = 200000
INTENT_THRESHOLD = 3.8
RISK_THRESHOLD = 6.0

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)

CITIES = [
    "Bangalore",
    "Kochi",
    "Mumbai",
    "Delhi",
    "Chennai",
    "Hyderabad",
    "Pune",
    "Kolkata",
    "Ahmedabad"
]

MERCHANT_CATEGORIES = [
    "Food",
    "Grocery",
    "Fashion",
    "Electronics",
    "Entertainment",
    "Travel",
    "Healthcare"
]

UPI_DOMAINS = [
    "@okaxis",
    "@ybl",
    "@oksbi",
    "@okhdfcbank",
    "@paytm",
    "@okicici"
]

UPI_DOMAIN_RISK = {
    "oksbi": 0.1,
    "okhdfcbank": 0.12,
    "paytm": 0.25,
    "ybl": 0.2
}

FRAUD_TYPES = [
    "account_takeover",
    "phishing",
    "mule_account",
    "transaction_splitting",
    "merchant_fraud"
]

FESTIVAL_PERIODS = [
    ("2025-01-10", "2025-01-15"),  # Pongal / Makar Sankranti
    ("2025-01-22", "2025-01-26"),  # Republic day
    ("2025-03-10", "2025-03-15"),  # Holi
    ("2025-08-10", "2025-08-15"),  # Independence Day sales
    ("2025-09-25", "2025-10-03"),  # Navratri
    ("2025-10-15", "2025-10-24"),  # Diwali
    ("2025-12-20", "2025-12-31"),  # Christmas + New Year
]
