from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_cols = [
    "sender_city",
    "receiver_city",
    "merchant_category",
    'sender_upi_domain', 
    'receiver_upi_domain'
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat",OneHotEncoder(handle_unknown="ignore", drop="first"),categorical_cols)],
    remainder="passthrough")
