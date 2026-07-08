#To store the pydantic models
from pydantic import BaseModel, Field
from datetime import datetime
class TransactionInput(BaseModel):
    txn_id: str
    timestamp: datetime
    sender_id: int= Field(ge=0)
    receiver_id: int= Field(ge=0)
    sender_upi_domain: str
    receiver_upi_domain: str
    sender_account_age_days: int = Field(gt=0)#Greater than 0
    receiver_account_age_days: int = Field(gt=0)
    amount: float = Field(gt=0, le=1e7)
    sender_city: str
    receiver_city: str
    device_changed: int= Field(ge=0, le=1)#[0,1]
    ip_changed: int = Field(ge=0, le=1)
    city_changed: int = Field(ge=0, le=1)
    failed_logins_last_24h: int = Field(ge=0)
    txn_count_last_1h: int = Field(ge=0)
    txn_count_last_24h: int = Field(ge=0)
    is_new_beneficiary: int = Field(ge=0, le=1)
    merchant_category: str
    amount_to_average_ratio: float = Field(ge=0)
    distance_from_last_transaction: int = Field(ge=0)
    payday_intensity: float = Field(ge=0)
    month: int = Field(ge=1, le=12)
    day_of_month: int = Field(ge=1, le=31)
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    is_festival: int = Field(ge=0, le=1)