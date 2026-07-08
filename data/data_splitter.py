import pandas as pd
import numpy as np
from config import DATASET, PROD_DATA, TEST_DATA, TRAIN_DATA
from data.data_config import MERCHANT_CATEGORIES

def add_festival_changes(data,dataset_type):#type refers to train/test/production
    #Adds changes in sales, time etc during festive season
    festival_mask = data["is_festival"] == 1
    if dataset_type=="train":#Jan-Sep => Less festive sales
        #Increase in transaction amount
        data.loc[festival_mask,"amount"] *= np.random.uniform(1.2, 1.8, festival_mask.sum())
        # Users transact more frequently
        data.loc[festival_mask,"txn_count_last_1h"] += np.random.randint(1, 3, festival_mask.sum())
        #Merchant category modification - during festive season some categories are purchased more
        probs = [
        0.20,  # Food
        0.20,  # Grocery
        0.15,  # Fashion
        0.10,  # Electronics
        0.10,  # Entertainment
        0.15,  # Travel
        0.10   # Healthcare
    ]
    elif dataset_type=="test":#Oct-Nov = Medium festive sales
        #Increase in transaction amount
        data.loc[festival_mask,"amount"] *= np.random.uniform(1.9, 2.2, festival_mask.sum())
        # Users transact more frequently
        data.loc[festival_mask,"txn_count_last_1h"] += np.random.randint(2, 4, festival_mask.sum())
        probs = [
        0.15,  # Food
        0.20,  # Grocery
        0.20,  # Fashion
        0.20,  # Electronics
        0.10,  # Entertainment
        0.10,  # Travel
        0.05   # Healthcare
    ]
    elif dataset_type=="production":#Dec - Christmas and new year - high sales
        #Increase in transaction amount
        data.loc[festival_mask,"amount"] *= np.random.uniform(2.3, 2.8, festival_mask.sum())
        # Users transact more frequently
        data.loc[festival_mask,"txn_count_last_1h"] += np.random.randint(3, 6, festival_mask.sum())
        probs = [
        0.10,  # Food
        0.15,  # Grocery
        0.25,  # Fashion
        0.25,  # Electronics
        0.15,  # Entertainment
        0.07,  # Travel
        0.03   # Healthcare
    ]
    else:
        print("Invalid type input")
        return None
    
    # More night-time activity for a fraction of data
    night_mask = data[festival_mask].sample(frac=0.4, random_state=42).index
    data.loc[night_mask, "hour_of_day"] = np.random.randint(0, 6, len(night_mask))

    #Modifying merchant category for a sample of data
    festival_rows = data[festival_mask].sample(frac=0.6, random_state=42).index
    data.loc[festival_rows, "merchant_category"] = np.random.choice(
        MERCHANT_CATEGORIES,
        size=len(festival_rows),
        p=probs
    )

    return data

if __name__=="__main__":
    # Load generated dataset
    df = pd.read_csv(DATASET,parse_dates=["timestamp"])

    # Sort by time
    df = df.sort_values("timestamp")

    # Time-based split
    train_df = df[df["timestamp"] < "2025-10-01"].copy()
    test_df = df[(df["timestamp"] >= "2025-10-01") &(df["timestamp"] < "2025-12-01")].copy()
    production_df = df[df["timestamp"] >= "2025-12-01"].copy()

    print("Train:", len(train_df))
    print("test:", len(test_df))
    print("Production:", len(production_df))

    # --------------------------------------------------
    # FEATURE DRIFT
    # --------------------------------------------------
    # Introducing different levels of festive changes in training, test and production data. Production data 
    # will have huge changes compared to train and test data 
    train_df = add_festival_changes(train_df,"train")
    test_df = add_festival_changes(test_df,"test")
    production_df = add_festival_changes(production_df,"production")

    # --------------------------------------------------
    # CONCEPT DRIFT
    # --------------------------------------------------
    fraud_mask = production_df["fraud_label"] == 1
    drift_mask = production_df[fraud_mask].sample(frac=0.3, random_state=42).index #Subset of fraud_mask
    # Fraudsters move to transaction splitting instead of bulk transfer as expected - similar to splitting case in 
    # available data, but values are slightly different
    production_df.loc[drift_mask,"amount"] = np.random.choice([4999, 4998, 4997, 3999],size=len(drift_mask))
    # Increase velocity moderately for fraud transactions
    production_df.loc[drift_mask,"txn_count_last_1h"] += np.random.randint(4,8,size=len(drift_mask))
    production_df.loc[drift_mask,"device_changed"] = 0 #Doing fraud from trusted device
    production_df.loc[drift_mask,"city_changed"] = 0 #Doing fraud from same city
    # Update fraud type label
    production_df.loc[drift_mask,"fraud_type"] = "transaction_splitting_v2"

    # --------------------------------------------------
    # Save datasets
    # --------------------------------------------------
    train_df.to_csv(TRAIN_DATA,index=False)
    test_df.to_csv(TEST_DATA,index=False)
    production_df.to_csv(PROD_DATA,index=False)

    print("\nSaved:")
    print("train.csv")
    print("test.csv")
    print("production.csv")
