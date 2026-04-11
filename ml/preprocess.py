"""
UPI Fraud Detection - Data Preprocessing & Feature Engineering
Loads PaySim dataset and engineers UPI-specific features.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PROCESSED_FILE = DATA_DIR / "processed_upi_data.csv"


def download_paysim():
    """Download PaySim dataset from Kaggle using kagglehub."""
    raw_file = DATA_DIR / "paysim.csv"
    if raw_file.exists():
        print(f"[OK] PaySim dataset already exists at {raw_file}")
        return raw_file

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
        print("[>>] Downloading PaySim dataset from Kaggle...")
        path = kagglehub.dataset_download("ealaxi/paysim1")
        # Find the CSV file in the downloaded path
        for f in Path(path).rglob("*.csv"):
            import shutil
            shutil.copy(f, raw_file)
            print(f"[OK] Dataset saved to {raw_file}")
            return raw_file
    except Exception as e:
        print(f"[!] Kaggle download failed: {e}")
        print("[>>] Generating synthetic PaySim-compatible dataset...")
        return generate_synthetic_paysim(raw_file)


def generate_synthetic_paysim(output_path):
    """
    Generate a synthetic dataset following PaySim's statistical properties.
    This is used as a fallback when Kaggle download is not available.
    Schema matches PaySim exactly: step, type, amount, nameOrig, oldbalanceOrg,
    newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
    """
    np.random.seed(42)
    n_samples = 200_000  # Smaller than full PaySim but statistically representative
    fraud_ratio = 0.013  # ~1.3% fraud rate matching PaySim

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Transaction types matching PaySim distribution
    types = np.random.choice(
        ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"],
        size=n_legit,
        p=[0.35, 0.34, 0.22, 0.08, 0.01],
    )
    # Fraud only happens on CASH_OUT and TRANSFER (PaySim behavior)
    fraud_types = np.random.choice(["CASH_OUT", "TRANSFER"], size=n_fraud, p=[0.5, 0.5])

    all_types = np.concatenate([types, fraud_types])

    # Generate amounts
    legit_amounts = np.abs(np.random.lognormal(mean=9.5, sigma=2.0, size=n_legit))
    legit_amounts = np.clip(legit_amounts, 100, 500_000)
    fraud_amounts = np.abs(np.random.lognormal(mean=11.5, sigma=1.5, size=n_fraud))
    fraud_amounts = np.clip(fraud_amounts, 5000, 10_000_000)
    all_amounts = np.concatenate([legit_amounts, fraud_amounts])

    # Generate sender/receiver IDs
    def gen_ids(prefix, n):
        import random
        return [f"{prefix}{random.randint(1000000000, 8999999999)}" for _ in range(n)]

    all_orig = gen_ids("C", n_samples)
    all_dest = gen_ids("C", n_samples)

    # Balances
    old_bal_org = np.abs(np.random.lognormal(mean=10, sigma=2, size=n_samples))
    new_bal_org_legit = np.maximum(0, old_bal_org[:n_legit] - legit_amounts)
    new_bal_org_fraud = np.zeros(n_fraud)  # Drained to 0 by fraud
    new_bal_org = np.concatenate([new_bal_org_legit, new_bal_org_fraud])

    old_bal_dest = np.abs(np.random.lognormal(mean=10, sigma=2, size=n_samples))
    new_bal_dest = old_bal_dest + all_amounts

    # Labels
    is_fraud = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)
    is_flagged = (is_fraud == 1) & (all_amounts > 200_000)
    is_flagged = is_flagged.astype(int)

    # Steps (time steps, 1-743 like PaySim)
    steps = np.random.randint(1, 744, size=n_samples)

    df = pd.DataFrame({
        "step": steps,
        "type": all_types,
        "amount": np.round(all_amounts, 2),
        "nameOrig": all_orig,
        "oldbalanceOrg": np.round(old_bal_org, 2),
        "newbalanceOrig": np.round(new_bal_org, 2),
        "nameDest": all_dest,
        "oldbalanceDest": np.round(old_bal_dest, 2),
        "newbalanceDest": np.round(new_bal_dest, 2),
        "isFraud": is_fraud,
        "isFlaggedFraud": is_flagged,
    })

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Synthetic PaySim dataset generated: {output_path} ({len(df)} rows)")
    return output_path


def engineer_upi_features(df):
    """
    Engineer UPI-specific features from base PaySim data.
    Transforms generic mobile money data into UPI-like dataset.
    """
    print("[..] Engineering UPI features...")

    # --- Transaction Features ---
    df["log_amount"] = np.log1p(df["amount"])
    df["large_transaction_flag"] = (df["amount"] > 50000).astype(int)
    df["balance_change_org"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["balance_ratio"] = np.where(
        df["oldbalanceOrg"] > 0,
        df["amount"] / df["oldbalanceOrg"],
        0,
    )
    df["balance_ratio"] = df["balance_ratio"].clip(0, 100)

    # --- Type Encoding ---
    type_map = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
    df["type_encoded"] = df["type"].map(type_map).fillna(0).astype(int)

    fraud_mask = df["isFraud"] == 1

    # Simulated historical velocity (no future leakage)
    df["transactions_per_user"] = np.random.randint(1, 50, size=len(df))
    df.loc[fraud_mask, "transactions_per_user"] = np.random.randint(1, 5, size=fraud_mask.sum())

    # Rolling velocity within time window (using step as proxy for time)
    df = df.sort_values(["nameOrig", "step"])
    df["transactions_last_1hr"] = (
        df.groupby("nameOrig")["step"]
        .transform(lambda x: x.rolling(window=5, min_periods=1).count())
        .astype(int)
    )

    # --- Device Features (Simulated) ---
    np.random.seed(42)
    device_types = ["android", "ios", "web"]
    df["device_type_encoded"] = np.random.choice([0, 1, 2], size=len(df))

    # Device change - higher probability for fraud
    df["device_change_flag"] = 0
    df.loc[fraud_mask, "device_change_flag"] = np.random.choice(
        [0, 1], size=fraud_mask.sum(), p=[0.3, 0.7]
    )
    df.loc[~fraud_mask, "device_change_flag"] = np.random.choice(
        [0, 1], size=(~fraud_mask).sum(), p=[0.95, 0.05]
    )

    # --- Geo Features (Simulated) ---
    # India lat/lng bounds approximately: 8-37 lat, 68-97 lng
    df["latitude"] = np.random.uniform(8.0, 37.0, size=len(df))
    df["longitude"] = np.random.uniform(68.0, 97.0, size=len(df))
    df["prev_latitude"] = df["latitude"] + np.random.normal(0, 0.5, size=len(df))
    df["prev_longitude"] = df["longitude"] + np.random.normal(0, 0.5, size=len(df))

    # Geo distance (Haversine approximation)
    lat_diff = np.radians(df["latitude"] - df["prev_latitude"])
    lng_diff = np.radians(df["longitude"] - df["prev_longitude"])
    a = np.sin(lat_diff / 2) ** 2 + (
        np.cos(np.radians(df["latitude"]))
        * np.cos(np.radians(df["prev_latitude"]))
        * np.sin(lng_diff / 2) ** 2
    )
    df["geo_distance"] = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # Impossible travel: >500km in <2 steps
    # Only ~30% of fraud involves impossible travel (otherwise model perfectly splits 1.0)
    fraud_geo = np.where(np.random.uniform(0, 1, fraud_mask.sum()) < 0.3, 
                         np.random.uniform(500, 5000, fraud_mask.sum()), 
                         np.random.uniform(1, 100, fraud_mask.sum()))
    df.loc[fraud_mask, "geo_distance"] = fraud_geo

    # Some legitimate users might accidentally trigger high geo distance via VPNs
    legit_geo_mask = (~fraud_mask) & (np.random.uniform(0, 1, len(df)) < 0.01)
    df.loc[legit_geo_mask, "geo_distance"] = np.random.uniform(500, 2000, size=legit_geo_mask.sum())
    
    df["impossible_travel_flag"] = (df["geo_distance"] > 500).astype(int)

    # --- Behavioral Features (Simulated avoiding future data leakage) ---
    df["avg_transaction"] = np.clip(df["amount"] * np.random.uniform(0.5, 2.0, size=len(df)), 100, None)
    # Fraudsters typically transact much more than their usual average
    fraud_avg_mult = np.random.uniform(0.1, 0.5, size=fraud_mask.sum())
    df.loc[fraud_mask, "avg_transaction"] = np.clip(df.loc[fraud_mask, "amount"] * fraud_avg_mult, 100, None)
    
    df["account_age"] = np.random.randint(1, 3650, size=len(df))  # days
    # Fraud often happens on newer accounts within first 90 days, but not always!
    fraud_age = np.where(np.random.uniform(0, 1, fraud_mask.sum()) < 0.7,
                         np.random.randint(1, 90, fraud_mask.sum()),
                         np.random.randint(90, 3650, fraud_mask.sum()))
    df.loc[fraud_mask, "account_age"] = fraud_age

    # --- Time Features ---
    df["hour_of_day"] = (df["step"] % 24).astype(int)
    df["is_night_flag"] = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)).astype(int)
    df.loc[fraud_mask, "is_night_flag"] = np.random.choice(
        [0, 1], size=fraud_mask.sum(), p=[0.3, 0.7]
    )

    # --- Security Features ---
    df["OTP_flag"] = np.random.choice([0, 1], size=len(df), p=[0.3, 0.7])
    df["QR_flag"] = np.random.choice([0, 1], size=len(df), p=[0.6, 0.4])
    df["login_attempts"] = np.random.choice([1, 2, 3], size=len(df), p=[0.7, 0.2, 0.1])
    # Fraud tends to have more login attempts & QR scams
    df.loc[fraud_mask, "login_attempts"] = np.random.choice(
        [3, 4, 5, 6], size=fraud_mask.sum(), p=[0.3, 0.3, 0.2, 0.2]
    )
    df.loc[fraud_mask, "QR_flag"] = np.random.choice(
        [0, 1], size=fraud_mask.sum(), p=[0.3, 0.7]
    )

    print(f"[OK] Feature engineering complete. Shape: {df.shape}")
    print(f"    Fraud rate: {df['isFraud'].mean():.4f}")
    return df


# Feature columns used for ML training
FEATURE_COLUMNS = [
    "log_amount",
    "large_transaction_flag",
    "balance_ratio",
    "type_encoded",
    "transactions_per_user",
    "transactions_last_1hr",
    "device_type_encoded",
    "device_change_flag",
    "geo_distance",
    "impossible_travel_flag",
    "avg_transaction",
    "account_age",
    "hour_of_day",
    "is_night_flag",
    "OTP_flag",
    "QR_flag",
    "login_attempts",
]

TARGET_COLUMN = "isFraud"


def load_and_process():
    """Main pipeline: download data → engineer features → save processed."""
    if PROCESSED_FILE.exists():
        print(f"[OK] Loading cached processed data from {PROCESSED_FILE}")
        df = pd.read_csv(PROCESSED_FILE)
        return df

    raw_path = download_paysim()
    df = pd.read_csv(raw_path)
    print(f"[OK] Loaded raw data: {df.shape}")

    df = engineer_upi_features(df)

    # Save processed
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"[OK] Processed data saved to {PROCESSED_FILE}")

    return df


if __name__ == "__main__":
    df = load_and_process()
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Fraud distribution:\n{df[TARGET_COLUMN].value_counts(normalize=True)}")
    print(f"\nSample:\n{df[FEATURE_COLUMNS + [TARGET_COLUMN]].head()}")
