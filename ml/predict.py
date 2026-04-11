"""
UPI Fraud Detection - Hybrid Prediction Engine
Combines ML prediction + anomaly detection + rule-based logic.
"""

import numpy as np
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"

# Feature columns must match training order
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


class HybridFraudEngine:
    """
    Hybrid fraud detection engine combining:
    1. ML prediction (Random Forest) - probability
    2. Anomaly detection (Isolation Forest) - anomaly score
    3. Rule-based logic - domain rules
    """

    def __init__(self):
        self.rf_model = None
        self.iso_model = None
        self.scaler = None
        self._load_models()

    def _load_models(self):
        """Load trained models from disk."""
        try:
            self.rf_model = joblib.load(MODEL_DIR / "saved_model.pkl")
            self.iso_model = joblib.load(MODEL_DIR / "isolation_forest.pkl")
            self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
            print("[✓] All models loaded successfully")
        except FileNotFoundError as e:
            print(f"[!] Model not found: {e}")
            print("[!] Run train.py first to train models")
            raise

    def prepare_features(self, transaction):
        """
        Extract and compute features from a raw transaction dict.
        Accepts raw transaction input and maps to feature vector.
        """
        amount = float(transaction.get("amount", 0))
        old_balance_org = float(transaction.get("oldbalanceOrg", amount * 2))
        new_balance_org = float(transaction.get("newbalanceOrig", old_balance_org - amount))
        old_balance_dest = float(transaction.get("oldbalanceDest", 10000))
        new_balance_dest = float(transaction.get("newbalanceDest", old_balance_dest + amount))

        # Map transaction type
        type_map = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
        tx_type = transaction.get("type", "TRANSFER")
        type_encoded = type_map.get(tx_type, 4)

        avg_tx = float(transaction.get("avg_transaction", amount * 0.5))
        account_age = int(transaction.get("account_age", 365))

        features = {
            "log_amount": np.log1p(amount),
            "large_transaction_flag": 1 if amount > 50000 else 0,
            "balance_change_org": old_balance_org - new_balance_org,
            "balance_change_dest": new_balance_dest - old_balance_dest,
            "balance_ratio": min(amount / max(old_balance_org, 1), 100),
            "type_encoded": type_encoded,
            "transactions_per_user": int(transaction.get("transactions_per_user", 5)),
            "transactions_last_1hr": int(transaction.get("transactions_last_1hr", 1)),
            "device_type_encoded": int(transaction.get("device_type_encoded", 0)),
            "device_change_flag": int(transaction.get("device_change_flag", 0)),
            "geo_distance": float(transaction.get("geo_distance", 10)),
            "impossible_travel_flag": int(transaction.get("impossible_travel_flag", 0)),
            "avg_transaction": avg_tx,
            "deviation_from_avg": abs(amount - avg_tx),
            "account_age": account_age,
            "hour_of_day": int(transaction.get("hour_of_day", 12)),
            "is_night_flag": int(transaction.get("is_night_flag", 0)),
            "OTP_flag": int(transaction.get("OTP_flag", 1)),
            "QR_flag": int(transaction.get("QR_flag", 0)),
            "login_attempts": int(transaction.get("login_attempts", 1)),
        }
        return features

    def apply_rules(self, features, transaction):
        """
        Rule-based fraud detection — domain-specific UPI rules.
        Returns a score (0–1) and list of triggered reasons.
        """
        score = 0.0
        reasons = []

        amount = float(transaction.get("amount", 0))

        # Rule 1: High amount anomaly
        if amount > 100000:
            score += 0.25
            reasons.append("High transaction amount (>₹1,00,000)")
        elif amount > 50000:
            score += 0.15
            reasons.append("Large transaction amount (>₹50,000)")

        # Rule 2: Rapid transactions
        if features["transactions_last_1hr"] > 5:
            score += 0.2
            reasons.append(f"High velocity: {features['transactions_last_1hr']} transactions in 1hr window")

        # Rule 3: Device switching
        if features["device_change_flag"] == 1:
            score += 0.2
            reasons.append("Recent device change detected")

        # Rule 4: Impossible travel
        if features["impossible_travel_flag"] == 1:
            score += 0.3
            reasons.append(f"Impossible travel detected (distance: {features['geo_distance']:.0f} km)")

        # Rule 5: Night transaction
        if features["is_night_flag"] == 1:
            score += 0.1
            reasons.append("Transaction during unusual hours (11PM–5AM)")

        # Rule 6: New account with large transaction
        if features["account_age"] < 30 and amount > 25000:
            score += 0.2
            reasons.append(f"New account ({features['account_age']} days) with large transaction")

        # Rule 7: Multiple login attempts
        if features["login_attempts"] >= 4:
            score += 0.2
            reasons.append(f"Suspicious login pattern ({features['login_attempts']} attempts)")

        # Rule 8: Amount far exceeds average
        if features["deviation_from_avg"] > features["avg_transaction"] * 3:
            score += 0.15
            reasons.append("Transaction amount significantly above user average")

        # Rule 9: QR code with device change
        if features["QR_flag"] == 1 and features["device_change_flag"] == 1:
            score += 0.15
            reasons.append("QR code transaction from a new device")

        # Rule 10: Balance drain pattern
        if features["balance_ratio"] > 0.9:
            score += 0.2
            reasons.append("Transaction drains >90% of account balance")

        return min(score, 1.0), reasons

    def predict(self, transaction):
        """
        Hybrid prediction combining ML + anomaly + rules.
        Returns comprehensive fraud analysis.
        """
        # 1. Prepare features
        features = self.prepare_features(transaction)
        feature_vector = np.array([[features[col] for col in FEATURE_COLUMNS]])

        # 2. Scale features
        feature_scaled = self.scaler.transform(feature_vector)

        # 3. ML prediction (Random Forest)
        ml_probability = float(self.rf_model.predict_proba(feature_scaled)[0][1])
        ml_prediction = int(self.rf_model.predict(feature_scaled)[0])

        # 4. Anomaly detection (Isolation Forest)
        anomaly_score_raw = float(self.iso_model.decision_function(feature_scaled)[0])
        # Convert to 0-1 scale (more negative = more anomalous)
        anomaly_score = max(0, min(1, 0.5 - anomaly_score_raw))

        # 5. Rule-based detection
        rule_score, reasons = self.apply_rules(features, transaction)

        # 6. Combine scores (weighted ensemble)
        # ML model: 50%, Anomaly: 20%, Rules: 30%
        fraud_probability = (
            ml_probability * 0.50
            + anomaly_score * 0.20
            + rule_score * 0.30
        )
        fraud_probability = round(min(fraud_probability, 1.0), 4)

        # 7. Determine risk score (0–100)
        risk_score = round(fraud_probability * 100, 1)

        # 8. Determine label
        if fraud_probability >= 0.6:
            fraud_label = "FRAUD"
        elif fraud_probability >= 0.3:
            fraud_label = "SUSPICIOUS"
        else:
            fraud_label = "SAFE"

        # 9. Add ML-based reasons
        if ml_probability > 0.5:
            reasons.insert(0, f"ML model flags high fraud probability ({ml_probability:.1%})")
        if anomaly_score > 0.5:
            reasons.insert(0, f"Anomaly detection flags unusual pattern (score: {anomaly_score:.2f})")

        if not reasons:
            reasons.append("No risk factors detected")

        # 10. Feature importance for this prediction
        feature_importance = {}
        if hasattr(self.rf_model, "feature_importances_"):
            importances = self.rf_model.feature_importances_
            top_indices = np.argsort(importances)[-8:][::-1]
            for idx in top_indices:
                feature_importance[FEATURE_COLUMNS[idx]] = round(float(importances[idx]), 4)

        return {
            "fraud_probability": fraud_probability,
            "fraud_label": fraud_label,
            "risk_score": risk_score,
            "reasons": reasons,
            "ml_probability": round(ml_probability, 4),
            "anomaly_score": round(anomaly_score, 4),
            "rule_score": round(rule_score, 4),
            "ml_prediction": ml_prediction,
            "feature_importance": feature_importance,
            "features_used": features,
        }


# Singleton engine instance
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = HybridFraudEngine()
    return _engine


if __name__ == "__main__":
    # Test prediction
    engine = get_engine()

    print("\n--- TEST: Normal Transaction ---")
    result = engine.predict({
        "amount": 500,
        "type": "PAYMENT",
        "account_age": 730,
        "avg_transaction": 450,
        "login_attempts": 1,
        "device_change_flag": 0,
        "is_night_flag": 0,
        "QR_flag": 0,
    })
    print(f"Label: {result['fraud_label']}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Reasons: {result['reasons']}")

    print("\n--- TEST: Fraudulent Transaction ---")
    result = engine.predict({
        "amount": 250000,
        "type": "TRANSFER",
        "account_age": 5,
        "avg_transaction": 2000,
        "login_attempts": 5,
        "device_change_flag": 1,
        "is_night_flag": 1,
        "impossible_travel_flag": 1,
        "geo_distance": 2000,
        "transactions_last_1hr": 8,
        "QR_flag": 1,
    })
    print(f"Label: {result['fraud_label']}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Reasons: {result['reasons']}")
