"""
UPI Fraud Detection - Model Training & Evaluation
Trains Logistic Regression, Random Forest (PRIMARY), SVM, Isolation Forest, LOF.
Handles class imbalance with SMOTE. Evaluates on holdout test set with no data leakage.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

from preprocess import load_and_process, FEATURE_COLUMNS, TARGET_COLUMN

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR = Path(__file__).parent / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_evaluate():
    """Full training pipeline with strictly NO data leakage."""
    print("=" * 60)
    print("  UPI FRAUD DETECTION - MODEL TRAINING")
    print("=" * 60)

    # 1. Load processed data
    df = load_and_process()
    
    # 4. CHECK DUPLICATES: Remove duplicate rows before training
    initial_shape = df.shape
    df = df.drop_duplicates()
    if df.shape != initial_shape:
        print(f"[DATA] Removed {initial_shape[0] - df.shape[0]} duplicate rows.")
        
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Handle any NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"\n[DATA] Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"[DATA] Class distribution:\n{y.value_counts()}")
    print(f"[DATA] Fraud ratio: {y.mean():.4f}")
    
    print("\n[INFO] Validated Feature Set (No Leakage Features):")
    for feat in FEATURE_COLUMNS:
        print(f"  - {feat}")

    # 2. PREVENT DATA LEAKAGE: Train/test split FIRST!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[SPLIT] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # 7. ADD CROSS VALIDATION: StratifiedKFold (5 folds) with pipeline to prevent SMOTE leak
    print("\n[CV] Performing 5-Fold Stratified Cross-Validation on Random Forest...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42, sampling_strategy=0.5)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight="balanced", n_jobs=-1))
    ])
    cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=kf, scoring='f1')
    print(f"[CV] F1-Scores for 5 folds: {cv_scores}")
    print(f"[CV] Mean F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # 3. Scale features (Fit only on Train to prevent leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. FIX SMOTE USAGE: Apply SMOTE ONLY on training data
    print("\n[SMOTE] Applying SMOTE oversampling on training set only...")
    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    print(f"[SMOTE] After: {pd.Series(y_train_smote).value_counts().to_dict()}")

    # 5. Train final Supervised Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
            class_weight="balanced",
            max_iter=2000,
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n{'-' * 40}")
        print(f"[TRAINING] {name}...")
        model.fit(X_train_smote, y_train_smote) # Fit on resampled
        
        # 6. USE PROPER EVALUATION: Evaluate ONLY on untouched X_test
        y_pred = model.predict(X_test_scaled) # Predict on original test features
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": {
                "TP": int(cm[1][1]),
                "TN": int(cm[0][0]),
                "FP": int(cm[0][1]),
                "FN": int(cm[1][0]),
            },
        }
        trained_models[name] = model

        # 8. OUTPUT REALISTIC RESULTS
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

    # Train Anomaly Detection
    print(f"\n{'-' * 40}")
    print("[TRAINING] Isolation Forest (Anomaly Detection)...")
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.013,
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_scaled) # No SMOTE for anomaly 
    iso_scores = iso_forest.decision_function(X_test_scaled)
    iso_pred = iso_forest.predict(X_test_scaled)
    iso_pred_binary = (iso_pred == -1).astype(int)

    iso_acc = accuracy_score(y_test, iso_pred_binary)
    iso_prec = precision_score(y_test, iso_pred_binary, zero_division=0)
    iso_rec = recall_score(y_test, iso_pred_binary, zero_division=0)
    iso_f1 = f1_score(y_test, iso_pred_binary, zero_division=0)

    results["Isolation Forest"] = {
        "accuracy": round(iso_acc, 4),
        "precision": round(iso_prec, 4),
        "recall": round(iso_rec, 4),
        "f1_score": round(iso_f1, 4),
        "roc_auc": None,
    }
    print(f"  Accuracy:  {iso_acc:.4f}")
    print(f"  Precision: {iso_prec:.4f}")
    print(f"  Recall:    {iso_rec:.4f}")
    print(f"  F1-Score:  {iso_f1:.4f}")

    # Feature importance (from Random Forest)
    rf_model = trained_models["Random Forest"]
    feature_importance = dict(
        zip(FEATURE_COLUMNS, [round(float(x), 4) for x in rf_model.feature_importances_])
    )
    sorted_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # 10. SAVE MODEL (Only after proper evaluation)
    print(f"\n{'=' * 60}")
    print("[SAVING] Models and artifacts...")

    joblib.dump(rf_model, MODEL_DIR / "saved_model.pkl")
    print(f"  -> saved_model.pkl (Random Forest)")

    joblib.dump(trained_models["Logistic Regression"], MODEL_DIR / "logistic_regression.pkl")
    joblib.dump(trained_models["SVM"], MODEL_DIR / "svm_model.pkl")
    joblib.dump(iso_forest, MODEL_DIR / "isolation_forest.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    metrics = {
        "models": results,
        "feature_importance": sorted_importance,
        "primary_model": "Random Forest",
        "feature_columns": FEATURE_COLUMNS,
        "training_samples": int(X_train_smote.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "fraud_ratio": round(float(y.mean()), 4),
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> metrics.json")

    # Generate visualization charts
    generate_charts(results, sorted_importance, y_test, X_test_scaled, trained_models)

    print(f"\n{'=' * 60}")
    print("[OK] TRAINING COMPLETE")
    print(f"    Primary model: Random Forest")
    print(f"    Final F1: {results['Random Forest']['f1_score']}")
    print(f"    Final AUC: {results['Random Forest']['roc_auc']}")
    print(f"{'=' * 60}")

    return results


def generate_charts(results, feature_importance, y_test, X_test_scaled, trained_models):
    """Generate all evaluation charts."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- ROC Curves ---
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#6366f1", "#10b981", "#f59e0b"]
    for (name, model), color in zip(trained_models.items(), colors):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = results[name]["roc_auc"]
        ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves — Supervised Models", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.set_facecolor("#fafbff")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> roc_curves.png")

    # --- Confusion Matrix (Random Forest) ---
    rf = trained_models["Random Forest"]
    y_pred = rf.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="PuBuGn",
        ax=ax,
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        annot_kws={"size": 16},
        linewidths=2,
        linecolor="white",
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix — Random Forest Pipeline Fix", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> confusion_matrix.png")

    # --- Feature Importance ---
    top_features = dict(list(feature_importance.items())[:12])
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        list(reversed(top_features.keys())),
        list(reversed(top_features.values())),
        color=["#6366f1" if i % 2 == 0 else "#818cf8" for i in range(len(top_features))],
        edgecolor="none",
        height=0.6,
    )
    ax.set_xlabel("Importance", fontsize=13)
    ax.set_title("Feature Importance — Random Forest (Top 12)", fontsize=16, fontweight="bold")
    ax.set_facecolor("#fafbff")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> feature_importance.png")


if __name__ == "__main__":
    train_and_evaluate()
