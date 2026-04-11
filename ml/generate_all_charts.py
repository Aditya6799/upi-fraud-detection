import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve

from preprocess import load_and_process, FEATURE_COLUMNS, TARGET_COLUMN

MODEL_DIR = Path(__file__).parent / "models"
CHARTS_DIR = Path(__file__).parent / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_old_charts():
    # 1. Load data
    print("Loading data...")
    df = load_and_process()
    df = df.drop_duplicates()
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    # Fill nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 2. Re-create exact split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Load Models
    print("Loading models...")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    X_test_scaled = scaler.transform(X_test)
    
    rf = joblib.load(MODEL_DIR / "saved_model.pkl")
    lr = joblib.load(MODEL_DIR / "logistic_regression.pkl")
    svm = joblib.load(MODEL_DIR / "svm_model.pkl")
    
    trained_models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "SVM": svm
    }

    with open(MODEL_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)
        
    results = metrics["models"]
    feature_importance = metrics["feature_importance"]

    plt.style.use("seaborn-v0_8-whitegrid")

    # --- ROC Curves ---
    print("Generating ROC Curves...")
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

    # --- Confusion Matrix (Random Forest) ---
    print("Generating Confusion Matrix...")
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

    # --- Feature Importance ---
    print("Generating Feature Importance...")
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

    # --- MODEL COMPARISON ---
    print("Generating Model Comparison...")
    target_models = ["Logistic Regression", "Random Forest", "SVM"]
    
    labels = []
    acc = []
    prec = []
    rec = []
    f1 = []
    auc = []
    
    for m in target_models:
        if m in results:
            labels.append(m)
            acc.append(results[m]["accuracy"])
            prec.append(results[m]["precision"])
            rec.append(results[m]["recall"])
            f1.append(results[m]["f1_score"])
            auc.append(results[m]["roc_auc"] if results[m]["roc_auc"] else 0)

    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - 2*width, acc, width, label='Accuracy', color='#6366f1')
    ax.bar(x - width, prec, width, label='Precision', color='#10b981')
    ax.bar(x, rec, width, label='Recall', color='#f59e0b')
    ax.bar(x + width, f1, width, label='F1-Score', color='#ef4444')
    ax.bar(x + 2*width, auc, width, label='ROC-AUC', color='#8b5cf6')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison — All Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("DONE recreating all files!")


if __name__ == "__main__":
    generate_old_charts()
