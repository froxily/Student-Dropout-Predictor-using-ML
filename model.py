"""
model.py — Model Training & Evaluation for Student Dropout Prediction
======================================================================
Trains Logistic Regression, Random Forest, and Decision Tree classifiers,
prints detailed metrics, plots Random Forest feature importance, and
saves the best model as ``best_model.pkl``.
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
DATASET_PATH  = os.path.join(os.path.dirname(__file__), "dataset.csv")
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "best_model.pkl")
OUT_DIR       = os.path.join(os.path.dirname(__file__), "plots")

# Label mapping for Target column
TARGET_MAP    = {"Dropout": 0, "Graduate": 1, "Enrolled": 2}
TARGET_NAMES  = ["Dropout", "Graduate", "Enrolled"]


# ── data helpers ───────────────────────────────────────────────────────────

def load_and_preprocess(path: str = DATASET_PATH):
    """Load ``dataset.csv``, encode the target, scale features, and split.

    The ``Target`` column is encoded as:
    - Dropout  → 0
    - Graduate → 1
    - Enrolled → 2

    Parameters
    ----------
    path : str
        Path to the CSV dataset file.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test, feature_names)``
    """
    df = pd.read_csv(path, delimiter=";")

    # Encode target
    df["Target"] = df["Target"].map(TARGET_MAP)

    # Separate features and label
    X = df.drop(columns=["Target"])
    y = df["Target"]

    feature_names = X.columns.tolist()

    # Handle any remaining non-numeric columns with label encoding
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Scale features (important for Logistic Regression)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 80 / 20 split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Training samples : {len(X_train):,}")
    print(f"  Test samples     : {len(X_test):,}")
    print(f"  Features         : {len(feature_names)}\n")

    return X_train, X_test, y_train, y_test, feature_names, scaler


# ── evaluation helpers ─────────────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test) -> float:
    """Print accuracy, classification report, and confusion matrix for *model*.

    Parameters
    ----------
    name : str
        Human-readable model name used in headers.
    model : sklearn estimator
        A fitted scikit-learn classifier.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True labels for the test set.

    Returns
    -------
    float
        Accuracy score on the test set.
    """
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'─' * 60}")
    print(f"  MODEL : {name}")
    print(f"{'─' * 60}")
    print(f"  Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f} %)")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES,
                                zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    cm_df = pd.DataFrame(cm,
                         index=[f"Actual {n}"   for n in TARGET_NAMES],
                         columns=[f"Pred {n}"   for n in TARGET_NAMES])
    print(cm_df.to_string())
    print()

    return accuracy


def plot_confusion_matrix(name: str, model, X_test, y_test,
                          out_dir: str = OUT_DIR) -> None:
    """Save a styled confusion-matrix heatmap for *model*.

    Parameters
    ----------
    name : str
        Model name used in the plot title and filename.
    model : sklearn estimator
        A fitted classifier.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True test labels.
    out_dir : str
        Output directory for the PNG file.
    """
    os.makedirs(out_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
                linewidths=0.5, linecolor="#ddd", ax=ax,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13,
                 fontweight="bold", color="#2C3E50", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, color="#555")
    ax.set_ylabel("True Label",      fontsize=11, color="#555")
    fig.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    save_path = os.path.join(out_dir, f"confusion_matrix_{safe_name}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  📊 Confusion matrix saved: {save_path}")


def plot_feature_importance(rf_model, feature_names: list,
                            top_n: int = 15,
                            out_dir: str = OUT_DIR) -> None:
    """Plot and save the top-*n* feature importances from a Random Forest.

    Parameters
    ----------
    rf_model : RandomForestClassifier
        A fitted Random Forest estimator.
    feature_names : list of str
        Column names matching the training feature matrix.
    top_n : int
        Number of top features to display (default 15).
    out_dir : str
        Output directory for the PNG file.
    """
    os.makedirs(out_dir, exist_ok=True)

    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    # Colour gradient from teal → purple
    palette = sns.color_palette("viridis", top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(top_n), top_values[::-1],
                   color=palette[::-1], edgecolor="white", linewidth=0.8)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=12, color="#555")
    ax.set_title(f"Random Forest — Top {top_n} Feature Importances",
                 fontsize=14, fontweight="bold", color="#2C3E50", pad=14)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate values
    for bar, val in zip(bars, top_values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color="#333")

    fig.tight_layout()
    save_path = os.path.join(out_dir, "rf_feature_importance.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  📊 Feature importance plot saved: {save_path}\n")


# ── training pipeline ──────────────────────────────────────────────────────

def train_models():
    """Full training pipeline: load data → train → evaluate → save best model.

    Three classifiers are trained and evaluated:
    1. Logistic Regression
    2. Random Forest
    3. Decision Tree

    The model with the highest test accuracy is serialised to
    ``best_model.pkl`` via :mod:`joblib`.

    Returns
    -------
    tuple
        ``(best_model, scaler, feature_names)`` for downstream inference.
    """
    print("\n🤖 Starting Model Training Pipeline …\n")
    print("📦 Loading & Preprocessing Data …")

    X_train, X_test, y_train, y_test, feature_names, scaler = \
        load_and_preprocess()

    # ── define classifiers ────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced",
            max_depth=12, min_samples_split=5
        ),
    }

    # ── train & evaluate ──────────────────────────────────────────────────
    results      = {}   # name → accuracy
    fitted_models = {}  # name → trained model

    for name, clf in models.items():
        print(f"⚙️  Training: {name} …")
        clf.fit(X_train, y_train)
        acc = evaluate_model(name, clf, X_test, y_test)
        plot_confusion_matrix(name, clf, X_test, y_test)
        results[name]       = acc
        fitted_models[name] = clf

    # ── feature importance (Random Forest only) ───────────────────────────
    print("\n📈 Plotting Random Forest Feature Importance …")
    plot_feature_importance(fitted_models["Random Forest"], feature_names)

    # ── select and save best model ────────────────────────────────────────
    best_name  = max(results, key=results.get)
    best_model = fitted_models[best_name]
    best_acc   = results[best_name]

    print("=" * 60)
    print(f"🏆 Best Model : {best_name}  (Accuracy: {best_acc * 100:.2f} %)")
    print("=" * 60 + "\n")

    # Persist model + scaler + feature list together
    payload = {
        "model":         best_model,
        "scaler":        scaler,
        "feature_names": feature_names,
        "model_name":    best_name,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"💾 Best model saved to: {MODEL_PATH}\n")

    return best_model, scaler, feature_names


if __name__ == "__main__":
    train_models()
