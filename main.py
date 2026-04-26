"""
main.py — Entry Point for Student Dropout Prediction
=====================================================
Orchestrates the full pipeline:
  1. Run Exploratory Data Analysis (eda.py)
  2. Train and evaluate three ML models (model.py)
  3. Load the best saved model and run an interactive prediction session
     where the user enters values for 5 key features to receive a
     Dropout-Risk assessment.
"""

import os
import sys
import joblib
import numpy as np

# ── ensure the project directory is on sys.path ────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from eda   import run_eda          # noqa: E402
from model import train_models     # noqa: E402

MODEL_PATH = os.path.join(PROJECT_DIR, "best_model.pkl")

# ── label mapping (must match model.py) ───────────────────────────────────
TARGET_MAP_INV = {0: "Dropout", 1: "Graduate", 2: "Enrolled"}

# ── 5 key features selected for interactive inference ─────────────────────
# These were identified from Random-Forest feature-importance analysis as
# consistently high-impact predictors of student dropout.
KEY_FEATURES = [
    {
        "col":     "Curricular units 2nd sem (approved)",
        "prompt":  "Number of curricular units approved in 2nd semester (e.g. 5)",
        "dtype":   int,
        "min":     0,
        "max":     30,
    },
    {
        "col":     "Curricular units 1st sem (approved)",
        "prompt":  "Number of curricular units approved in 1st semester (e.g. 5)",
        "dtype":   int,
        "min":     0,
        "max":     30,
    },
    {
        "col":     "Curricular units 2nd sem (grade)",
        "prompt":  "Average grade in 2nd semester (0 – 20, e.g. 13.5)",
        "dtype":   float,
        "min":     0.0,
        "max":     20.0,
    },
    {
        "col":     "Tuition fees up to date",
        "prompt":  "Are tuition fees paid up to date? (1 = Yes, 0 = No)",
        "dtype":   int,
        "min":     0,
        "max":     1,
    },
    {
        "col":     "Age at enrollment",
        "prompt":  "Age at enrollment (e.g. 19)",
        "dtype":   int,
        "min":     15,
        "max":     70,
    },
]


# ── helpers ────────────────────────────────────────────────────────────────

def print_banner(text: str, char: str = "=", width: int = 62) -> None:
    """Print a padded banner line around *text*.

    Parameters
    ----------
    text  : str  — Banner message.
    char  : str  — Repeat character for the border.
    width : int  — Total banner width.
    """
    border = char * width
    print(f"\n{border}\n  {text}\n{border}")


def prompt_user_input() -> dict:
    """Prompt the user to enter values for the 5 key features.

    Validates each input against its expected dtype and range before
    accepting it.

    Returns
    -------
    dict
        Mapping of ``{column_name: user_value}`` for the 5 key features.
    """
    print_banner("🎓 Student Dropout Risk Predictor", char="─")
    print("  Please answer the following 5 questions about the student.\n")

    user_values = {}

    for idx, feat in enumerate(KEY_FEATURES, start=1):
        col    = feat["col"]
        dtype  = feat["dtype"]
        lo, hi = feat["min"], feat["max"]

        while True:
            raw = input(f"  [{idx}] {feat['prompt']}: ").strip()
            try:
                val = dtype(raw)
                if lo <= val <= hi:
                    user_values[col] = val
                    break
                else:
                    print(f"      ⚠️  Please enter a value between {lo} and {hi}.")
            except ValueError:
                print(f"      ⚠️  Invalid input — expected a {dtype.__name__} value.")

    return user_values


def build_feature_vector(user_values: dict,
                          feature_names: list,
                          scaler) -> np.ndarray:
    """Construct a scaled feature vector from partial user inputs.

    Un-specified features are filled with 0 (a neutral baseline). The
    vector is then scaled using the same :class:`~sklearn.preprocessing.StandardScaler`
    fitted during training.

    Parameters
    ----------
    user_values   : dict       — ``{col: value}`` from :func:`prompt_user_input`.
    feature_names : list[str]  — All feature column names from training.
    scaler        : StandardScaler — Fitted scaler from training.

    Returns
    -------
    np.ndarray
        Scaled 1-D feature vector ready for model inference.
    """
    vec = np.zeros(len(feature_names), dtype=float)

    for col, val in user_values.items():
        if col in feature_names:
            idx      = feature_names.index(col)
            vec[idx] = float(val)
        else:
            # Try case-insensitive match
            lower_names = [n.lower() for n in feature_names]
            try:
                idx      = lower_names.index(col.lower())
                vec[idx] = float(val)
            except ValueError:
                print(f"  ⚠️  Feature '{col}' not found in training set — skipped.")

    vec_scaled = scaler.transform(vec.reshape(1, -1))
    return vec_scaled


def predict_risk(model, feature_vector: np.ndarray) -> tuple:
    """Run inference and return the predicted class and its probability.

    Parameters
    ----------
    model          : sklearn estimator — Best trained classifier.
    feature_vector : np.ndarray       — Scaled 1-D feature vector.

    Returns
    -------
    tuple
        ``(predicted_class_int, confidence_percent)``
    """
    prediction   = model.predict(feature_vector)[0]
    probabilities = model.predict_proba(feature_vector)[0]
    confidence   = round(probabilities[prediction] * 100, 2)
    return int(prediction), confidence


def display_result(prediction: int, confidence: float) -> None:
    """Print a clear, colour-coded risk assessment message.

    Parameters
    ----------
    prediction : int   — Encoded class (0=Dropout, 1=Graduate, 2=Enrolled).
    confidence : float — Model confidence in percent.
    """
    label = TARGET_MAP_INV.get(prediction, "Unknown")

    print("\n" + "=" * 62)
    if prediction == 0:
        print("  ⚠️   HIGH DROPOUT RISK DETECTED")
        print(f"  The model predicts this student is likely to → DROPOUT")
    elif prediction == 1:
        print("  ✅  LOW DROPOUT RISK")
        print(f"  The model predicts this student is likely to → GRADUATE")
    else:
        print("  📘  CURRENTLY ENROLLED")
        print(f"  The model predicts this student will remain → ENROLLED")

    print(f"\n  Predicted outcome : {label}")
    print(f"  Model confidence  : {confidence:.2f} %")
    print("=" * 62 + "\n")


def run_prediction_session() -> None:
    """Load the best model and run an interactive dropout-risk session.

    Reads ``best_model.pkl``, prompts the user for 5 feature values,
    builds a scaled feature vector, runs inference, and prints the result.
    """
    print_banner("📦 Loading Best Model …", char="─")

    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ Model file not found at: {MODEL_PATH}")
        print("     Please run model.py first.")
        return

    payload       = joblib.load(MODEL_PATH)
    model         = payload["model"]
    scaler        = payload["scaler"]
    feature_names = payload["feature_names"]
    model_name    = payload["model_name"]

    print(f"  ✅ Loaded: {model_name}\n")

    user_values    = prompt_user_input()
    feature_vector = build_feature_vector(user_values, feature_names, scaler)
    prediction, confidence = predict_risk(model, feature_vector)
    display_result(prediction, confidence)


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    """Top-level orchestrator: EDA → Training → Prediction.

    Steps
    -----
    1. Run full EDA and save plots.
    2. Train three classifiers, evaluate, and save the best.
    3. Interactive prediction session using the saved model.
    """
    print_banner("🎓 Student Dropout Prediction — Full Pipeline")

    # ── Step 1: EDA ───────────────────────────────────────────────────────
    print_banner("STEP 1 / 3 — Exploratory Data Analysis", char="─")
    run_eda()

    # ── Step 2: Model Training ────────────────────────────────────────────
    print_banner("STEP 2 / 3 — Model Training & Evaluation", char="─")
    train_models()

    # ── Step 3: Interactive Prediction ────────────────────────────────────
    print_banner("STEP 3 / 3 — Interactive Prediction", char="─")
    run_prediction_session()

    print_banner("✅ Pipeline Complete! Check the 'plots/' folder for charts.")


if __name__ == "__main__":
    main()
