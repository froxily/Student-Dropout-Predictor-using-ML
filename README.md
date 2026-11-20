# 🎓 Student Dropout Prediction

> A complete Machine Learning pipeline that predicts whether a student will **drop out**, **graduate**, or remain **enrolled** — using the UCI *Predict Students' Dropout and Academic Success* dataset.

---

## 📋 What This Project Does

This project trains three classifiers on real higher-education data and selects the best-performing one to power an **interactive risk predictor**. Given just 5 key facts about a student, the tool instantly outputs a risk label:

- ⚠️ **High Dropout Risk** — Intervention may be needed
- ✅ **Low Dropout Risk** — Student is on track to graduate
- 📘 **Currently Enrolled** — Student is progressing normally

---

## 📁 Project Structure

```
student_dropout_prediction/
├── dataset.csv          ← UCI dataset (semicolon-delimited)
├── main.py              ← Full pipeline entry point
├── eda.py               ← Exploratory Data Analysis
├── model.py             ← Model training, evaluation & saving
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
├── best_model.pkl       ← Saved best model (generated on first run)
└── plots/               ← Auto-generated visualisation PNGs
    ├── class_distribution.png
    ├── correlation_heatmap.png
    ├── rf_feature_importance.png
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_random_forest.png
    └── confusion_matrix_decision_tree.png
```

---

## 📊 Dataset

| Property       | Details                                                         |
|----------------|-----------------------------------------------------------------|
| **Name**       | Predict Students' Dropout and Academic Success                  |
| **Source**     | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) |
| **Instances**  | ~4,424 students                                                 |
| **Features**   | 36 attributes (demographics, academic, socio-economic)          |
| **Target**     | `Dropout` / `Graduate` / `Enrolled`                             |
| **Format**     | CSV with semicolon (`;`) delimiter                              |

Download the CSV from the UCI link above and save it as **`dataset.csv`** inside the project folder.

---

## 🚀 How to Run

### 1. Prerequisites

- Python 3.9 or later
- pip

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline

```bash
python main.py
```

This will:
1. Perform EDA and save plots to `plots/`
2. Train three models and print their metrics
3. Save the best model as `best_model.pkl`
4. Launch the interactive student risk predictor

### 4. Run Individual Modules

```bash
# EDA only
python eda.py

# Training only
python model.py
```

---

## 🧠 Models Trained

| Model               | Notes                                         |
|---------------------|-----------------------------------------------|
| Logistic Regression | Baseline linear model with balanced classes   |
| Random Forest       | Ensemble of 200 trees; also gives importances |
| Decision Tree       | Interpretable single-tree model               |

The model with the highest test-set accuracy is automatically saved as `best_model.pkl`.

---

## 📂 File Descriptions

### `eda.py`
- Loads `dataset.csv` with pandas
- Prints dataset shape, dtypes, and null-value counts
- Plots and saves:
  - **class_distribution.png** — Bar chart of Dropout / Graduate / Enrolled counts
  - **correlation_heatmap.png** — Heatmap of top-20 feature correlations

### `model.py`
- Encodes the `Target` column (`Dropout=0`, `Graduate=1`, `Enrolled=2`)
- Scales features with `StandardScaler`
- Performs an 80/20 stratified train/test split
- Trains Logistic Regression, Random Forest, and Decision Tree
- For each model prints: accuracy, classification report, confusion matrix
- Saves confusion-matrix PNGs for each model
- Plots and saves `rf_feature_importance.png` (top 15 features)
- Serialises the best model, scaler, and feature names to `best_model.pkl`

### `main.py`
- Calls `eda.py` → `model.py` in sequence
- Loads `best_model.pkl`
- Prompts the user for **5 key features**:
  1. Curricular units approved — 2nd semester
  2. Curricular units approved — 1st semester
  3. Average grade — 2nd semester
  4. Tuition fees up to date (1/0)
  5. Age at enrollment
- Predicts the student's outcome and prints a clear risk label

### `requirements.txt`
Lists all Python dependencies with pinned versions for reproducibility.

---

## 🔑 Key Features Used for Prediction

| # | Feature                                         | Why It Matters                            |
|---|--------------------------------------------------|-------------------------------------------|
| 1 | Curricular units approved (2nd sem)              | Academic progression indicator            |
| 2 | Curricular units approved (1st sem)              | Early performance signal                  |
| 3 | Average grade (2nd sem)                          | Direct academic performance measure       |
| 4 | Tuition fees up to date                          | Financial stability strongly linked to retention |
| 5 | Age at enrollment                                | Older students face different retention challenges |

---

## 📈 Sample Output

```
══════════════════════════════════════════════════════════════
🏆 Best Model : Random Forest  (Accuracy: 87.45 %)
══════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════
  ⚠️   HIGH DROPOUT RISK DETECTED
  The model predicts this student is likely to → DROPOUT

  Predicted outcome : Dropout
  Model confidence  : 78.50 %
══════════════════════════════════════════════════════════════
```

---

## Copyright & Ownership

Copyright (c) 2026 Akshat Mishra. All Rights Reserved.

This applies to original source code, documentation, architecture, and assets authored by the copyright holder. Third-party software, data, fonts, and contributions remain subject to their own licenses and notices. No additional permission is granted for those original materials. Viewing and forking through GitHub remain subject to GitHub’s Terms of Service. For licensing inquiries, contact the copyright holder through GitHub.

## Third-party data

The UCI dataset remains subject to its source terms; see the dataset source and existing license history before reuse.
