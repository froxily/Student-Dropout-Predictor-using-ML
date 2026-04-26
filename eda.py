"""
eda.py — Exploratory Data Analysis for Student Dropout Prediction
=================================================================
Loads the dataset, prints key statistics, and saves visualisation
plots (class distribution + correlation heatmap) as PNG files.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all OSes)
import matplotlib.pyplot as plt
import seaborn as sns


# ── paths ──────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
OUT_DIR      = os.path.join(os.path.dirname(__file__), "plots")


def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the student dataset from *path* and return a DataFrame.

    Parameters
    ----------
    path : str
        Absolute or relative path to ``dataset.csv``.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns intact.
    """
    df = pd.read_csv(path, delimiter=";")
    return df


def print_overview(df: pd.DataFrame) -> None:
    """Print shape, dtypes, and null-value counts for *df*.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset returned by :func:`load_data`.
    """
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"\n📐 Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n📋 Data Types:")
    print(df.dtypes.to_string())

    null_counts = df.isnull().sum()
    print(f"\n🔍 Null Values  (total: {null_counts.sum()}):")
    if null_counts.sum() == 0:
        print("   No missing values found — dataset is complete ✅")
    else:
        print(null_counts[null_counts > 0].to_string())

    print("\n📊 Target distribution:")
    print(df["Target"].value_counts().to_string())
    print("=" * 60 + "\n")


def plot_class_distribution(df: pd.DataFrame, out_dir: str = OUT_DIR) -> None:
    """Plot and save a bar chart of the Target class distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing a ``Target`` column.
    out_dir : str
        Directory where the PNG file will be saved.
    """
    os.makedirs(out_dir, exist_ok=True)

    counts = df["Target"].value_counts().sort_index()
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]   # Dropout, Enrolled, Graduate

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.55)

    # Annotate counts on top of each bar
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 40,
                f"{count:,}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#2C3E50")

    ax.set_title("Student Outcome Distribution", fontsize=16, fontweight="bold",
                 pad=15, color="#2C3E50")
    ax.set_xlabel("Target Class", fontsize=13, color="#555")
    ax.set_ylabel("Number of Students", fontsize=13, color="#555")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    save_path = os.path.join(out_dir, "class_distribution.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ Saved: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str = OUT_DIR) -> None:
    """Compute and save a correlation heatmap for numeric columns.

    Only the top 20 numeric features (by variance) are shown to keep
    the chart readable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset (numeric + Target columns present).
    out_dir : str
        Directory where the PNG file will be saved.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Encode Target temporarily for correlation
    target_map  = {"Dropout": 0, "Graduate": 1, "Enrolled": 2}
    df_enc      = df.copy()
    df_enc["Target"] = df_enc["Target"].map(target_map).fillna(df_enc["Target"])

    numeric_df = df_enc.select_dtypes(include="number")

    # Keep only top 20 columns by variance to avoid an unreadable 37×37 grid
    top_cols   = numeric_df.var().nlargest(20).index.tolist()
    corr       = numeric_df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = corr.abs() < 0.05          # hide near-zero correlations

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        linecolor="#e8e8e8",
        annot_kws={"size": 7},
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap (Top 20 by Variance)",
                 fontsize=14, fontweight="bold", pad=14, color="#2C3E50")
    plt.xticks(fontsize=8, rotation=45, ha="right")
    plt.yticks(fontsize=8, rotation=0)
    fig.tight_layout()

    save_path = os.path.join(out_dir, "correlation_heatmap.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ Saved: {save_path}")


def run_eda() -> pd.DataFrame:
    """Orchestrate the full EDA pipeline and return the loaded DataFrame.

    Returns
    -------
    pd.DataFrame
        The raw dataset (unchanged).
    """
    print("\n🔎 Running Exploratory Data Analysis …\n")
    df = load_data()
    print_overview(df)
    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    print("\n📁 All EDA plots saved to the 'plots/' directory.\n")
    return df


if __name__ == "__main__":
    run_eda()
