"""
model_evaluation.py
Unified ML model evaluation and comparison across all 5 models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def compare_models(model_results: dict, output_dir: str = "outputs/figures") -> pd.DataFrame:
    """
    Builds a comparison table across all 5 ML models.

    Args:
        model_results: dict mapping model_name → artifacts dict from train_*()
        output_dir:    where to save comparison chart

    Returns:
        DataFrame comparison table
    """
    rows = []

    for name, artifacts in model_results.items():
        res = artifacts.get("results", {})
        test = res.get("test", {})

        row = {"Model": name}

        if "r2" in test:
            row["Test R²"]   = test.get("r2", "-")
            row["Test RMSE"] = test.get("rmse", "-")
            row["CV R² Mean"] = res.get("cv_r2_mean", "-")
        if "accuracy" in test:
            row["Test Accuracy"] = test.get("accuracy", "-")
            row["Test F1"]       = test.get("f1", "-")
            row["CV Acc Mean"]   = res.get("cv_accuracy_mean", "-")
        if "auc_roc" in test:
            row["Test AUC-ROC"] = test.get("auc_roc", "-")

        row["N Train"] = artifacts.get("n_train", "-")
        row["N Test"]  = artifacts.get("n_test", "-")
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    return df


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    model_name: str,
    top_n: int = 10,
    output_dir: str = "outputs/figures",
) -> str:
    """
    Saves a horizontal bar chart of feature importance.

    Args:
        feature_importance_df: DataFrame with 'feature' and 'importance' columns
        model_name:            label for chart title
        top_n:                 number of top features to show
        output_dir:            save directory

    Returns:
        path to saved figure
    """
    os.makedirs(output_dir, exist_ok=True)
    top = feature_importance_df.head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2196F3" if i < 3 else "#64B5F6" for i in range(len(top))]
    ax.barh(top["feature"], top["importance"], color=colors[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance — {model_name} (Top {top_n})")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_shap_summary(shap_values, feature_names: list, output_dir: str = "outputs/figures") -> str:
    """
    Saves a SHAP summary plot if SHAP is installed.

    Returns:
        path to saved figure, or empty string if SHAP unavailable
    """
    try:
        import shap
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        return path
    except ImportError:
        return ""


def print_model_summary(artifacts: dict, model_name: str) -> None:
    """Prints a concise model performance summary to stdout."""
    res = artifacts.get("results", {})
    print(f"\n{'=' * 50}")
    print(f"  {model_name}")
    print(f"{'=' * 50}")
    for split in ["train", "val", "test"]:
        split_res = res.get(split, {})
        metrics_str = "  ".join(f"{k}: {v}" for k, v in split_res.items()
                                if k not in ("confusion_matrix",))
        print(f"  [{split:5s}]  {metrics_str}")
    if "cv_r2_mean" in res:
        print(f"  [5-fold CV]  R² = {res['cv_r2_mean']} ± {res['cv_r2_std']}")
    if "cv_accuracy_mean" in res:
        print(f"  [5-fold CV]  Acc = {res['cv_accuracy_mean']} ± {res['cv_accuracy_std']}")
