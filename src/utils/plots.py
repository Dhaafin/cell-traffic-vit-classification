"""
Publication-Quality Figure Visualizer for Model Benchmark Results.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = ["UU", "UD", "CU", "CD"]


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    title: str = "Out-of-Fold Confusion Matrix",
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Plots and saves a normalized + raw count confusion matrix.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    cm_arr = np.array(cm)
    cm_norm = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    annot = np.empty_like(cm_arr).astype(str)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            annot[i, j] = f"{cm_arr[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title, fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
    ax.set_ylabel("True Ground-Truth Label", fontsize=10, labelpad=8)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cv_boxplot(
    model_fold_accs: Dict[str, List[float]],
    output_path: Path,
    title: str = "5-Fold Cross Validation Accuracy Distribution",
) -> None:
    """
    Plots boxplot comparison across different models.
    model_fold_accs: Dict mapping Model Name -> list of 5 accuracy percentages.
    """
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    names = list(model_fold_accs.keys())
    data = [model_fold_accs[n] for n in names]

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    colors = palette[:len(names)]

    box = ax.boxplot(data, patch_artist=True, labels=names, widths=0.45)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay scatter points per fold
    for idx, (name, values) in enumerate(model_fold_accs.items()):
        x_jitter = np.random.normal(idx + 1, 0.04, size=len(values))
        ax.scatter(x_jitter, values, color="black", alpha=0.8, s=30, zorder=3)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_leaderboard(
    benchmark_data: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Cellular ViT Benchmark: Accuracy vs. Parameter Footprint",
) -> None:
    """
    Plots accuracy bar comparison alongside parameter count bubble/secondary bar.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

    names = [d["name"] for d in benchmark_data]
    accs = [d["mean_acc"] for d in benchmark_data]
    stds = [d.get("std_acc", 0.0) for d in benchmark_data]
    params = [d["params"] for d in benchmark_data]

    x = np.arange(len(names))
    width = 0.45

    bars = ax1.bar(x, accs, width, yerr=stds, capsize=5, color="#2b5c8f", alpha=0.85, label="5-Fold CV Acc (%)")
    ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold", color="#2b5c8f")
    ax1.set_ylim(bottom=max(0, min(accs) - 15), top=100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10, fontweight="bold")

    # Add accuracy text above bars
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4),
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Secondary axis for parameter count (in millions or log scale)
    ax2 = ax1.twinx()
    param_k = [p / 1e3 for p in params]  # In Thousands (K)
    line = ax2.plot(x, param_k, color="#d95f02", marker="o", linewidth=2.0, markersize=7, label="Parameters (K)")
    ax2.set_ylabel("Parameters (in Thousands / K)", fontsize=11, fontweight="bold", color="#d95f02")
    ax2.set_yscale("log")

    plt.title(title, fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
