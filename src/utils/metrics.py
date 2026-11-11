"""
Comprehensive Classification & Probabilistic Metrics Calculator.
Supports:
1. Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall.
2. Macro ROC-AUC (One-vs-Rest) & Average Precision / PR-AUC.
3. Log-Loss (Cross-Entropy Loss), Cohen's Kappa, Matthews Correlation Coefficient.
4. Per-Fold Mean ± Std across ALL evaluation metrics.
5. Consolidated Single-JSON report builder.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    log_loss,
    cohen_kappa_score,
    matthews_corrcoef
)

CLASS_NAMES = ["UU", "UD", "CU", "CD"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Basic discrete classification metrics."""
    if target_names is None:
        target_names = CLASS_NAMES

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)

    per_class = {}
    for name in target_names:
        if name in report_dict:
            per_class[name] = {
                "precision": round(float(report_dict[name]["precision"]), 4),
                "recall": round(float(report_dict[name]["recall"]), 4),
                "f1_score": round(float(report_dict[name]["f1-score"]), 4),
                "support": int(report_dict[name]["support"]),
            }

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes full discrete + probabilistic + agreement metrics.
    """
    if target_names is None:
        target_names = CLASS_NAMES

    num_classes = len(target_names)
    base = compute_metrics(y_true, y_pred, target_names=target_names)

    # Multi-class agreement metrics
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    base["cohen_kappa"] = round(kappa, 4)
    base["matthews_corrcoef"] = round(mcc, 4)

    # Probabilistic metrics if probabilities are provided
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == num_classes:
        # Binarize true labels for One-vs-Rest calculation
        y_true_onehot = np.eye(num_classes)[y_true]

        try:
            macro_roc_auc = float(roc_auc_score(y_true_onehot, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            macro_roc_auc = 0.0

        try:
            macro_pr_auc = float(average_precision_score(y_true_onehot, y_prob, average="macro"))
        except Exception:
            macro_pr_auc = 0.0

        try:
            ce_loss = float(log_loss(y_true, y_prob))
        except Exception:
            ce_loss = 0.0

        base["macro_roc_auc"] = round(macro_roc_auc, 4)
        base["macro_pr_auc"] = round(macro_pr_auc, 4)
        base["log_loss"] = round(ce_loss, 4)

        # Per-class ROC-AUC and PR-AUC
        for idx, name in enumerate(target_names):
            if name in base["per_class"]:
                try:
                    c_roc = float(roc_auc_score(y_true_onehot[:, idx], y_prob[:, idx]))
                except Exception:
                    c_roc = 0.0
                try:
                    c_pr = float(average_precision_score(y_true_onehot[:, idx], y_prob[:, idx]))
                except Exception:
                    c_pr = 0.0
                base["per_class"][name]["roc_auc"] = round(c_roc, 4)
                base["per_class"][name]["pr_auc"] = round(c_pr, 4)

    return base


def compute_cross_validation_distribution(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes Mean ± Std for ALL metrics across the 5 cross-validation folds.
    """
    accs = [f["accuracy"] * 100 for f in fold_metrics]
    f1s = [f["macro_f1"] * 100 for f in fold_metrics]
    precs = [f.get("macro_precision", f["macro_f1"]) * 100 for f in fold_metrics]
    recs = [f.get("macro_recall", f["macro_f1"]) * 100 for f in fold_metrics]

    return {
        "n_splits": len(fold_metrics),
        "accuracy_mean": round(float(np.mean(accs)), 2),
        "accuracy_std": round(float(np.std(accs, ddof=1)), 2),
        "macro_f1_mean": round(float(np.mean(f1s)), 2),
        "macro_f1_std": round(float(np.std(f1s, ddof=1)), 2),
        "macro_precision_mean": round(float(np.mean(precs)), 2),
        "macro_precision_std": round(float(np.std(precs, ddof=1)), 2),
        "macro_recall_mean": round(float(np.mean(recs)), 2),
        "macro_recall_std": round(float(np.std(recs, ddof=1)), 2),
        "per_fold": fold_metrics
    }


def build_single_json_report(
    model_name: str,
    cv_dist: Dict[str, Any],
    overall_metrics: Dict[str, Any],
    efficiency_profile: Dict[str, Any],
    total_time_sec: float,
    cv_mode: str = "stratified",
    group_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Consolidates ALL experimental findings into a single, comprehensive dictionary.
    """
    acc_m, acc_s = cv_dist["accuracy_mean"], cv_dist["accuracy_std"]
    f1_m, f1_s = cv_dist["macro_f1_mean"], cv_dist["macro_f1_std"]
    prec_m, prec_s = cv_dist["macro_precision_mean"], cv_dist["macro_precision_std"]
    rec_m, rec_s = cv_dist["macro_recall_mean"], cv_dist["macro_recall_std"]

    cv_mode_desc = (
        f"Stratified GroupKFold ({group_count} eNodeB Sectors / Site-Independent)"
        if cv_mode in ["group", "group_kfold"]
        else "Stratified K-Fold (Sample-Independent)"
    )

    summary_card = {
        "model_architecture": model_name,
        "cross_validation_mode": cv_mode_desc,
        "5fold_cv_accuracy": f"{acc_m:.2f}% ± {acc_s:.2f}%",
        "5fold_cv_macro_f1": f"{f1_m:.2f}% ± {f1_s:.2f}%",
        "5fold_cv_macro_precision": f"{prec_m:.2f}% ± {prec_s:.2f}%",
        "5fold_cv_macro_recall": f"{rec_m:.2f}% ± {rec_s:.2f}%",
        "oof_overall_accuracy": f"{overall_metrics['accuracy']*100:.2f}%",
        "macro_roc_auc": overall_metrics.get("macro_roc_auc", "N/A"),
        "macro_pr_auc": overall_metrics.get("macro_pr_auc", "N/A"),
        "cohen_kappa": overall_metrics.get("cohen_kappa", "N/A"),
        "matthews_corrcoef": overall_metrics.get("matthews_corrcoef", "N/A"),
        "total_parameters": f"{efficiency_profile.get('total_parameters', 0):,}",
        "flops_computation": efficiency_profile.get("flops_formatted", "N/A"),
        "inference_latency_per_sample": f"{efficiency_profile.get('inference_latency_ms', 0.0)} ms",
        "throughput_fps": f"{efficiency_profile.get('throughput_fps', 0.0)} samples/sec",
        "peak_vram_mb": f"{efficiency_profile.get('peak_vram_mb', 0.0)} MB",
        "total_experiment_time": f"{total_time_sec:.1f} s",
    }

    cv_dist["cv_mode"] = cv_mode_desc
    if group_count is not None:
        cv_dist["unique_groups"] = group_count

    return {
        "model_name": model_name,
        "summary_card": summary_card,
        "efficiency_profiling": efficiency_profile,
        "cross_validation_distribution": cv_dist,
        "overall_out_of_fold_evaluation": overall_metrics
    }


def format_cv_summary_table(fold_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Standardized pandas table representation."""
    rows = []
    for r in fold_results:
        rows.append({
            "Fold": r.get("fold", len(rows) + 1),
            "Accuracy (%)": round(r.get("accuracy", 0.0) * 100, 2),
            "Macro F1 (%)": round(r.get("macro_f1", 0.0) * 100, 2),
            "Precision (%)": round(r.get("macro_precision", r.get("macro_f1", 0.0)) * 100, 2),
            "Recall (%)": round(r.get("macro_recall", r.get("macro_f1", 0.0)) * 100, 2),
            "Elapsed (s)": round(r.get("elapsed_sec", 0.0), 2)
        })
    return pd.DataFrame(rows)
