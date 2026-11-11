"""
Comprehensive Classification Metrics Calculator for 4-Class Cellular Regimes.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

CLASS_NAMES = ["UU", "UD", "CU", "CD"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes accuracy, macro F1, weighted F1, per-class metrics, and confusion matrix.
    """
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
                "precision": float(report_dict[name]["precision"]),
                "recall": float(report_dict[name]["recall"]),
                "f1-score": float(report_dict[name]["f1-score"]),
                "support": int(report_dict[name]["support"]),
            }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def format_cv_summary_table(fold_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts list of fold results into a standardized pandas DataFrame.
    """
    rows = []
    for r in fold_results:
        rows.append({
            "Fold": r.get("fold", len(rows) + 1),
            "Accuracy (%)": round(r.get("accuracy", 0.0) * 100, 2),
            "Macro F1 (%)": round(r.get("macro_f1", 0.0) * 100, 2),
            "Elapsed (s)": round(r.get("elapsed_sec", 0.0), 2)
        })
    df = pd.DataFrame(rows)
    return df
