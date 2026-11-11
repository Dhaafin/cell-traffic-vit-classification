"""
Cross-Validation Splitter Factory for Cellular Traffic Benchmarks.
Supports:
1. 'stratified': Standard sample-level StratifiedKFold (Temporal Generalization).
2. 'group': Base-station sector StratifiedGroupKFold (Spatial / Site-Independent Generalization).
"""
from typing import Any
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold


def get_cv_splitter(cv_mode: str = "stratified", n_splits: int = 5, seed: int = 42) -> Any:
    """
    Returns the appropriate scikit-learn cross-validation splitter.
    """
    mode_clean = str(cv_mode).lower().strip()
    if mode_clean in ["group", "group_kfold", "sgkf", "site"]:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    elif mode_clean in ["stratified", "skf", "sample"]:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        raise ValueError(f"Unknown cv_mode '{cv_mode}'. Choose from 'stratified' or 'group'.")


def format_cv_mode_name(cv_mode: str) -> str:
    """Returns academic display string for cv mode."""
    mode_clean = str(cv_mode).lower().strip()
    if mode_clean in ["group", "group_kfold", "sgkf", "site"]:
        return "Stratified GroupKFold (Site-Independent / 19 eNodeB Sectors)"
    return "Stratified K-Fold (Sample-Independent)"
