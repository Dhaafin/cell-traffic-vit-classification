"""
Stratified 5-Fold Cross Validation Pipeline for DayViT (Native 7x24 Compact Transformer).
Consolidates all metrics into a single comprehensive JSON report: outputs/dayvit/benchmark_results.json
Usage:
    python experiments/evaluate_cv_dayvit.py
"""
import sys
import time
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score

# Ensure repo root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset import load_raw_traffic_matrices, TrafficDataset, INV_LABEL_MAP
from src.models import build_model, count_parameters
from src.transforms import ZScore7x24Transform
from src.engine import train_standard
from src.utils import (
    set_seed,
    compute_comprehensive_metrics,
    compute_cross_validation_distribution,
    build_single_json_report,
    format_cv_summary_table,
    plot_confusion_matrix,
    profile_model_efficiency
)

OUTPUT_DIR = ROOT_DIR / "outputs" / "dayvit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_cv_dayvit(
    epochs: int = 70,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    seed_base: int = 42,
    verbose: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("      Stratified 5-Fold Cross Validation: DayViT (Native 7x24)   ")
    print("=" * 70)
    print(f"Device: {device}")

    X_raw, y_raw, _ = load_raw_traffic_matrices()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_base)

    fold_metrics = []
    oof_predictions = np.zeros(len(y_raw), dtype=np.int64)
    oof_probabilities = np.zeros((len(y_raw), 4), dtype=np.float32)

    sample_model = None
    t_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_raw), 1):
        set_seed(seed_base + fold)
        print(f"\n--- [DayViT] Running Fold {fold}/5 ---")

        X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
        y_train, y_val = y_raw[train_idx], y_raw[val_idx]

        # Isolated normalizer
        normalizer = ZScore7x24Transform()
        X_train_tensor = normalizer.fit_transform(X_train_raw)
        X_val_tensor = normalizer.transform(X_val_raw)

        train_ds = TrafficDataset(X_train_tensor, y_train)
        val_ds = TrafficDataset(X_val_tensor, y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = build_model("dayvit", num_classes=4, embed_dim=48, num_heads=4, num_layers=3)
        if fold == 1:
            sample_model = model
            print(f"Model Parameters: {count_parameters(model):,}")

        _, best_acc, best_f1, best_preds, best_probs, hist = train_standard(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            verbose=False
        )

        oof_predictions[val_idx] = best_preds
        oof_probabilities[val_idx] = best_probs

        prec = float(precision_score(y_val, best_preds, average="macro", zero_division=0))
        rec = float(recall_score(y_val, best_preds, average="macro", zero_division=0))

        fold_metrics.append({
            "fold": fold,
            "accuracy": round(best_acc, 4),
            "macro_f1": round(best_f1, 4),
            "macro_precision": round(prec, 4),
            "macro_recall": round(rec, 4),
            "elapsed_sec": round(hist["elapsed_sec"], 2)
        })
        print(f"Fold {fold} Result: Val Acc = {best_acc*100:.2f}% | Macro F1 = {best_f1*100:.2f}% | Prec = {prec*100:.2f}% | Rec = {rec*100:.2f}% ({hist['elapsed_sec']:.1f}s)")

    total_time = time.time() - t_start

    # 1. Hardware efficiency profiling
    print("\n[Hardware Profiling] Profiling DayViT inference and compute footprint...")
    efficiency_profile = profile_model_efficiency(
        model=sample_model if sample_model is not None else build_model("dayvit"),
        sample_input=torch.randn(1, 7, 24),
        device=device
    )

    # 2. Comprehensive multi-metric calculation
    cv_dist = compute_cross_validation_distribution(fold_metrics)
    overall_metrics = compute_comprehensive_metrics(y_raw, oof_predictions, oof_probabilities)

    # 3. Build ALL-IN-ONE single JSON report
    single_json_report = build_single_json_report(
        model_name="DayViT (Ours)",
        cv_dist=cv_dist,
        overall_metrics=overall_metrics,
        efficiency_profile=efficiency_profile,
        total_time_sec=total_time
    )

    # 4. Save to SINGLE JSON file
    out_json_path = OUTPUT_DIR / "benchmark_results.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(single_json_report, f, indent=2)

    # 5. Confusion matrix plot
    plot_confusion_matrix(
        overall_metrics["confusion_matrix"],
        OUTPUT_DIR / "confusion_matrix_oof_dayvit.png",
        title="DayViT OOF Confusion Matrix (5-Fold CV)"
    )

    # 6. Display executive summary card
    print("\n" + "=" * 65)
    print("        ALL-IN-ONE BENCHMARK RESULTS: DayViT (Ours)")
    print("=" * 65)
    for k, v in single_json_report["summary_card"].items():
        print(f"  {k:30s}: {v}")
    print("=" * 65)
    print(f"Single JSON Report saved to: {out_json_path}")

    return single_json_report


if __name__ == "__main__":
    run_cv_dayvit()
