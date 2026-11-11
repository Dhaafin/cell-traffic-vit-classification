"""
Stratified & Group Cross Validation Pipeline for ViT-Tiny From-Scratch (Random Initialization).
Supports:
  --cv_mode stratified (Default: Sample-independent temporal split)
  --cv_mode group      (Site-independent 19 eNodeB spatial split)
Consolidates all metrics into: outputs/<cv_mode>/vit_tiny_scratch/benchmark_results.json

Usage:
    python experiments/evaluate_cv_vit_tiny_scratch.py --cv_mode stratified
    python experiments/evaluate_cv_vit_tiny_scratch.py --cv_mode group
"""
import sys
import time
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score

# Ensure repo root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset import load_raw_traffic_matrices, extract_cell_groups, MultiViewViTDataset, INV_LABEL_MAP
from src.models import build_model, count_parameters
from src.transforms import MultiViewRGBTransform
from src.engine import train_standard
from src.utils import (
    set_seed,
    compute_comprehensive_metrics,
    compute_cross_validation_distribution,
    build_single_json_report,
    format_cv_summary_table,
    plot_confusion_matrix,
    profile_model_efficiency,
    get_cv_splitter,
    format_cv_mode_name
)


def run_cv_vit_tiny_scratch(
    epochs: int = 70,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    seed_base: int = 42,
    cv_mode: str = "stratified",
    verbose: bool = True
):
    cv_folder = "group" if cv_mode in ["group", "group_kfold"] else "stratified"
    output_dir = ROOT_DIR / "outputs" / cv_folder / "vit_tiny_scratch"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_title = format_cv_mode_name(cv_mode)
    print("=" * 75)
    print("  Cross Validation: ViT-Tiny From-Scratch (Random Init)")
    print(f"  Mode: {mode_title}")
    print("=" * 75)
    print(f"Device: {device}")

    X_raw, y_raw, id_list = load_raw_traffic_matrices()
    groups = extract_cell_groups(id_list) if cv_mode in ["group", "group_kfold"] else None
    unique_groups = len(set(groups)) if groups is not None else None

    splitter = get_cv_splitter(cv_mode=cv_mode, n_splits=5, seed=seed_base)
    split_iter = splitter.split(X_raw, y_raw, groups=groups) if groups is not None else splitter.split(X_raw, y_raw)

    fold_metrics = []
    oof_predictions = np.zeros(len(y_raw), dtype=np.int64)
    oof_probabilities = np.zeros((len(y_raw), 4), dtype=np.float32)

    sample_model = None
    t_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(split_iter, 1):
        set_seed(seed_base + fold)
        val_info = f", Val Groups: {len(set(np.array(groups)[val_idx]))}" if groups is not None else ""
        print(f"\n--- [ViT-Tiny Scratch | {cv_folder.upper()}] Running Fold {fold}/5 (Train: {len(train_idx)}, Val: {len(val_idx)}{val_info}) ---")

        X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
        y_train, y_val = y_raw[train_idx], y_raw[val_idx]

        # Multi-View 3-Channel RGB Transform
        transform = MultiViewRGBTransform(target_size=(224, 224))
        X_train_224 = transform.fit_transform(X_train_raw)
        X_val_224 = transform.transform(X_val_raw)

        train_ds = MultiViewViTDataset(X_train_224, y_train)
        val_ds = MultiViewViTDataset(X_val_224, y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = build_model("vit_tiny_scratch", num_classes=4, pretrained=False, dropout=0.1)
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

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - t_start

    # 1. Hardware efficiency profiling
    print("\n[Hardware Profiling] Profiling ViT-Tiny Scratch footprint...")
    profiling_model = build_model("vit_tiny_scratch", num_classes=4, pretrained=False)
    efficiency_profile = profile_model_efficiency(
        model=profiling_model,
        sample_input=torch.randn(1, 3, 224, 224),
        device=device
    )
    del profiling_model

    # 2. Comprehensive multi-metric calculation
    cv_dist = compute_cross_validation_distribution(fold_metrics)
    overall_metrics = compute_comprehensive_metrics(y_raw, oof_predictions, oof_probabilities)

    # 3. Build ALL-IN-ONE single JSON report
    single_json_report = build_single_json_report(
        model_name="ViT-Tiny From-Scratch",
        cv_dist=cv_dist,
        overall_metrics=overall_metrics,
        efficiency_profile=efficiency_profile,
        total_time_sec=total_time,
        cv_mode=cv_mode,
        group_count=unique_groups
    )

    # 4. Save to SINGLE JSON file
    out_json_path = output_dir / "benchmark_results.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(single_json_report, f, indent=2)

    # 5. Confusion matrix plot
    plot_confusion_matrix(
        overall_metrics["confusion_matrix"],
        output_dir / "confusion_matrix_oof_vit_tiny_scratch.png",
        title=f"ViT-Tiny Scratch OOF Confusion Matrix ({cv_folder.capitalize()} 5-Fold CV)"
    )

    # 6. Display executive summary card
    print("\n" + "=" * 70)
    print(f"  ALL-IN-ONE BENCHMARK: ViT-Tiny From-Scratch [{cv_folder.upper()}]")
    print("=" * 70)
    for k, v in single_json_report["summary_card"].items():
        print(f"  {k:30s}: {v}")
    print("=" * 70)
    print(f"Single JSON Report saved to: {out_json_path}")

    return single_json_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_mode", type=str, default="stratified", choices=["stratified", "group"])
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_cv_vit_tiny_scratch(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed_base=args.seed,
        cv_mode=args.cv_mode
    )
