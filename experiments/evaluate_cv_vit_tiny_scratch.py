"""
Stratified 5-Fold Cross Validation Pipeline for ViT-Tiny From-Scratch (Random Initialization).
Benchmarks the empirical impact of sample inefficiency on a 5.5M parameter transformer without ImageNet pretraining.
Usage:
    python experiments/evaluate_cv_vit_tiny_scratch.py
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

# Ensure repo root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset import load_raw_traffic_matrices, MultiViewViTDataset, INV_LABEL_MAP
from src.models import build_model, count_parameters
from src.transforms import MultiViewRGBTransform
from src.engine import train_standard
from src.utils import set_seed, compute_metrics, format_cv_summary_table, plot_confusion_matrix

OUTPUT_DIR = ROOT_DIR / "outputs" / "vit_tiny_scratch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_cv_vit_tiny_scratch(
    epochs: int = 70,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    seed_base: int = 42,
    verbose: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  Stratified 5-Fold Cross Validation: ViT-Tiny From-Scratch (Random Init)")
    print("=" * 70)
    print(f"Device: {device}")

    X_raw, y_raw, _ = load_raw_traffic_matrices()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_base)

    fold_metrics = []
    oof_predictions = np.zeros(len(y_raw), dtype=np.int64)

    t_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_raw), 1):
        set_seed(seed_base + fold)
        print(f"\n--- [ViT-Tiny Scratch] Running Fold {fold}/5 ---")

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
            print(f"Model Parameters: {count_parameters(model):,}")

        _, best_acc, best_f1, best_preds, hist = train_standard(
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
        fold_metrics.append({
            "fold": fold,
            "accuracy": best_acc,
            "macro_f1": best_f1,
            "elapsed_sec": hist["elapsed_sec"]
        })
        print(f"Fold {fold} Result: Val Acc = {best_acc*100:.2f}% | Macro F1 = {best_f1*100:.2f}% ({hist['elapsed_sec']:.1f}s)")

        # Clear memory per fold
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - t_start
    overall_metrics = compute_metrics(y_raw, oof_predictions)
    df_folds = format_cv_summary_table(fold_metrics)

    mean_acc = df_folds["Accuracy (%)"].mean()
    std_acc = df_folds["Accuracy (%)"].std()
    mean_f1 = df_folds["Macro F1 (%)"].mean()

    print("\n" + "=" * 50)
    print("      5-FOLD CV SUMMARY: ViT-Tiny From-Scratch")
    print("=" * 50)
    print(df_folds.to_string(index=False))
    print("-" * 50)
    print(f"Mean Accuracy : {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Mean Macro F1 : {mean_f1:.2f}%")
    print(f"Overall OOF Acc: {overall_metrics['accuracy']*100:.2f}%")
    print(f"Total Time    : {total_time:.2f}s")
    print("=" * 50)

    # Save artifacts to outputs/vit_tiny_scratch
    df_folds.to_csv(OUTPUT_DIR / "cv_5fold_summary.csv", index=False)
    with open(OUTPUT_DIR / "metrics_overall.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)

    plot_confusion_matrix(
        overall_metrics["confusion_matrix"],
        OUTPUT_DIR / "confusion_matrix_oof_vit_tiny_scratch.png",
        title="ViT-Tiny Scratch OOF Confusion Matrix (5-Fold CV)"
    )

    return {
        "model": "ViT-Tiny From-Scratch",
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_f1": mean_f1,
        "fold_accs": df_folds["Accuracy (%)"].tolist(),
        "params": 5525188,
        "total_time": total_time
    }


if __name__ == "__main__":
    run_cv_vit_tiny_scratch()
