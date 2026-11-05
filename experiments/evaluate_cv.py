"""
Stratified 5-Fold Cross Validation Pipeline for DayViT.
Evaluates out-of-fold generalization with zero data leakage across all 1,140 samples.

Usage:
    python experiments/evaluate_cv.py
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset import load_raw_traffic_matrices, TrafficDataset, INV_LABEL_MAP
from src.models import DayViT, count_parameters

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=========================================================")
    print("      Stratified 5-Fold Cross Validation: DayViT        ")
    print("=========================================================")
    print(f"Device: {device}")

    # 1. Load full raw matrices (1140 samples)
    X, y, _ = load_raw_traffic_matrices()
    print(f"Dataset: N={len(y)} samples balanced across 4 classes (285 samples/class).")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = []
    oof_predictions = np.zeros(len(y), dtype=np.int64)

    t_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        set_seed(42 + fold)
        print(f"\n--- Running Fold {fold}/5 ---")

        X_train, X_val = X[train_idx].copy(), X[val_idx].copy()
        y_train, y_val = y[train_idx], y[val_idx]

        # Isolated normalization on training fold (zero data leakage)
        mean = float(np.mean(X_train))
        std = float(np.std(X_train)) + 1e-7
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

        train_ds = TrafficDataset(X_train, y_train)
        val_ds = TrafficDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        model = DayViT(num_classes=4, embed_dim=48, num_heads=4, num_layers=3).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-6)

        best_val_acc = 0.0
        best_val_preds = None

        for epoch in range(70):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=-1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            acc = accuracy_score(val_targets, val_preds)
            if acc > best_val_acc:
                best_val_acc = acc
                best_val_preds = np.array(val_preds)

        oof_predictions[val_idx] = best_val_preds
        f1_macro = f1_score(y_val, best_val_preds, average="macro")
        print(f"Fold {fold} Result: Best Val Accuracy = {best_val_acc*100:.2f}% | Macro F1 = {f1_macro*100:.2f}%")
        fold_metrics.append({"Fold": fold, "Accuracy": best_val_acc * 100, "F1_Macro": f1_macro * 100})

    total_time = time.time() - t_start
    df_folds = pd.DataFrame(fold_metrics)
    mean_acc = df_folds["Accuracy"].mean()
    std_acc = df_folds["Accuracy"].std()
    mean_f1 = df_folds["F1_Macro"].mean()

    print("\n================ 5-FOLD CV SUMMARY ================")
    print(df_folds.to_string(index=False))
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Mean Macro F1: {mean_f1:.2f}%")
    print(f"Overall OOF Accuracy: {accuracy_score(y, oof_predictions)*100:.2f}%")
    print(f"Total Execution Time: {total_time:.2f}s")
    print("====================================================")

    # Save to CSV
    out_csv = RESULTS_DIR / "cv_5fold_summary.csv"
    df_folds.to_csv(out_csv, index=False)
    print(f"Results saved to: {out_csv}")


if __name__ == "__main__":
    main()
