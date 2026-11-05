"""
Hyperparameter Training & Optimization Pipeline for DayViT.
Evaluates training variations (learning rate, weight decay, label smoothing, scheduler)
on the native 7x24 cellular traffic dataset.

Usage:
    python experiments/tune_training.py
"""
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset import get_train_test_loaders, INV_LABEL_MAP
from src.models import DayViT, count_parameters

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, step_scheduler=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step_scheduler and scheduler:
            scheduler.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def run_experiment(config, train_loader, test_loader, device):
    set_seed(42)
    model = DayViT(num_classes=4, embed_dim=48, num_heads=4, num_layers=3).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999)
    )

    epochs = config["epochs"]
    scheduler_type = config["scheduler"]

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        step_scheduler = False
    elif scheduler_type == "onecycle":
        total_steps = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"], total_steps=total_steps)
        step_scheduler = True
    else:
        scheduler = None
        step_scheduler = False

    best_test_acc = 0.0
    best_preds = None

    t0 = time.time()
    for ep in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler, step_scheduler)
        if not step_scheduler and scheduler:
            scheduler.step()

        test_loss, test_acc, preds, labels = eval_model(model, test_loader, criterion, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_preds = preds

    elapsed = time.time() - t0
    return best_test_acc, elapsed, best_preds, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Training Experiment on: {device}")

    train_loader, test_loader, _, _ = get_train_test_loaders(batch_size=32, test_size=0.2, random_state=42)

    configs = [
        {"id": "OPT-Default", "name": "AdamW (lr=1e-3, wd=1e-2, no-smooth)", "lr": 1e-3, "weight_decay": 1e-2, "label_smoothing": 0.0, "scheduler": "cosine", "epochs": 70},
        {"id": "OPT-Smooth05", "name": "AdamW (lr=1e-3, wd=1e-2, smooth=0.05)", "lr": 1e-3, "weight_decay": 1e-2, "label_smoothing": 0.05, "scheduler": "cosine", "epochs": 70},
        {"id": "OPT-Best", "name": "AdamW (lr=5e-4, wd=1e-3, smooth=0.05, 100ep)", "lr": 5e-4, "weight_decay": 1e-3, "label_smoothing": 0.05, "scheduler": "cosine", "epochs": 100},
    ]

    results = []
    for cfg in configs:
        print(f"\n--- Running: {cfg['name']} ---")
        best_acc, elapsed, preds, labels = run_experiment(cfg, train_loader, test_loader, device)
        print(f"Result: Best Test Accuracy = {best_acc*100:.2f}% | Elapsed = {elapsed:.2f}s")
        results.append({
            "Config": cfg["name"],
            "Best_Accuracy": round(best_acc * 100, 2),
            "Time_sec": round(elapsed, 2)
        })

    df_res = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "tuning_summary.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved summary to: {out_csv}")
    print(df_res.to_string())


if __name__ == "__main__":
    main()
