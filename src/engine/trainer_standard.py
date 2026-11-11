"""
Standard End-to-End Training Engine (for DayViT and ViT-Tiny From-Scratch).
"""
from typing import Dict, Any, Tuple, Optional
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .evaluator import evaluate_model


def train_standard(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 70,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.05,
    max_grad_norm: float = 1.0,
    verbose: bool = False,
) -> Tuple[nn.Module, float, float, np.ndarray, Dict[str, Any]]:
    """
    Standard single-stage end-to-end training with AdamW and Cosine Annealing.
    Returns:
      - model: trained model
      - best_val_acc: float
      - best_val_f1: float
      - best_val_preds: np.ndarray
      - history: Dict of loss and acc per epoch
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_preds = np.array([])
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            n_train += inputs.size(0)

        scheduler.step()
        avg_train_loss = train_loss / max(n_train, 1)

        # Evaluate on validation set
        val_loss, val_acc, val_f1, val_preds, _ = evaluate_model(model, val_loader, criterion, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_preds = val_preds.copy()

        if verbose and (epoch % 10 == 0 or epoch == epochs):
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%")

    history["elapsed_sec"] = time.time() - t0
    return model, best_val_acc, best_val_f1, best_val_preds, history
