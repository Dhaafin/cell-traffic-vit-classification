"""
Two-Stage LP-FT Engine (Linear Probing then Fine-Tuning) for Pretrained ViT-Tiny.
Follows the protocol of Kumar et al. (ICLR 2022):
- Phase 1 (Linear Probing): Freeze 100% backbone, train only classification head.
- Phase 2 (Partial Fine-Tuning): Unfreeze top N encoder blocks with differential learning rate.
"""
from typing import Dict, Any, Tuple, Optional
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .evaluator import evaluate_model


def train_lpft(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lp_epochs: int = 5,
    ft_epochs: int = 15,
    lp_lr: float = 1e-3,
    ft_backbone_lr: float = 3e-5,
    ft_head_lr: float = 2e-4,
    unfreeze_last_n: int = 4,
    label_smoothing: float = 0.05,
    max_grad_norm: float = 1.0,
    verbose: bool = False,
) -> Tuple[nn.Module, float, float, np.ndarray, Dict[str, Any]]:
    """
    Executes the two-stage LP-FT pipeline.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_preds = np.array([])
    history = {"phase1": [], "phase2": []}
    t0 = time.time()

    # ==============================================================
    # PHASE 1: LINEAR PROBING
    # ==============================================================
    if hasattr(model, "set_trainable_mode"):
        model.set_trainable_mode("linear_probe")
    else:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True

    opt_lp = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lp_lr,
        weight_decay=0.01
    )

    for epoch in range(1, lp_epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            opt_lp.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            opt_lp.step()
            train_loss += loss.item() * inputs.size(0)
            n_train += inputs.size(0)

        val_loss, val_acc, val_f1, val_preds, _ = evaluate_model(model, val_loader, criterion, device)
        history["phase1"].append({"epoch": epoch, "train_loss": train_loss / max(n_train, 1), "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_preds = val_preds.copy()

        if verbose:
            print(f"[LP Epoch {epoch}/{lp_epochs}] Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%")

    # ==============================================================
    # PHASE 2: PARTIAL FINE-TUNING (DIFFERENTIAL LR)
    # ==============================================================
    if hasattr(model, "set_trainable_mode"):
        model.set_trainable_mode("partial_finetune", unfreeze_last_n=unfreeze_last_n)
    else:
        # Fallback for raw HF model
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        for p in model.vit.layernorm.parameters():
            p.requires_grad = True
        for block in model.vit.encoder.layer[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" in n]

    opt_ft = torch.optim.AdamW([
        {"params": backbone_params, "lr": ft_backbone_lr, "weight_decay": 0.05},
        {"params": head_params, "lr": ft_head_lr, "weight_decay": 0.01},
    ])
    sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=ft_epochs, eta_min=1e-6)

    for epoch in range(1, ft_epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            opt_ft.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            opt_ft.step()
            train_loss += loss.item() * inputs.size(0)
            n_train += inputs.size(0)

        sched_ft.step()
        val_loss, val_acc, val_f1, val_preds, _ = evaluate_model(model, val_loader, criterion, device)
        history["phase2"].append({"epoch": epoch, "train_loss": train_loss / max(n_train, 1), "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_preds = val_preds.copy()

        if verbose:
            print(f"[FT Epoch {epoch}/{ft_epochs}] Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%")

    history["elapsed_sec"] = time.time() - t0
    return model, best_val_acc, best_val_f1, best_val_preds, history
