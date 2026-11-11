"""
Evaluation Engine for Cellular Vision Transformers.
Computes loss, accuracy, macro F1, discrete predictions, and softmax probability distributions.
"""
from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates a model over a validation dataloader.
    Returns:
      - avg_loss: float
      - accuracy: float [0.0 - 1.0]
      - macro_f1: float [0.0 - 1.0]
      - preds: np.ndarray of shape (N,)
      - targets: np.ndarray of shape (N,)
      - probs: np.ndarray of shape (N, num_classes) - Softmax probability distribution
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=-1)
            preds = outputs.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    n_samples = len(all_targets)
    avg_loss = total_loss / max(n_samples, 1)
    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)
    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 4))

    acc = float(accuracy_score(targets_np, preds_np))
    f1 = float(f1_score(targets_np, preds_np, average="macro", zero_division=0))

    return avg_loss, acc, f1, preds_np, targets_np, probs_np
