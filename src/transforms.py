"""
Spatio-Temporal Transformations for Cellular Throughput Matrices.
Provides isolated, zero-data-leakage transformations:
1. MultiViewRGBTransform: Projects 7x24 raw matrices into (3, 224, 224) RGB tensors
   with three complementary domain views (Global Z-Score, Log1p Z-Score, Local MinMax [-1, 1])
   for standard Vision Transformers (ViT-Tiny).
2. ZScore7x24Transform: Isolated scalar normalization for native 7x24 spatio-temporal inputs (DayViT).
"""
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


class MultiViewRGBTransform:
    """
    Multi-View 3-Channel RGB Spatio-Temporal Transformation with Bicubic Upsampling to 224x224.
    Fitted strictly on training fold statistics to ensure zero data leakage.
    """
    def __init__(self, target_size: Tuple[int, int] = (224, 224), eps: float = 1e-7):
        self.target_size = target_size
        self.eps = eps
        self.g_mean: float = 0.0
        self.g_std: float = 1.0
        self.l_mean: float = 0.0
        self.l_std: float = 1.0
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray) -> "MultiViewRGBTransform":
        """
        Fit global and log-normal statistics strictly on training samples.
        Args:
            X_train: np.ndarray of shape (N, 7, 24)
        """
        self.g_mean = float(np.mean(X_train))
        self.g_std = float(np.std(X_train)) + self.eps

        X_train_log = np.log1p(np.maximum(X_train, 0.0))
        self.l_mean = float(np.mean(X_train_log))
        self.l_std = float(np.std(X_train_log)) + self.eps

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> torch.Tensor:
        """
        Applies multi-view projection and bicubic interpolation.
        Args:
            X: np.ndarray of shape (N, 7, 24)
        Returns:
            torch.Tensor of shape (N, 3, 224, 224)
        """
        if not self.is_fitted:
            raise RuntimeError("MultiViewRGBTransform must be fitted with training data before calling transform().")

        N = X.shape[0]

        # Channel 0 (R): Global Z-Score (Macro volume differences)
        ch_global = (X - self.g_mean) / self.g_std

        # Channel 1 (G): Log1p Z-Score (Compressed bursty tail)
        X_log = np.log1p(np.maximum(X, 0.0))
        ch_log = (X_log - self.l_mean) / self.l_std

        # Channel 2 (B): Per-Sample MinMax Standardized [-1, 1] (Pure diurnal signature)
        mins = X.reshape(N, -1).min(axis=1).reshape(N, 1, 1)
        maxs = X.reshape(N, -1).max(axis=1).reshape(N, 1, 1)
        denoms = np.where((maxs - mins) > self.eps, maxs - mins, 1.0)
        ch_local = ((X - mins) / denoms) * 2.0 - 1.0

        # Stack into (N, 3, 7, 24)
        X_3ch = np.stack([ch_global, ch_log, ch_local], axis=1).astype(np.float32)
        X_tensor = torch.from_numpy(X_3ch)

        # Upsampling Bicubic to (N, 3, 224, 224)
        X_224 = F.interpolate(
            X_tensor,
            size=self.target_size,
            mode="bicubic",
            align_corners=False
        )
        return X_224

    def fit_transform(self, X_train: np.ndarray) -> torch.Tensor:
        """Fits on X_train and returns transformed tensor."""
        return self.fit(X_train).transform(X_train)


class ZScore7x24Transform:
    """
    Isolated Scalar Z-Score Normalization for native 7x24 spatio-temporal inputs.
    """
    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.mean: float = 0.0
        self.std: float = 1.0
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray) -> "ZScore7x24Transform":
        self.mean = float(np.mean(X_train))
        self.std = float(np.std(X_train)) + self.eps
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("ZScore7x24Transform must be fitted with training data before calling transform().")
        normed = (X - self.mean) / self.std
        return torch.tensor(normed, dtype=torch.float32)

    def fit_transform(self, X_train: np.ndarray) -> torch.Tensor:
        return self.fit(X_train).transform(X_train)
