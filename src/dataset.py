"""
Dataset & DataLoader Module for Cellular Traffic Spatio-Temporal Matrices.
Processes 7x24 weekly throughput matrices across 4 classes:
- UU: User Uplink
- UD: User Downlink
- CU: Cell Uplink
- CD: Cell Downlink

Implements isolated hourly-mean imputation and strict zero-data-leakage train/test partitioning.
"""
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CLASSES: Dict[str, str] = {
    "UU": "Matrix_UU_User_UL.xlsx",
    "UD": "Matrix_UD_User_DL.xlsx",
    "CU": "Matrix_CU_Cell_UL.xlsx",
    "CD": "Matrix_CD_Cell_DL.xlsx"
}
LABEL_MAP: Dict[str, int] = {"UU": 0, "UD": 1, "CU": 2, "CD": 3}
INV_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}


def load_raw_traffic_matrices(data_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Reads 4 Excel files containing 7x24 weekly throughput matrices,
    applies hourly-mean imputation per class to handle missing values,
    and returns:
      - X: np.ndarray of shape (1140, 7, 24)
      - y: np.ndarray of shape (1140,)
      - id_list: list of sample identifiers
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    X_list = []
    y_list = []
    id_list = []

    for class_name, file_name in CLASSES.items():
        file_path = data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}. Please check your data directory.")

        df = pd.read_excel(file_path, sheet_name="Weekly_Data")
        ids = df["ID"].values
        pixels_df = df.loc[:, "R0C0":"R6C23"].copy()

        # Impute missing entries via hourly-mean across available days
        for hour in range(24):
            hour_cols = [f"R{day}C{hour}" for day in range(7) if f"R{day}C{hour}" in pixels_df.columns]
            hour_vals = pixels_df[hour_cols].values.flatten()
            valid_mean = float(np.nanmean(hour_vals)) if np.any(~np.isnan(hour_vals)) else 0.0
            pixels_df[hour_cols] = pixels_df[hour_cols].fillna(valid_mean)

        mat_data = pixels_df.values.astype(np.float32)  # (N, 168)

        for img_id, row in zip(ids, mat_data):
            img_7x24 = row.reshape(7, 24)
            X_list.append(img_7x24)
            y_list.append(LABEL_MAP[class_name])
            id_list.append(str(img_id))

    X = np.stack(X_list, axis=0)  # (1140, 7, 24)
    y = np.array(y_list, dtype=np.int64)
    return X, y, id_list


class TrafficDataset(Dataset):
    """
    Native PyTorch Dataset for 7x24 spatio-temporal traffic matrices.
    """
    def __init__(self, X, y, transform=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.X[idx]
        label = self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class MultiViewViTDataset(Dataset):
    """
    Multi-View 3-Channel RGB Dataset for ViT-Tiny (224x224x3).
    """
    def __init__(self, X_tensors, y_labels):
        self.X = torch.as_tensor(X_tensors, dtype=torch.float32)
        self.y = torch.as_tensor(y_labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_train_test_loaders(
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: str = "standard",
    data_dir: Optional[Path] = None
) -> Tuple[DataLoader, DataLoader, Dict[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Stratified 80:20 partition with isolated training set normalization (zero data leakage).
    """
    X, y, _ = load_raw_traffic_matrices(data_dir=data_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    norm_stats = {}
    if normalize == "standard":
        mean = float(np.mean(X_train))
        std = float(np.std(X_train)) + 1e-7
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        norm_stats = {"mean": mean, "std": std}
    elif normalize == "minmax":
        min_v = float(np.min(X_train))
        max_v = float(np.max(X_train)) + 1e-7
        X_train = (X_train - min_v) / (max_v - min_v)
        X_test = (X_test - min_v) / (max_v - min_v)
        norm_stats = {"min": min_v, "max": max_v}

    train_dataset = TrafficDataset(X_train, y_train)
    test_dataset = TrafficDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, norm_stats, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    X, y, ids = load_raw_traffic_matrices()
    print(f"[Dataset Loaded] Total Samples: {X.shape[0]} | Matrix Shape: {X.shape[1]}x{X.shape[2]}")
    tr_loader, te_loader, stats, _ = get_train_test_loaders(batch_size=32)
    print(f"[DataLoader Ready] Train batches: {len(tr_loader)} | Test batches: {len(te_loader)}")
    print(f"[Normalization Stats]: {stats}")
