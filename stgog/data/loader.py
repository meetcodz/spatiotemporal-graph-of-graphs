import os
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler


class MTSDataset(Dataset):
    """PyTorch Dataset wrapper for Multivariate Time Series tensors."""
    def __init__(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X.float()

        if isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y).long()
        else:
            self.y = y.long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def generate_synthetic_mts(
    n_train: int = 100,
    n_test: int = 40,
    seq_len: int = 50,
    n_channels: int = 6,
    num_classes: int = 3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Generate synthetic multivariate time series for testing and offline runs.
    Each class exhibits a distinct sinusoidal frequency and phase across channels.
    """
    rng = np.random.default_rng(seed)

    def make_split(n):
        X = np.zeros((n, seq_len, n_channels), dtype=np.float32)
        y = rng.integers(0, num_classes, size=n)
        t = np.linspace(0, 4 * np.pi, seq_len)
        for i in range(n):
            c_label = y[i]
            freq = (c_label + 1) * 1.5
            for ch in range(n_channels):
                phase = (ch * np.pi) / n_channels
                noise = rng.normal(0, 0.15, size=seq_len)
                X[i, :, ch] = np.sin(freq * t + phase) + noise
        return X, y

    X_train, y_train = make_split(n_train)
    X_test, y_test = make_split(n_test)
    return X_train, y_train, X_test, y_test, num_classes


def dataframe_to_numpy(df) -> np.ndarray:
    """Convert an sktime nested DataFrame of series into a 3D numpy array (N, T, C)."""
    N, C = df.shape
    T = len(df.iloc[0, 0])
    X = np.zeros((N, T, C), dtype=np.float32)
    for i in range(N):
        for j in range(C):
            X[i, :, j] = df.iloc[i, j].to_numpy()
    return X


def load_uea_dataset(
    dataset_name: str,
    cache_dir: Optional[Union[str, Path]] = "data/cache",
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load and preprocess a dataset from the UEA/UCR benchmark.
    Checks local .npz cache first. If not found, calls sktime to download and caches it.
    Performs standard scaling across channels and integer label encoding.

    Returns:
        X_train: (N_train, T, C)
        y_train: (N_train,)
        X_test: (N_test, T, C)
        y_test: (N_test,)
        num_classes: int
    """
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{dataset_name}.npz"

    if use_cache and cache_path is not None and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["X_train"],
            data["y_train"],
            data["X_test"],
            data["y_test"],
            int(data["num_classes"]),
        )

    # Load via sktime
    try:
        from sktime.datasets import load_UCR_UEA_dataset
    except ImportError:
        raise ImportError(
            f"sktime is not installed. To download '{dataset_name}', install sktime: pip install sktime"
        )

    Xtr_raw, ytr_raw = load_UCR_UEA_dataset(dataset_name, split="train", return_X_y=True)
    Xte_raw, yte_raw = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

    # Convert to 3D numpy arrays (N, T, C)
    if hasattr(Xtr_raw, "iloc"):
        X_train = dataframe_to_numpy(Xtr_raw)
        X_test = dataframe_to_numpy(Xte_raw)
    else:
        # numpy3D returns (N, C, T) in sktime; transpose to (N, T, C)
        X_train = np.transpose(Xtr_raw, (0, 2, 1)).astype(np.float32)
        X_test = np.transpose(Xte_raw, (0, 2, 1)).astype(np.float32)

    # Standard scaling across channels
    N_tr, T, C = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, C)).reshape(N_tr, T, C)
    X_test = scaler.transform(X_test.reshape(-1, C)).reshape(X_test.shape)

    # Label encoding
    le = LabelEncoder()
    le.fit(np.concatenate([ytr_raw, yte_raw]).astype(str))
    y_train = le.transform(ytr_raw.astype(str)).astype(np.int64)
    y_test = le.transform(yte_raw.astype(str)).astype(np.int64)
    num_classes = len(le.classes_)

    # Save to cache
    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
        )

    return X_train, y_train, X_test, y_test, num_classes


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders."""
    train_ds = MTSDataset(X_train, y_train)
    test_ds = MTSDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(len(train_ds) > batch_size),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
