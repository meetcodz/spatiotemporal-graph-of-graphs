import time
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .scheduler import build_cosine_scheduler, compute_stochastic_depth, get_snapshot_epochs
from .snapshot import SnapshotManager, get_base_module
from ..data.augmentations import MixUp, test_time_augmentation
from ..utils.metrics import compute_classification_metrics


class STGoGTrainer:
    """
    Hardware-aware trainer for ST-GoG models.
    Supports:
      - Automatic GPU / CPU device routing
      - Mixed-precision training via torch.amp
      - MixUp data augmentation
      - Exponential stochastic depth decay
      - Cosine annealing with warm restarts
      - Snapshot ensembling and multi-pass Test-Time Augmentation (TTA)
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: Optional[float] = None,
        mixup_alpha: float = 0.2,
        stochastic_p: float = 0.2,
        tta_passes: int = 7,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = "checkpoints",
        use_amp: bool = True,
    ):
        self.raw_model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes

        # Device configuration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self.raw_model.to(self.device)

        # Multi-GPU support if >1 GPU
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        # Optimization
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = build_cosine_scheduler(self.optimizer, T_0=50, T_mult=2)

        # Loss function with adaptive label smoothing
        if label_smoothing is None:
            label_smoothing = 0.10 if num_classes <= 5 else 0.15
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Regularization & augmentation
        self.mixup = MixUp(alpha=mixup_alpha)
        self.stochastic_p = stochastic_p
        self.tta_passes = tta_passes

        # AMP setup
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Snapshot management
        self.snapshot_mgr = SnapshotManager(checkpoint_dir=checkpoint_dir)
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_state: Optional[Dict[str, Any]] = None

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch with MixUp and gradient scaling."""
        self.model.train()
        # Update stochastic depth drop rate
        current_p = compute_stochastic_depth(self.stochastic_p, epoch=epoch)
        get_base_module(self.model).set_stochastic_depth(current_p)

        total_loss = 0.0
        num_batches = 0

        for xb, yb in self.train_loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            xb_mixed, ya, yb_m, lam = self.mixup(xb, yb)
            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(xb_mixed)
                loss = lam * self.criterion(out, ya) + (1.0 - lam) * self.criterion(out, yb_m)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self.scheduler.step()
        return total_loss / max(1, num_batches)

    def evaluate(self, loader: Optional[DataLoader] = None, use_tta: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model on dataset."""
        if loader is None:
            loader = self.test_loader

        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                if use_tta:
                    logits = test_time_augmentation(self.model, xb, passes=self.tta_passes)
                else:
                    logits = self.model(xb)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(yb.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        acc = float((y_pred == y_true).mean())
        return acc, y_pred, y_true

    def fit(self, epochs: int = 100, print_every: int = 10, verbose: bool = True) -> Dict[str, Any]:
        """Execute full training schedule with snapshot capturing and final ensemble evaluation."""
        snapshot_epochs = get_snapshot_epochs(epochs, T_0=50, T_mult=2)
        history = {"train_loss": [], "val_acc": [], "epoch_times": []}
        start_time = time.time()

        if verbose:
            print(f"Starting ST-GoG training on {self.device} for {epochs} epochs...")
            print(f"Snapshot saves scheduled at epochs: {sorted(snapshot_epochs)}")

        for ep in range(1, epochs + 1):
            ep_start = time.time()
            loss = self.train_epoch(ep)
            acc, _, _ = self.evaluate(use_tta=(self.tta_passes > 1))
            ep_time = time.time() - ep_start

            history["train_loss"].append(loss)
            history["val_acc"].append(acc)
            history["epoch_times"].append(ep_time)

            is_best = acc > self.best_acc
            if is_best:
                self.best_acc = acc
                self.best_epoch = ep
                self.best_state = {
                    k: v.cpu().clone() for k, v in get_base_module(self.model).state_dict().items()
                }

            if ep in snapshot_epochs:
                self.snapshot_mgr.add_snapshot(self.model, epoch=ep, accuracy=acc)
                if verbose:
                    print(f"  --> Snapshot saved at epoch {ep:03d} (acc: {acc*100:.2f}%)")

            if verbose and (ep % print_every == 0 or is_best or ep == epochs):
                lr_curr = self.optimizer.param_groups[0]["lr"]
                tag = "* BEST" if is_best else ""
                print(
                    f"Epoch {ep:03d}/{epochs:03d} | Loss: {loss:.4f} | Acc: {acc*100:.2f}% "
                    f"| Best: {self.best_acc*100:.2f}% (ep {self.best_epoch}) | LR: {lr_curr:.2e} {tag}"
                )

        total_duration = time.time() - start_time

        # Run final ensemble inference across all snapshots
        if verbose:
            print("\nEvaluating Snapshot Ensemble across saved checkpoints...")
        ensemble_logits = self.snapshot_mgr.predict_ensemble(
            loader=self.test_loader,
            ref_model=self.model,
            device=self.device,
            tta_passes=self.tta_passes,
            include_best=self.best_state,
        )
        ens_preds = ensemble_logits.argmax(dim=-1).numpy()
        test_targets = np.concatenate([yb.numpy() for _, yb in self.test_loader])
        ens_acc = float((ens_preds == test_targets).mean())

        # Full classification metrics
        metrics = compute_classification_metrics(test_targets, ens_preds, ensemble_logits.numpy())

        results = {
            "single_best_acc": self.best_acc,
            "single_best_epoch": self.best_epoch,
            "ensemble_acc": ens_acc,
            "total_duration_sec": total_duration,
            "y_pred": ens_preds,
            "y_true": test_targets,
            "metrics": metrics,
            "history": history,
            "best_state_dict": self.best_state,
        }

        if verbose:
            print("=" * 60)
            print(f"Training Complete in {total_duration/60:.2f} minutes")
            print(f"Single Best Model Acc : {self.best_acc*100:.2f}% (Epoch {self.best_epoch})")
            print(f"Snapshot Ensemble Acc : {ens_acc*100:.2f}% (Lift: {(ens_acc - self.best_acc)*100:+.2f}%)")
            print("=" * 60)

        return results
