from typing import List, Optional, Dict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = "outputs/confusion_matrix.png",
    title: str = "Confusion Matrix",
):
    """Plot and save normalized confusion matrix."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    cm = np.zeros((len(class_names) if class_names else 2, len(class_names) if class_names else 2))
    from sklearn.metrics import confusion_matrix
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=True)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    if class_names and len(class_names) > 6:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = "outputs/training_curves.png",
):
    """Plot training loss and validation accuracy curves."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Loss Curve
    ax1.plot(epochs, history["train_loss"], color="#1E3A8A", linewidth=1.8, label="Train Loss")
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("CrossEntropy Loss", fontsize=10, fontweight="bold")
    ax1.set_title("Training Loss Curve", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Accuracy Curve
    val_acc_pct = [acc * 100 for acc in history["val_acc"]]
    ax2.plot(epochs, val_acc_pct, color="#0D9488", linewidth=1.8, label="Test / Val Acc")
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Evaluation Accuracy Curve", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig
