import os
import sys
import json
import argparse
from pathlib import Path
import torch
import numpy as np

# Add parent directory to sys.path so stgog can be imported when running directly
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from stgog.models import STGoGModel
from stgog.data import load_uea_dataset, generate_synthetic_mts, create_dataloaders
from stgog.training import STGoGTrainer
from stgog.utils import load_config, save_config, plot_confusion_matrix, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train ST-GoG for Multivariate Time Series Classification")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML configuration file")
    parser.add_argument("--dataset", type=str, default=None, help="Name of UEA dataset (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu')")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data for testing")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save outputs and plots")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.lr:
        overrides["lr"] = args.lr
    if args.synthetic:
        overrides["use_synthetic"] = True
    if args.output_dir:
        overrides["output_dir"] = args.output_dir

    cfg = load_config(args.config, overrides=overrides)

    # Output directory setup
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save active config
    save_config(cfg, str(out_dir / "config.yaml"))

    # Reproducibility
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Load Data
    print("=" * 60)
    print(f"Loading data for dataset: {cfg['dataset']}")
    if cfg.get("use_synthetic", False):
        print("Using synthetic time series generator...")
        X_train, y_train, X_test, y_test, num_classes = generate_synthetic_mts(
            n_train=cfg.get("synthetic_train_samples", 100),
            n_test=cfg.get("synthetic_test_samples", 40),
            seq_len=cfg.get("synthetic_seq_len", 50),
            n_channels=cfg.get("synthetic_channels", 6),
            num_classes=cfg.get("synthetic_classes", 3),
            seed=seed,
        )
    else:
        X_train, y_train, X_test, y_test, num_classes = load_uea_dataset(
            dataset_name=cfg["dataset"],
            cache_dir=cfg.get("cache_dir", "data/cache"),
        )

    N_tr, T, C = X_train.shape
    N_te = len(y_test)
    print(f"Data Loaded: Train {N_tr} samples, Test {N_te} samples")
    print(f"Shape: Sequence Length T={T}, Channels C={C}, Classes={num_classes}")
    print("=" * 60)

    # Dataloaders
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test,
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", False),
    )

    # 2. Build Model
    model = STGoGModel(
        in_channels=C,
        seq_len=T,
        num_classes=num_classes,
        tcn_hidden=cfg.get("tcn_hidden", 128),
        gcn_hidden=cfg.get("gcn_hidden", 128),
        gog_hidden=cfg.get("gog_hidden", 128),
        num_segments=cfg.get("num_segments", None),
        dropout=cfg.get("dropout", 0.42),
        use_se=cfg.get("use_se", True),
        gcn_radii=cfg.get("gcn_radii", [1, 3, 5]),
        pooling=cfg.get("pooling", "mean"),
        strided_downsample=cfg.get("strided_downsample", None),
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Initialized: {param_count:,} trainable parameters")

    # 3. Setup Trainer
    device = torch.device(args.device) if args.device else None
    trainer = STGoGTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
        label_smoothing=cfg.get("label_smoothing", None),
        mixup_alpha=cfg.get("mixup_alpha", 0.2),
        stochastic_p=cfg.get("stochastic_p", 0.2),
        tta_passes=cfg.get("tta_passes", 7),
        device=device,
        checkpoint_dir=str(ckpt_dir),
        use_amp=cfg.get("use_amp", True),
    )

    # 4. Train Model
    results = trainer.fit(
        epochs=cfg.get("epochs", 100),
        print_every=cfg.get("print_every", 10),
        verbose=True,
    )

    # 5. Save Artifacts & Visualizations
    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": cfg["dataset"],
            "single_best_acc": results["single_best_acc"],
            "single_best_epoch": results["single_best_epoch"],
            "ensemble_acc": results["ensemble_acc"],
            "total_duration_sec": results["total_duration_sec"],
            "metrics": results["metrics"],
        }, f, indent=2)

    # Save Best Checkpoint
    best_ckpt_file = ckpt_dir / "best_model.pt"
    if results["best_state_dict"] is not None:
        torch.save(results["best_state_dict"], best_ckpt_file)
        print(f"Best model weights saved to: {best_ckpt_file}")

    # Plot Visualizations
    plot_training_curves(results["history"], save_path=str(out_dir / "training_curves.png"))
    plot_confusion_matrix(
        y_true=results["y_true"],
        y_pred=results["y_pred"],
        save_path=str(out_dir / "confusion_matrix.png"),
        title=f"{cfg['dataset']} - Confusion Matrix",
    )
    print(f"Artifacts and plots saved in: {out_dir}")


if __name__ == "__main__":
    main()
