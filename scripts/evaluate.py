import sys
import argparse
from pathlib import Path
import torch
import numpy as np

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from stgog.models import STGoGModel
from stgog.data import load_uea_dataset, generate_synthetic_mts, create_dataloaders, test_time_augmentation
from stgog.utils import load_config, compute_classification_metrics, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ST-GoG checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint file")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default=None, help="Name of UEA dataset")
    parser.add_argument("--tta-passes", type=int, default=7, help="Number of TTA passes (1 = no TTA)")
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset:
        cfg["dataset"] = args.dataset

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running evaluation on device: {device}")

    # Load Data
    if args.synthetic or cfg.get("use_synthetic", False):
        _, _, X_test, y_test, num_classes = generate_synthetic_mts()
    else:
        _, _, X_test, y_test, num_classes = load_uea_dataset(
            dataset_name=cfg["dataset"],
            cache_dir=cfg.get("cache_dir", "data/cache"),
        )

    _, T, C = X_test.shape
    _, test_loader = create_dataloaders(
        X_test[:2], y_test[:2], X_test, y_test,
        batch_size=cfg.get("batch_size", 16)
    )

    # Rebuild Model
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
    ).to(device)

    # Load weights
    ckpt_path = Path(args.checkpoint)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint weights from: {ckpt_path}")

    # Evaluate
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = test_time_augmentation(model, xb, passes=args.tta_passes)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(yb.numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = logits_arr.argmax(axis=-1)

    metrics = compute_classification_metrics(y_true, y_pred, logits_or_probs=logits_arr)

    print("=" * 60)
    print(f"Evaluation Results on {cfg['dataset']}:")
    print(f"  Test Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro-F1 Score : {metrics['macro_f1']:.4f}")
    if "ece" in metrics:
        print(f"  Calibration ECE: {metrics['ece']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
