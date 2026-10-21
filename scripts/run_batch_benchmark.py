import sys
import argparse
import time
from pathlib import Path
import pandas as pd
import torch

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from stgog.models import STGoGModel
from stgog.data import load_uea_dataset, create_dataloaders
from stgog.training import STGoGTrainer
from stgog.utils import load_config

UEA_22_DATASETS = [
    "ArticularyWordRecognition", "AtrialFibrillation", "DuckDuckGeese", 
    "EigenWorms", "Epilepsy", "FaceDetection", "FingerMovements", 
    "HandMovementDirection", "Handwriting", "Heartbeat", "Libras", "LSST", 
    "MotorImagery", "NATOPS", "PenDigits", "PEMS-SF", "PhonemeSpectra", 
    "RacketSports", "SelfRegulationSCP1", "SelfRegulationSCP2", 
    "StandWalkJump", "UWaveGestureLibrary"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run ST-GoG across UEA benchmark datasets")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per dataset")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--datasets", nargs="+", default=None, help="Specific datasets to run (default: all 22)")
    parser.add_argument("--output-csv", type=str, default="outputs/benchmark_results.csv", help="CSV save path")
    return parser.parse_args()


def main():
    args = parse_args()
    target_datasets = args.datasets or UEA_22_DATASETS
    results = []

    print(f"Beginning benchmark run across {len(target_datasets)} datasets...")

    for idx, d_name in enumerate(target_datasets, 1):
        print("\n" + "=" * 70)
        print(f"[{idx}/{len(target_datasets)}] Benchmark Dataset: {d_name}")
        print("=" * 70)
        try:
            X_tr, y_tr, X_te, y_te, num_classes = load_uea_dataset(d_name)
            N_tr, T, C = X_tr.shape
            N_te = len(y_te)

            train_loader, test_loader = create_dataloaders(
                X_tr, y_tr, X_te, y_te, batch_size=args.batch_size
            )

            # Strided downsample for ultra-long sequences like EigenWorms
            downsample = 4 if T > 2000 else None

            model = STGoGModel(
                in_channels=C,
                seq_len=T,
                num_classes=num_classes,
                strided_downsample=downsample,
            )

            trainer = STGoGTrainer(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=num_classes,
            )

            res = trainer.fit(epochs=args.epochs, print_every=25, verbose=False)

            results.append({
                "Dataset": d_name,
                "Train (N)": N_tr,
                "Channels (C)": C,
                "Length (T)": T,
                "Classes": num_classes,
                "Best Single Acc (%)": f"{res['single_best_acc']*100:.2f}",
                "Ensemble Acc (%)": f"{res['ensemble_acc']*100:.2f}",
                "Duration (min)": f"{res['total_duration_sec']/60:.1f}",
                "Status": "Success"
            })
            print(f"✅ {d_name} finished: Single Best={res['single_best_acc']*100:.2f}% | Ensemble={res['ensemble_acc']*100:.2f}%")

        except Exception as e:
            print(f"❌ Failed {d_name}: {e}")
            results.append({
                "Dataset": d_name,
                "Train (N)": 0, "Channels (C)": 0, "Length (T)": 0, "Classes": 0,
                "Best Single Acc (%)": "N/A", "Ensemble Acc (%)": "N/A",
                "Duration (min)": "0", "Status": f"Error: {e}"
            })

    # Save summary dataframe
    df = pd.DataFrame(results)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("\n" + "=" * 70)
    print("Benchmark summary:")
    print(df.to_string())
    print(f"\nSaved CSV report to: {out_path}")


if __name__ == "__main__":
    main()
