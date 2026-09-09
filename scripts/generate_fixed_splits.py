"""Generate reusable manuscript split files without exporting observations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "common"))

from data_loader import make_spatial_splits
from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+",
        default=["synthetic", "pm25", "lucas_large", "california"],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[1042, 1052, 1062, 1072, 1082],
    )
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "splits")
    args = parser.parse_args()

    for dataset in args.datasets:
        for seed in args.seeds:
            coordinates, covariates, response = load_dataset(
                dataset, args.data_root, seed=seed, max_samples=None
            )
            _, (train_idx, val_idx, test_idx), _ = make_spatial_splits(
                coordinates, covariates, response, seed=seed
            )
            target = args.output_dir / dataset
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target / f"seed_{seed}.npz",
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )


if __name__ == "__main__":
    main()
