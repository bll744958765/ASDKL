"""Run ASDKL across reduced EPA observation densities using grouped locations."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ASDKL_DIR = HERE.parents[1]
REPOSITORY_ROOT = ASDKL_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "common"))
from datasets import load_dataset

SEEDS = [1042, 1052, 1062, 1072, 1082]
FRACTIONS = [10, 20, 30, 40, 50, 60, 70]


def grouped_split(coords, train_fraction, val_fraction, seed):
    """Keep identical coordinates together across train, validation and test."""
    rounded = np.round(np.asarray(coords, float), 8)
    _, group_id = np.unique(rounded, axis=0, return_inverse=True)
    groups = np.random.default_rng(seed).permutation(np.unique(group_id))

    def take(pool, target):
        chosen, count = [], 0
        for group in pool:
            chosen.append(int(group)); count += int(np.sum(group_id == group))
            if count >= target:
                break
        return set(chosen)

    test_groups = take(groups, int(round(0.20 * len(coords))))
    remaining = [group for group in groups if group not in test_groups]
    validation_groups = take(remaining, int(round(val_fraction * len(coords))))
    train_pool = [group for group in remaining if group not in validation_groups]
    train_groups = take(train_pool, int(round(train_fraction * len(coords))))
    train = np.flatnonzero(np.isin(group_id, list(train_groups)))
    validation = np.flatnonzero(np.isin(group_id, list(validation_groups)))
    test = np.flatnonzero(np.isin(group_id, list(test_groups)))
    assert not (set(group_id[train]) & set(group_id[validation]))
    assert not (set(group_id[train]) & set(group_id[test]))
    assert not (set(group_id[validation]) & set(group_id[test]))
    return train, validation, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "results" / "epa_sparse_fraction")
    parser.add_argument("--fractions", nargs="+", type=int, default=FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data")
    args = parser.parse_args()
    coordinates, _, _ = load_dataset("pm25", args.data_root, args.seeds[0], None)
    rows = []
    for percentage in args.fractions:
        split_dir = args.output_dir / f"fraction_{percentage}" / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            train, validation, test = grouped_split(coordinates, percentage / 100, 0.10, seed)
            split_file = split_dir / f"seed_{seed}.npz"
            np.savez_compressed(split_file, train_idx=train, val_idx=validation, test_idx=test)
            output = args.output_dir / f"fraction_{percentage}" / "ASDKL" / f"seed_{seed}"
            result = output / "pm25_run.json"
            if not result.exists():
                command = [sys.executable, str(ASDKL_DIR / "run.py"), "--dataset", "pm25", "--seed", str(seed), "--device", args.device, "--data-root", str(args.data_root), "--output-dir", str(output), "--split-file", str(split_file), "--epochs", str(args.epochs), "--neighbors", "64", "--freq-dim", "64", "--hidden-dim", "64", "--fusion", "gated", "--trend-width", "128", "--trend-depth", "6", "--trend-residual", "1", "--lambda-mse", "0", "--lambda-spectral", "0.1", "--lr", "0.003", "--batch-size", "64", "--restore-best", "1", "--representation-kernel", "1", "--base-kernel", "rbf", "--gp-inference", "exact", "--resume", "1"]
                subprocess.run(command, check=True)
            record = json.loads(result.read_text(encoding="utf-8"))
            record["training_fraction_pct"] = percentage
            rows.append(record)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.output_dir / "asdkl_fraction_results.csv", index=False)


if __name__ == "__main__":
    main()
