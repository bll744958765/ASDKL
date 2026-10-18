"""Run all EPA sparse probabilistic comparator cells used in manuscript Table 4.

The split files are produced by experiments/run_epa_sparse.py. This script
uses the same frozen train/validation/test indices for each baseline and
stops immediately if one run fails. Re-running skips completed JSON results.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = (10, 20, 30, 40, 50, 60)
METHODS = ("SimpleKriging", "RegressionKriging", "RFKriging")
SEEDS = (1042, 1052, 1062, 1072, 1082)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--results-root", type=Path,
                        default=ROOT / "results" / "epa_sparse_fraction_fixedtest")
    args = parser.parse_args()
    runner = ROOT / "experiments" / "run_one_epa_sparse_probabilistic_baseline.py"
    for fraction in FRACTIONS:
        part = args.results_root / f"fraction_{fraction}"
        split_dir = part / "splits"
        for seed in SEEDS:
            split = split_dir / f"seed_{seed}.npz"
            if not split.exists():
                raise FileNotFoundError(f"Run experiments/run_epa_sparse.py first: {split}")
            for method in METHODS:
                command = [sys.executable, str(runner), "--method", method,
                           "--seed", str(seed), "--training-fraction-pct", str(fraction),
                           "--split-dir", str(split_dir), "--output-dir",
                           str(part / "probabilistic_baselines"), "--data-root", str(args.data_root)]
                subprocess.run(command, cwd=ROOT, check=True)
    print("Completed 90 EPA sparse probabilistic baseline cells.")


if __name__ == "__main__":
    main()
