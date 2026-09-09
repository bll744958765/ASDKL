"""Run the ASDKL experiments reported in the main benchmark table."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


MODERATE_DATASETS = ("synthetic", "pm25")
FULL_DATASETS = ("lucas_large", "california")
MODERATE_SEEDS = (1042, 1052, 1062, 1072, 1082)
FULL_SEEDS = (1042, 1052, 1062, 1072, 1082)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the formal ASDKL benchmark runs.")
    parser.add_argument("--datasets", nargs="+", choices=[*MODERATE_DATASETS, *FULL_DATASETS],
                        default=[*MODERATE_DATASETS, *FULL_DATASETS])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    runner = Path(__file__).resolve().parents[1] / "run.py"
    rows: list[dict] = []
    for dataset in args.datasets:
        full_data = dataset in FULL_DATASETS
        seeds = FULL_SEEDS if full_data else MODERATE_SEEDS
        for seed in seeds:
            output = args.output_dir / dataset / f"seed_{seed}"
            result = output / f"{dataset}_run.json"
            command = [
                sys.executable, str(runner), "--dataset", dataset, "--seed", str(seed),
                "--device", args.device, "--output-dir", str(output),
                "--epochs", "300", "--restore-best", "1", "--checkpoint-every", "25",
                "--neighbors", "64", "--freq-dim", "64", "--hidden-dim", "64",
                "--trend-width", "128", "--trend-depth", "6", "--trend-residual", "1",
                "--fusion", "gated", "--lambda-spectral", "0.1", "--lambda-mse", "0",
                "--lr", "0.003", "--batch-size", "64", "--representation-kernel", "1",
                "--base-kernel", "rbf", "--residual-ensemble", "0", "--local-expert", "0",
                "--covariate-expert", "0",
            ]
            if args.data_root is not None:
                command += ["--data-root", str(args.data_root)]
            if full_data:
                command += ["--gp-inference", "nngp", "--gp-neighbors", "32"]
            else:
                command += ["--gp-inference", "exact"]
            if args.evaluate_only:
                command += ["--evaluate-only", "1"]
            status = "ok" if subprocess.run(command, check=False).returncode == 0 else "failed"
            record = {"dataset": dataset, "seed": seed, "status": status}
            if result.exists():
                record.update(json.loads(result.read_text(encoding="utf-8")))
            rows.append(record)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.output_dir / "asdkl_main_results.csv", index=False)


if __name__ == "__main__":
    main()

