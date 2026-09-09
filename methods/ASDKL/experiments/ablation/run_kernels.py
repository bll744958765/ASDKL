"""Kernel-structure ablation under the frozen 300-epoch protocol."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


KERNELS = {
    "representation_rbf_plus_matern32": (1, "matern32"),
    "representation_rbf_only": (1, "none"),
    "matern32_only": (0, "matern32"),
    "representation_rbf_plus_base_rbf": (1, "rbf"),
}


def completed_epochs(out: Path, result: Path) -> int:
    history_path = out / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if history.get("epoch"):
            return int(max(history["epoch"]))
    return 300 if result.exists() else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["synthetic", "pm25", "lucas_large", "california"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1042, 1052, 1062, 1072, 1082])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reevaluate", action="store_true",
                        help="Reload completed checkpoints and recompute predictions/metrics")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernels", nargs="+", choices=list(KERNELS), default=list(KERNELS))
    args = parser.parse_args()
    runner = Path(__file__).resolve().parents[1] / "run.py"
    rows = []
    for kernel_name in args.kernels:
        representation_kernel, base_kernel = KERNELS[kernel_name]
        for dataset in args.datasets:
            for seed in args.seeds:
                out = args.output_dir / kernel_name / dataset / f"seed_{seed}"
                result = out / f"{dataset}_run.json"
                status = "ok"
                if args.reevaluate or not result.exists() or completed_epochs(out, result) < 300:
                    command = [
                        sys.executable, str(runner), "--dataset", dataset,
                        "--seed", str(seed), "--device", args.device,
                        "--output-dir", str(out),
                        "--epochs", "300", "--restore-best", "1",
                        "--neighbors", "64", "--freq-dim", "64", "--hidden-dim", "64",
                        "--trend-width", "128", "--trend-depth", "6", "--trend-residual", "1",
                        "--fusion", "gated", "--lambda-spectral", "0.1", "--lambda-mse", "0",
                        "--lr", "0.003", "--batch-size", "64",
                        "--representation-kernel", str(representation_kernel),
                        "--base-kernel", base_kernel,
                        "--residual-ensemble", "0", "--local-expert", "0",
                        "--covariate-expert", "0",
                    ]
                    status = "ok" if subprocess.run(command, check=False).returncode == 0 else "failed"
                row = {"kernel": kernel_name, "dataset": dataset, "seed": seed, "status": status}
                if result.exists():
                    row.update(json.loads(result.read_text(encoding="utf-8")))
                rows.append(row)
                args.output_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(args.output_dir / "kernel_ablation_results.csv", index=False)


if __name__ == "__main__":
    main()


