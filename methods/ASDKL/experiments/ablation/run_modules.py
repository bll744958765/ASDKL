"""Five-seed module/loss ablations matched to the frozen ASDKL model."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


CONFIGS = {
    "six_block_reference": [],
    "no_context": ["--ablation", "no_context"],
    "no_fourier": ["--ablation", "no_fourier"],
    "independent_frequencies": ["--ablation", "independent_frequencies"],
    "no_relation_encoder": ["--ablation", "no_relation_encoder"],
    "no_spectral_regularization": ["--ablation", "no_spectral_reg"],
    "no_spectral_attention": ["--ablation", "no_spectral_attention"],
    "no_gp": ["--ablation", "no_gp"],
    "no_residual_connections": ["--trend-residual", "0"],
    "three_residual_blocks": ["--trend-depth", "3"],
    "auxiliary_mse": ["--lambda-mse", "0.1"],
}


def completed_epochs(out: Path, result: Path) -> int:
    history_path = out / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        epochs = history.get("epoch", [])
        if epochs:
            return int(max(epochs))
    return 300 if result.exists() else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["synthetic", "lucas", "pm25", "california"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1042, 1052, 1062, 1072, 1082])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reevaluate", action="store_true",
                        help="Reload selected checkpoints and recompute metrics without training")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configurations", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = parser.parse_args()

    runner = Path(__file__).resolve().parents[1] / "run.py"
    common = [
        "--epochs", "300",
        "--neighbors", "64",
        "--freq-dim", "64",
        "--hidden-dim", "64",
        "--lambda-spectral", "0.1",
        "--lr", "0.003",
        "--batch-size", "64",
        "--fusion", "gated",
        "--trend-width", "128",
        "--trend-depth", "6",
        "--trend-residual", "1",
        "--residual-ensemble", "0",
        "--local-expert", "0",
        "--covariate-expert", "0",
        "--representation-kernel", "1",
        "--base-kernel", "rbf",
        "--lambda-mse", "0.0",
        "--restore-best", "1",
    ]
    rows = []
    for configuration in args.configurations:
        override = CONFIGS[configuration]
        for dataset in args.datasets:
            for seed in args.seeds:
                out = args.output_dir / configuration / dataset / f"seed_{seed}"
                result = out / f"{dataset}_run.json"
                status = "ok"
                if args.reevaluate or not result.exists() or completed_epochs(out, result) < 300:
                    # Later command-line values override argparse defaults, but
                    # duplicate flags are avoided here to keep run manifests clear.
                    overridden = {override[i] for i in range(0, len(override), 2)}
                    base = []
                    for i in range(0, len(common), 2):
                        if common[i] not in overridden:
                            base.extend(common[i:i + 2])
                    command = [
                        sys.executable,
                        str(runner),
                        "--dataset", dataset,
                        "--seed", str(seed),
                        "--device", args.device,
                        "--output-dir", str(out),
                        *base,
                        *override,
                    ]
                    if args.reevaluate:
                        command += ["--evaluate-only", "1"]
                    status = "ok" if subprocess.run(command, check=False).returncode == 0 else "failed"
                record = {"configuration": configuration, "dataset": dataset, "seed": seed, "status": status}
                if result.exists():
                    record.update(json.loads(result.read_text(encoding="utf-8")))
                rows.append(record)
                args.output_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(args.output_dir / "ablation_results.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "ablation_results.csv", index=False)


if __name__ == "__main__":
    main()

