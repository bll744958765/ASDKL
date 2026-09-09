"""Validation-only OFAT sensitivity analysis for the frozen ASDKL model.

The default configuration is exactly the one used for the five-seed final
comparison.  Every non-default setting changes one factor only.  Test metrics
are saved for audit, but configuration selection and the manuscript plots use
validation RMSE exclusively.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT = dict(
    neighbors=64,
    freq_dim=64,
    hidden_dim=64,
    lambda_spectral=0.1,
    lr=3e-3,
    epochs=300,
    batch_size=64,
    fusion="gated",
    trend_width=128,
    trend_depth=6,
    dropout=0.0,
    trend_residual=1,
    lambda_mse=0.0,
)

# Four prespecified levels per factor; each family contains the frozen default.
LEVELS = {
    "neighbors": [16, 32, 64, 96],
    "freq_dim": [16, 32, 64, 128],
    "hidden_dim": [32, 64, 128, 256],
    "lambda_spectral": [0.0, 0.01, 0.1, 0.5],
    "lr": [3e-4, 1e-3, 3e-3, 1e-2],
    "epochs": [100, 200, 300, 400],
}


def completed_epochs(out: Path, result: Path) -> int:
    history_path = out / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        epochs = history.get("epoch", [])
        if epochs:
            return int(max(epochs))
    return 300 if result.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["synthetic", "california"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1042, 1052, 1062, 1072, 1082])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=list(LEVELS), default=list(LEVELS))
    parser.add_argument("--include-default", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--epoch-chain-source-dir", type=Path, default=None,
        help=("Completed 300-epoch default runs used so epoch sensitivity follows "
              "one trajectory: 100->200 and default 300->400."),
    )
    parser.add_argument("--reevaluate", action="store_true",
                        help="Reload selected checkpoints and recompute validation/test metrics")
    args = parser.parse_args()

    runner = Path(__file__).resolve().parents[1] / "run.py"
    configs = [("default", "default", DEFAULT.copy())] if args.include_default else []
    for family in args.families:
        values = LEVELS[family]
        for value in values:
            if value == DEFAULT[family]:
                continue
            candidate = DEFAULT.copy()
            candidate[family] = value
            configs.append((family, value, candidate))

    rows: list[dict] = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for family, value, cfg in configs:
                tag = str(value).replace(".", "p")
                out = args.output_dir / dataset / f"seed_{seed}" / f"{family}_{tag}"
                result = out / f"{dataset}_run.json"
                status = "ok"
                if args.reevaluate or not result.exists() or completed_epochs(out, result) < int(cfg["epochs"]):
                    # Compare epoch budgets along one optimization trajectory.
                    # The runner restores model, GP, optimizer, history, and RNG states.
                    state = out / f"{dataset}_training_state.pt"
                    if family == "epochs" and not state.exists() and not args.reevaluate:
                        source = None
                        if int(value) == 200:
                            source = (args.output_dir / dataset / f"seed_{seed}" /
                                      "epochs_100" / f"{dataset}_training_state.pt")
                        elif int(value) == 400 and args.epoch_chain_source_dir is not None:
                            source = (args.epoch_chain_source_dir / dataset / f"seed_{seed}" /
                                      f"{dataset}_training_state.pt")
                        if source is not None:
                            if not source.exists():
                                raise FileNotFoundError(f"Missing epoch-chain checkpoint: {source}")
                            out.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source, state)
                    command = [
                        sys.executable,
                        str(runner),
                        "--dataset", dataset,
                        "--seed", str(seed),
                        "--device", args.device,
                        "--output-dir", str(out),
                        "--neighbors", str(cfg["neighbors"]),
                        "--freq-dim", str(cfg["freq_dim"]),
                        "--hidden-dim", str(cfg["hidden_dim"]),
                        "--lambda-spectral", str(cfg["lambda_spectral"]),
                        "--lr", str(cfg["lr"]),
                        "--epochs", str(cfg["epochs"]),
                        "--batch-size", str(cfg["batch_size"]),
                        "--fusion", cfg["fusion"],
                        "--trend-width", str(cfg["trend_width"]),
                        "--trend-depth", str(cfg["trend_depth"]),
                        "--dropout", str(cfg["dropout"]),
                        "--trend-residual", str(cfg["trend_residual"]),
                        "--lambda-mse", str(cfg["lambda_mse"]),
                        "--restore-best", "1",
                        "--representation-kernel", "1", "--base-kernel", "rbf",
                        "--residual-ensemble", "0", "--local-expert", "0",
                        "--covariate-expert", "0",
                    ]
                    if args.reevaluate:
                        command += ["--evaluate-only", "1"]
                    status = "ok" if subprocess.run(command, check=False).returncode == 0 else "failed"
                record = {"dataset": dataset, "seed": seed, "family": family, "value": value, **cfg, "status": status}
                if result.exists():
                    record.update(json.loads(result.read_text(encoding="utf-8")))
                rows.append(record)
                args.output_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(args.output_dir / "sensitivity_results.csv", index=False)

    pd.DataFrame(rows).sort_values(["dataset", "family", "value", "seed"]).to_csv(
        args.output_dir / "sensitivity_results.csv", index=False
    )


if __name__ == "__main__":
    main()

