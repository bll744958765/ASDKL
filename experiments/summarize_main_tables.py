"""Create manuscript table source and five-seed summaries from fresh runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATASETS = ("synthetic", "pm25", "lucas_large", "california")
SEEDS = (1042, 1052, 1062, 1072, 1082)
POINT_METRICS = ("rmse", "mae", "r2")
PROB_METRICS = ("nll", "crps", "picp90", "mpiw90")


def require_complete(frame: pd.DataFrame, methods: int, label: str) -> None:
    expected = len(DATASETS) * len(SEEDS) * methods
    if len(frame) != expected:
        raise ValueError(f"{label}: expected {expected} runs, found {len(frame)}")
    counts = frame.groupby(["dataset", "method"]).seed.nunique()
    if not counts.eq(len(SEEDS)).all():
        raise ValueError(f"{label}: at least one dataset/method lacks five seeds")


def summarize(frame: pd.DataFrame, metrics: tuple[str, ...], output: Path, stem: str) -> None:
    source = frame[["dataset", "method", "seed", *metrics]].copy()
    source.to_csv(output / f"{stem}_source.csv", index=False)
    summary = source.groupby(["dataset", "method"], sort=False)[list(metrics)].agg(["mean", "std"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    summary.to_csv(output / f"{stem}_summary.csv")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--asdkl-dir", type=Path, default=root / "results" / "asdkl_main")
    parser.add_argument("--baseline-file", type=Path, default=root / "results" / "benchmarks" / "baseline_results.csv")
    parser.add_argument("--probabilistic-file", type=Path, default=root / "results" / "probabilistic_baselines" / "baseline_results.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "main_table_summaries")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = args.asdkl_dir / dataset / f"seed_{seed}" / f"{dataset}_run.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            records.append({"dataset": dataset, "seed": seed, "method": "ASDKL", **row})
    asdkl = pd.DataFrame(records)
    require_complete(asdkl, 1, "ASDKL")

    baseline = pd.read_csv(args.baseline_file)
    if "status" in baseline and not baseline.status.eq("ok").all():
        raise ValueError("At least one point baseline run failed")
    require_complete(baseline, 13, "point baselines")
    summarize(pd.concat([baseline, asdkl], ignore_index=True), POINT_METRICS,
              args.output_dir, "point_prediction")

    probability = pd.read_csv(args.probabilistic_file)
    require_complete(probability, 3, "probabilistic baselines")
    summarize(pd.concat([probability, asdkl], ignore_index=True), PROB_METRICS,
              args.output_dir, "probabilistic_prediction")
    print(args.output_dir)


if __name__ == "__main__":
    main()
