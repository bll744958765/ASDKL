"""Aggregate five-seed kernel-structure ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


WORK = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=WORK / "generated")
args = parser.parse_args()
ROOT = args.input_dir
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)

ORDER = [
    "representation_rbf_plus_matern32",
    "representation_rbf_only",
    "matern32_only",
    "representation_rbf_plus_base_rbf",
]
LABELS = {
    "representation_rbf_plus_matern32": r"Representation RBF + Mat\'ern-$3/2$",
    "representation_rbf_only": "Representation RBF only",
    "matern32_only": r"Mat\'ern-$3/2$ only",
    "representation_rbf_plus_base_rbf": "Representation RBF + base-space RBF",
}
DATASETS = ["synthetic", "pm25", "lucas_large", "california"]
DLABELS = {"synthetic": "Nonstationary synthetic", "pm25": r"EPA PM$_{2.5}$", "lucas_large": "LUCAS", "california": "California"}
METRICS = ["rmse", "nll", "crps"]

rows = []
for kernel in ORDER:
    for dataset in DATASETS:
        for path in sorted((ROOT / kernel / dataset).glob("seed_*/*_run.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"kernel": kernel, "dataset": dataset, "seed": int(item["seed"]),
                         **{metric: float(item[metric]) for metric in METRICS}})
source = pd.DataFrame(rows)
if len(source) != len(ORDER) * len(DATASETS) * 5:
    raise RuntimeError(f"Expected 40 runs, found {len(source)}")
source.to_csv(OUT / "kernel_ablation_epoch300_source.csv", index=False)
summary = source.groupby(["dataset", "kernel"])[METRICS].agg(["mean", "std"]).reset_index()
summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
summary.to_csv(OUT / "kernel_ablation_epoch300_summary.csv", index=False)

slash = chr(92)
row_end = slash * 2
lines = [
    r"\begin{table*}[htbp]", r"\centering", r"\scriptsize",
    r"\setlength{\tabcolsep}{3.2pt}",
    r"\caption{Kernel-structure comparison over five seeds. Every run uses the same six-block architecture, a maximum of 300 epochs and validation-MSE checkpoint selection; external regression-kriging and local-predictor mixtures are disabled. Values are mean $\pm$ s.d.; lower is better. Best values are bold and second-best values are underlined within each dataset and metric.}",
    r"\label{tab:kernel-ablation}", r"\begin{tabular}{llccc}", r"\toprule",
    "Dataset & Residual covariance & RMSE $\\downarrow$ & NLL $\\downarrow$ & CRPS $\\downarrow$ " + row_end,
    r"\midrule",
]
for di, dataset in enumerate(DATASETS):
    block = summary[summary.dataset.eq(dataset)]
    rankings = {metric: block.sort_values(f"{metric}_mean").kernel.tolist() for metric in METRICS}
    for ki, kernel in enumerate(ORDER):
        row = block[block.kernel.eq(kernel)].iloc[0]
        cells = []
        for metric in METRICS:
            cell = f"{row[f'{metric}_mean']:.3f} $\\pm$ {row[f'{metric}_std']:.3f}"
            if kernel == rankings[metric][0]:
                cell = r"\textbf{" + cell + "}"
            elif kernel == rankings[metric][1]:
                cell = r"\underline{" + cell + "}"
            cells.append(cell)
        dataset_cell = DLABELS[dataset] if ki == 0 else ""
        lines.append(dataset_cell + " & " + LABELS[kernel] + " & " + " & ".join(cells) + " " + row_end)
    if di == 0:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(OUT / "generated_kernel_ablation_epoch300.tex").write_text("\n".join(lines), encoding="utf-8")
print(summary[["dataset", "kernel", "rmse_mean", "rmse_std", "nll_mean", "nll_std", "crps_mean", "crps_std"]].to_string(index=False))

