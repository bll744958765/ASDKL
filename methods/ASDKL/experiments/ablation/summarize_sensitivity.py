"""Aggregate frozen-model sensitivity runs and export two 2-by-3 figures."""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


WORK = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=WORK / "generated")
args = parser.parse_args()
RESULTS = args.input_dir
OUT = args.output_dir / "figures"
OUT.mkdir(parents=True, exist_ok=True)

frame = pd.read_csv(RESULTS / "sensitivity_results.csv")
frame = frame[frame.status.eq("ok")].copy()
families = ["neighbors", "freq_dim", "hidden_dim", "lambda_spectral", "lr", "epochs"]
default = frame[frame.family.eq("default")][["dataset", "seed", "val_rmse"]].rename(
    columns={"val_rmse": "default_val_rmse"}
)
parts = []
for family in families:
    varied = frame[frame.family.eq(family)].copy()
    shared = frame[frame.family.eq("default")].copy()
    shared["family"] = family
    shared["value"] = shared[family]
    parts.append(pd.concat([varied, shared], ignore_index=True))
source = pd.concat(parts, ignore_index=True).merge(default, on=["dataset", "seed"])
source["value"] = pd.to_numeric(source.value)
source["relative_val_rmse"] = source.val_rmse / source.default_val_rmse
(args.output_dir / "source_data").mkdir(parents=True, exist_ok=True)
source.to_csv(args.output_dir / "source_data" / "sensitivity_source.csv", index=False)

summary = source.groupby(["dataset", "family", "value"], as_index=False).agg(
    val_rmse=("val_rmse", "mean"),
    val_rmse_sd=("val_rmse", "std"),
    relative_val_rmse=("relative_val_rmse", "mean"),
    relative_val_rmse_sd=("relative_val_rmse", "std"),
    n=("seed", "count"),
)
summary.to_csv(args.output_dir / "source_data" / "sensitivity_summary.csv", index=False)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
titles = {
    "neighbors": "Spatial neighbours $K$",
    "freq_dim": "Frequency dimension $M$",
    "hidden_dim": "Representation width",
    "lambda_spectral": "Spectral weight $\\lambda_s$",
    "lr": "Learning rate",
    "epochs": "Training epochs",
}
colors = {"synthetic": "#2F6F8F"}
critical = t.ppf(0.975, 2)
all_ci = critical * summary.relative_val_rmse_sd / np.sqrt(summary.n)
ymin = float(np.nanmin(summary.relative_val_rmse - all_ci)) - 0.03
ymax = float(np.nanmax(summary.relative_val_rmse + all_ci)) + 0.03

for dataset in ("synthetic",):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True)
    for panel, (ax, family) in enumerate(zip(axes.flat, families)):
        group = summary[(summary.dataset.eq(dataset)) & (summary.family.eq(family))].sort_values("value")
        values = group.relative_val_rmse.to_numpy(float)
        ci = critical * group.relative_val_rmse_sd.to_numpy(float) / np.sqrt(group.n.to_numpy(float))
        positions = np.arange(len(group))
        ax.fill_between(positions, values - ci, values + ci, color=colors[dataset], alpha=0.16, linewidth=0)
        ax.plot(positions, values, marker="o", ms=3.5, lw=1.1, color=colors[dataset])
        ax.set_xticks(positions, [f"{value:g}" for value in group.value])
        ax.axhline(1.0, color="#666666", ls="--", lw=0.7)
        ax.set_ylim(ymin, ymax)
        ax.set_title(titles[family])
        ax.set_xlabel("Value")
        ax.set_ylabel("Validation RMSE / default")
        ax.grid(axis="y", color="#E7ECEF", lw=0.5)
        ax.text(-0.13, 1.07, chr(97 + panel), transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
    for extension in ("pdf", "svg", "png", "tiff"):
        fig.savefig(
            OUT / f"hyperparameter_sensitivity_{dataset}_epoch300.{extension}",
            dpi=600 if extension == "tiff" else 300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


print(summary.to_string(index=False))


