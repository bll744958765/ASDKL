"""Aggregate final five-seed ablations into source data, table and figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORK = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=WORK / "generated")
args = parser.parse_args()
ABLATIONS = args.input_dir
OUT_TABLE = args.output_dir / "tables"
OUT_FIGURE = args.output_dir / "figures"
OUT_TABLE.mkdir(parents=True, exist_ok=True)
OUT_FIGURE.mkdir(parents=True, exist_ok=True)

datasets = ["synthetic", "pm25", "lucas_large", "california"]
dataset_labels = {
    "synthetic": "Nonstationary synthetic",
    "pm25": r"EPA PM$_{2.5}$",
}
configurations = [
    "four_block_reference",
    "no_context",
    "no_fourier",
    "no_spectral_regularization",
    "no_spectral_attention",
    "no_gp",
    "no_residual_connections",
    "three_residual_blocks",
    "six_residual_blocks",
    "auxiliary_mse",
]
configuration_labels = {
    "four_block_reference": "Four-block development reference",
    "no_context": "Without local context encoder",
    "no_fourier": "Without adaptive Fourier features",
    "no_spectral_regularization": "Without spectral regularization",
    "no_spectral_attention": "Without spectral attention",
    "no_gp": "Without residual GP",
    "no_residual_connections": "Without residual connections",
    "three_residual_blocks": "Three residual blocks",
    "six_residual_blocks": "Selected ASDKL (six residual blocks)",
    "auxiliary_mse": "With auxiliary MSE",
}
metrics = ["rmse", "mae", "r2", "nll", "crps", "picp90", "mpiw90"]


def read(path, configuration):
    content = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset": content["dataset"],
        "seed": int(content["seed"]),
        "configuration": configuration,
        **{metric: float(content[metric]) for metric in metrics},
    }


rows = []
for configuration in configurations:
    for dataset in datasets:
        for path in sorted((ABLATIONS / configuration / dataset).glob("seed_*/*_run.json")):
            rows.append(read(path, configuration))
source = pd.DataFrame(rows).sort_values(["dataset", "configuration", "seed"])
expected = len(datasets) * len(configurations) * 5
if len(source) != expected:
    raise RuntimeError(f"Expected {expected} complete runs, found {len(source)}")
source.to_csv(OUT_TABLE / "ablation_epoch300_source.csv", index=False)

summary = source.groupby(["dataset", "configuration"])[metrics].agg(["mean", "std"]).reset_index()
summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
summary.to_csv(OUT_TABLE / "ablation_epoch300_summary.csv", index=False)

full = source[source.configuration.eq("four_block_reference")][["dataset", "seed", "rmse"]].rename(columns={"rmse": "full_rmse"})
paired = source.merge(full, on=["dataset", "seed"], how="left")
paired["delta_rmse_pct"] = 100 * (paired.rmse / paired.full_rmse - 1)
paired.to_csv(OUT_TABLE / "ablation_epoch300_paired.csv", index=False)

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
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8), sharey=True, constrained_layout=True)
for ax, dataset in zip(axes, datasets):
    subset = paired[paired.dataset.eq(dataset)]
    for position, configuration in enumerate(configurations):
        values = subset[subset.configuration.eq(configuration)].delta_rmse_pct.to_numpy()
        ax.scatter(values, np.full(values.size, position), s=10, color="#AAB7C2", alpha=0.72, zorder=2)
        ax.errorbar(values.mean(), position, xerr=values.std(ddof=1), fmt="o", ms=4.2, color="#B6493A" if configuration == "four_block_reference" else "#236987", ecolor="#236987", elinewidth=0.7, capsize=2, zorder=3)
    ax.axvline(0, color="#444444", lw=0.75, ls="--")
    ax.set_title(dataset_labels[dataset], fontweight="bold")
    ax.set_xlabel("RMSE change from four-block reference (%)")
    ax.grid(axis="x", color="#E5E9EC", lw=0.5)
axes[0].set_yticks(range(len(configurations)), [configuration_labels[item] for item in configurations])
axes[1].tick_params(labelleft=False)
axes[0].invert_yaxis()
for extension in ("pdf", "svg", "png", "tiff"):
    fig.savefig(OUT_FIGURE / f"ablation_epoch300.{extension}", dpi=600 if extension == "tiff" else 300, bbox_inches="tight", facecolor="white")
plt.close(fig)

slash = chr(92)
row_end = slash * 2
lines = [
    r"\begin{table*}[htbp]",
    r"\centering",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{3.0pt}",
    r"\caption{Five-seed module, depth and loss analysis with a maximum of 300 epochs per run and validation-MSE checkpoint selection. External regression-kriging and local-predictor mixtures are disabled throughout. The experiments retain the Mat\'ern-$3/2$ base covariance used during module development; the subsequent kernel comparison is reported in Table~\ref{tab:kernel-ablation}. Module removals use the common four-block reference, whereas the six-block row gives the selected network depth. Values are test RMSE (mean $\pm$ s.d.); lower is better. Best values are bold and second-best values are underlined.}",
    r"\label{tab:ablation-modules}",
    r"\begin{tabular}{lcc}",
    r"\toprule",
    "Configuration & " + " & ".join(dataset_labels[dataset] for dataset in datasets) + " " + row_end,
    r"\midrule",
]
for configuration in configurations:
    cells = []
    for dataset in datasets:
        row = summary[(summary.dataset.eq(dataset)) & (summary.configuration.eq(configuration))].iloc[0]
        cell = f"{row.rmse_mean:.3f} $\\pm$ {row.rmse_std:.3f}"
        ranking = summary[summary.dataset.eq(dataset)].sort_values("rmse_mean").configuration.tolist()
        if configuration == ranking[0]:
            cell = r"\textbf{" + cell + "}"
        elif configuration == ranking[1]:
            cell = r"\underline{" + cell + "}"
        cells.append(cell)
    lines.append(configuration_labels[configuration] + " & " + " & ".join(cells) + " " + row_end)
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(args.output_dir / "generated_ablation_epoch300.tex").write_text("\n".join(lines), encoding="utf-8")

print(summary[["dataset", "configuration", "rmse_mean", "rmse_std"]].to_string(index=False))


