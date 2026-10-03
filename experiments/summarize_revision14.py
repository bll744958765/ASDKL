from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
STRUCT = ROOT / "results" / "revision14_shared_relation"
KERNEL = ROOT / "results" / "revision14_kernel_variants"
SENS = ROOT / "results" / "revision14_sensitivity"
OUT = ROOT / "results" / "revision14_summary"
FIG = ROOT / "results" / "generated_figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

METRICS = ["rmse", "mae", "r2", "nll", "crps", "picp90", "mpiw90", "val_rmse"]
SEEDS = [1042, 1052, 1062, 1072, 1082]
DATASETS = ["synthetic", "pm25", "lucas_large", "california"]


def load_runs(root: Path, groups: list[str], datasets: list[str]) -> pd.DataFrame:
    rows = []
    for group in groups:
        for dataset in datasets:
            for seed in SEEDS:
                path = root / group / dataset / f"seed_{seed}" / f"{dataset}_run.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                item = json.loads(path.read_text(encoding="utf-8"))
                rows.append({"group": group, "dataset": dataset, "seed": seed,
                             **{m: float(item[m]) for m in METRICS}})
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = frame.groupby(keys)[METRICS].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(c).rstrip("_") for c in out.columns]
    return out


main = load_runs(STRUCT, ["six_block_reference"], DATASETS)
main.to_csv(OUT / "main_source.csv", index=False)
aggregate(main, ["dataset"]).to_csv(OUT / "main_summary.csv", index=False)

struct_groups = ["six_block_reference", "independent_frequencies", "no_relation_encoder"]
struct = load_runs(STRUCT, struct_groups, DATASETS)
struct.to_csv(OUT / "structural_ablation_source.csv", index=False)
aggregate(struct, ["dataset", "group"]).to_csv(OUT / "structural_ablation_summary.csv", index=False)

module_groups = [
    "six_block_reference", "independent_frequencies", "no_relation_encoder",
    "no_context", "no_fourier", "no_spectral_regularization",
    "no_spectral_attention", "no_gp", "no_residual_connections",
    "three_residual_blocks", "auxiliary_mse",
]
module = load_runs(STRUCT, module_groups, ["synthetic", "pm25"])
module.to_csv(OUT / "module_ablation_source.csv", index=False)
aggregate(module, ["dataset", "group"]).to_csv(OUT / "module_ablation_summary.csv", index=False)

kernel_groups = ["representation_rbf_plus_matern32", "representation_rbf_only", "matern32_only"]
kernels = load_runs(KERNEL, kernel_groups, DATASETS)
selected = main.copy()
selected["group"] = "representation_rbf_plus_base_rbf"
kernels = pd.concat([kernels, selected], ignore_index=True)
kernels.to_csv(OUT / "kernel_source.csv", index=False)
aggregate(kernels, ["dataset", "group"]).to_csv(OUT / "kernel_summary.csv", index=False)

# Add the already completed default main run to the sensitivity source.
sens = pd.read_csv(SENS / "sensitivity_results.csv")
sens = sens[sens.status.eq("ok")].copy()
defaults = main[main.dataset.eq("synthetic")].copy()
for key, value in {"family": "default", "value": "default", "neighbors": 64,
                   "freq_dim": 64, "hidden_dim": 64, "lambda_spectral": .1,
                   "lr": .003, "epochs": 300}.items():
    defaults[key] = value
sens = pd.concat([sens, defaults], ignore_index=True, sort=False)
sens.to_csv(OUT / "sensitivity_source_with_default.csv", index=False)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.2, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": .75,
})
seed_colors = dict(zip(SEEDS, ["#4878A8", "#6F9E75", "#D19A4A", "#9B77B4", "#C66B64"]))

# Figure 6: paired seed-level percentage changes relative to complete ASDKL.
labels = {
    "six_block_reference": "Complete ASDKL",
    "independent_frequencies": "Independent neighbour frequencies",
    "no_relation_encoder": "Without relative-relation encoder",
    "no_context": "Without local context",
    "no_fourier": "Without adaptive Fourier features",
    "no_spectral_regularization": "Without spectral regularization",
    "no_spectral_attention": "Without spectral attention",
    "no_gp": "Without residual GP",
    "no_residual_connections": "Without residual connections",
    "three_residual_blocks": "Three residual blocks",
    "auxiliary_mse": "With auxiliary MSE",
}
ref = module[module.group.eq("six_block_reference")][["dataset", "seed", "rmse"]].rename(columns={"rmse": "ref_rmse"})
paired = module.merge(ref, on=["dataset", "seed"])
paired["delta"] = 100 * (paired.rmse / paired.ref_rmse - 1)
paired.to_csv(OUT / "figure6_source.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.9), sharey=True, constrained_layout=False)
fig.subplots_adjust(left=.31, right=.98, top=.92, bottom=.18, wspace=.08)
order = module_groups
for ax, dataset, title in zip(axes, ["synthetic", "pm25"], ["Nonstationary synthetic", r"EPA PM$_{2.5}$"]):
    sub = paired[paired.dataset.eq(dataset)]
    for y, group in enumerate(order):
        vals = sub[sub.group.eq(group)].sort_values("seed")
        for _, row in vals.iterrows():
            ax.scatter(row.delta, y, s=17, color=seed_colors[int(row.seed)], alpha=.82,
                       edgecolor="white", linewidth=.3, zorder=3)
        mean, sd = vals.delta.mean(), vals.delta.std(ddof=1)
        ax.errorbar(mean, y, xerr=sd, fmt="D", ms=4, color="#263238",
                    ecolor="#59666B", elinewidth=.8, capsize=2, zorder=4)
    ax.axvline(0, color="#555", lw=.8, ls="--")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Test RMSE change from complete ASDKL (%)")
    ax.grid(axis="x", color="#E4E8EA", lw=.5)
axes[0].set_yticks(range(len(order)), [labels[g] for g in order])
axes[0].invert_yaxis()
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=str(s), ms=4) for s, c in seed_colors.items()]
fig.legend(handles=handles, title="Seed", ncol=5, loc="lower center", bbox_to_anchor=(.56, .015))
for ext in ["pdf", "svg", "png", "tiff"]:
    fig.savefig(FIG / f"ablation_epoch300_bar.{ext}", dpi=600 if ext == "tiff" else 300,
                bbox_inches="tight", facecolor="white")
plt.close(fig)

# Figure 7: validation-only sensitivity, five seeds, default inserted per family.
families = ["neighbors", "freq_dim", "hidden_dim", "lambda_spectral", "lr", "epochs"]
parts = []
default = sens[sens.family.eq("default")][["dataset", "seed", "val_rmse"]].rename(columns={"val_rmse": "default_val_rmse"})
for family in families:
    varied = sens[sens.family.eq(family)].copy()
    shared = sens[sens.family.eq("default")].copy()
    shared["family"] = family
    shared["value"] = shared[family]
    parts.append(pd.concat([varied, shared], ignore_index=True))
ss = pd.concat(parts, ignore_index=True).merge(default, on=["dataset", "seed"])
ss["value"] = pd.to_numeric(ss.value)
ss["relative"] = ss.val_rmse / ss.default_val_rmse
ss.to_csv(OUT / "figure7_source.csv", index=False)
summ = ss.groupby(["family", "value"], as_index=False).agg(mean=("relative", "mean"), sd=("relative", "std"), n=("seed", "count"))
summ.to_csv(OUT / "sensitivity_summary.csv", index=False)
titles = {"neighbors": "Spatial neighbours $K$", "freq_dim": "Frequency dimension $M$",
          "hidden_dim": "Representation width", "lambda_spectral": "Spectral weight $\\lambda_s$",
          "lr": "Learning rate", "epochs": "Training epochs"}
crit = t.ppf(.975, 4)
ci_all = crit * summ.sd / np.sqrt(summ.n)
ymin = float((summ["mean"] - ci_all).min() - .02)
ymax = float((summ["mean"] + ci_all).max() + .02)
fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.9), constrained_layout=True)
for i, (ax, family) in enumerate(zip(axes.flat, families)):
    g = summ[summ.family.eq(family)].sort_values("value")
    x = np.arange(len(g)); y = g["mean"].to_numpy(); ci = crit * g.sd.to_numpy() / np.sqrt(g.n.to_numpy())
    ax.fill_between(x, y-ci, y+ci, color="#4878A8", alpha=.16, lw=0)
    ax.plot(x, y, "o-", color="#4878A8", lw=1.15, ms=3.7)
    seed_sub = ss[ss.family.eq(family)]
    for j, value in enumerate(g.value):
        q = seed_sub[seed_sub.value.eq(value)]
        for _, row in q.iterrows():
            ax.scatter(j, row.relative, s=10, color=seed_colors[int(row.seed)], alpha=.62, zorder=3)
    ax.axhline(1, color="#555", lw=.75, ls="--")
    ax.set_xticks(x, [f"{v:g}" for v in g.value])
    ax.set_ylim(ymin, ymax); ax.set_title(titles[family]); ax.set_xlabel("Value")
    ax.set_ylabel("Validation RMSE / default"); ax.grid(axis="y", color="#E4E8EA", lw=.5)
    ax.text(-.13, 1.07, chr(97+i), transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")
for ext in ["pdf", "svg", "png", "tiff"]:
    fig.savefig(FIG / f"hyperparameter_sensitivity_synthetic_epoch300.{ext}",
                dpi=600 if ext == "tiff" else 300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(aggregate(main, ["dataset"]).to_string(index=False))
print(aggregate(struct, ["dataset", "group"])[["dataset", "group", "rmse_mean", "rmse_std"]].to_string(index=False))
