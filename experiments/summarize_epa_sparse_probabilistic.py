from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "epa_sparse_fraction_fixedtest"
FRACTIONS = [10, 20, 30, 40, 50, 60]
METHODS = ["SimpleKriging", "RegressionKriging", "RFKriging", "ASDKL"]
METRICS = ["nll", "crps", "mpiw90", "picp90"]
rows = []
for fraction in FRACTIONS:
    for path in (RESULTS / f"fraction_{fraction}" / "probabilistic_baselines").glob("*.json"):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
baseline = pd.DataFrame(rows)
asdkl = pd.read_csv(RESULTS / "asdkl_fraction_results.csv")
asdkl = asdkl[asdkl.training_fraction_pct.isin(FRACTIONS)].copy()
asdkl["method"] = "ASDKL"
columns = ["dataset", "training_fraction_pct", "seed", "method", *METRICS]
complete = pd.concat([baseline[columns], asdkl[columns]], ignore_index=True)
assert len(complete) == len(FRACTIONS) * len(METHODS) * 5
assert complete.groupby(["training_fraction_pct", "method"]).size().eq(5).all()
complete.to_csv(RESULTS / "probabilistic_comparison_10_60_complete.csv", index=False)
summary = complete.groupby(["training_fraction_pct", "method"])[METRICS].agg(["mean", "std"])
summary.to_csv(RESULTS / "probabilistic_comparison_10_60_summary.csv")

def decorate(text, rank):
    return rf"\textbf{{{text}}}" if rank == 0 else (rf"\underline{{{text}}}" if rank == 1 else text)

def ranks(block, metric):
    values = block[(metric, "mean")]
    score = (values - 0.90).abs() if metric == "picp90" else values
    order = list(score.sort_values().index)
    return {method: order.index(method) for method in order}

lines = [
    r"\begin{table*}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
    r"\caption{Probabilistic prediction under reduced EPA PM$_{2.5}$ observation density. Coordinate groups were assigned to the same fixed validation and test partitions used in the learning-curve experiment. All methods used exact residual-GP inference; predictive variances were scaled using validation residuals, and test responses were used only for final evaluation. Values are mean $\pm$ s.d. over five seeds. Bold and underlined values are the best and second-ranked unrounded means within each fraction; PICP$_{90}$ is ranked by absolute deviation from 0.90. Ranking does not imply statistical separation.}",
    r"\label{tab:epa-sparse-probabilistic}", r"\begin{tabular}{clcccc}", r"\toprule",
    r"Training & Method & NLL $\downarrow$ & CRPS $\downarrow$ & MPIW$_{90}$ $\downarrow$ & PICP$_{90}$ $\rightarrow 0.90$ \\", r"\midrule"
]
labels = {"ASDKL": "ASDKL (ours)"}
for fi, fraction in enumerate(FRACTIONS):
    block = summary.loc[fraction].reindex(METHODS)
    metric_ranks = {metric: ranks(block, metric) for metric in METRICS}
    for mi, method in enumerate(METHODS):
        prefix = rf"\multirow{{4}}{{*}}{{{fraction}\%}}" if mi == 0 else ""
        cells = []
        for metric in METRICS:
            value = f"{block.loc[method, (metric, 'mean')]:.3f} $\\pm$ {block.loc[method, (metric, 'std')]:.3f}"
            cells.append(decorate(value, metric_ranks[metric][method]))
        lines.append(f"{prefix} & {labels.get(method, method)} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule" if fi < len(FRACTIONS)-1 else r"\bottomrule")
lines += [r"\end{tabular}", r"\end{table*}"]
(ROOT / "results" / "generated_epa_sparse_probabilistic_table.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
print(summary.to_string())
