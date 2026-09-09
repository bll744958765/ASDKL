# Figure code

This directory contains plotting code only. It does not distribute manuscript result tables, experiment-derived source CSVs, or rendered result figures.

`figure.py` accepts a figure name and an output directory. Its quantitative panels expect source CSVs produced by the experiment and summarization scripts. Put those generated files in `figure/data/` locally, then run:

```bash
python figure/figure.py --figure all
```

The generator-only image embedded in the top-level README is created independently with:

```bash
python examples/generate_synthetic_example.py
```

Generated files under `figure/data/` and `figure/output/` are ignored by Git.
