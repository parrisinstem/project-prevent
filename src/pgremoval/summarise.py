"""Summarise metrics.csv into a per-attack table (mean ± std across images).

    python -m pgremoval.summarise results/metrics.csv

Prints a Markdown table you can paste into WRITEUP.md. Plotting is left to you
(task 5): the table tells you what to plot.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

KEY_COLUMNS = [
    "latent_rel_dist",
    "edit_vs_attacked_clean_psnr",
    "edit_vs_attacked_clean_ssim",
    "edit_vs_attacked_clean_lpips",
    "image_cost_psnr",
    "image_cost_ssim",
    "image_cost_lpips",
]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEY_COLUMNS if c in df.columns]
    clean = df[cols].replace([np.inf, -np.inf], np.nan)
    grouped = clean.groupby(df["attack"])
    mean, std = grouped.mean(), grouped.std()
    table = mean.copy().astype(object)
    for c in cols:
        table[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c].fillna(0))]
    table.insert(0, "n_images", grouped.size())
    order = ["none"] + sorted(a for a in table.index if a != "none")
    return table.reindex([a for a in order if a in table.index])


def to_markdown(table: pd.DataFrame) -> str:
    headers = ["attack"] + list(table.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for attack, row in table.iterrows():
        lines.append("| " + " | ".join([str(attack)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    path = argv[0] if argv else "results/metrics.csv"
    df = pd.read_csv(path)
    if df.get("safety_flagged", pd.Series(dtype=bool)).any():
        n = int(df["safety_flagged"].sum())
        print(f"Note: {n} rows had an output flagged by the safety checker. Inspect those images.\n")
    print(to_markdown(summarise(df)))


if __name__ == "__main__":
    main()
