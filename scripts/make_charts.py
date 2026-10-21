#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Render the README charts from docs/data/bench.csv.

    uv run scripts/make_charts.py

Every drawn number is a CSV row; nothing is interpolated beyond joining measured points.
Output: docs/img/<name>-light.svg and <name>-dark.svg (the README picks one through <picture>) and docs/img/<name>.png
(light, a fallback). Type is Inter from docs/fonts/ (SIL OFL 1.1), drawn as outlines.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV, IMG, FONTS = ROOT / "docs/data/bench.csv", ROOT / "docs/img", ROOT / "docs/fonts"
for f in sorted(FONTS.glob("Inter-*.ttf")):
    font_manager.fontManager.addfont(str(f))
FAMILY = "Inter" if any(FONTS.glob("Inter-*.ttf")) else "DejaVu Sans"

# Same palette as my other recipe repos: colour-blind-checked categorical slots, one step per theme.
THEMES = {
    "light": dict(ink="#1f2328", muted="#59636e", grid="#e8ebef", axis="#c8d1da",
                  series=["#eb6834", "#2a78d6", "#1baf7a", "#818b98"]),
    "dark": dict(ink="#e6edf3", muted="#9198a1", grid="#21272f", axis="#3d444d",
                 series=["#d95926", "#3987e5", "#199e70", "#6e7681"]),
}
T = THEMES["light"]


def theme(mode):
    global T
    T = THEMES[mode]
    plt.rcParams.update({
        "font.family": FAMILY, "font.size": 12, "text.color": T["ink"], "axes.labelcolor": T["muted"],
        "xtick.color": T["muted"], "ytick.color": T["muted"], "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
        "axes.edgecolor": T["axis"], "svg.fonttype": "path", "svg.hashsalt": "make_charts",
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.labelsize": 11.5,
    })


def load():
    out = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(CSV.open()):
        out[r["chart"]][r["series"]].append((int(r["x"]), float(r["value"]), r))
    return out


def axes(w=5.0, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.grid(axis="y", color=T["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    return fig, ax


def lines(ax, series, xlabel, ylabel, xticks, dashed=(), legend="upper left"):
    ends = []
    for i, (name, pts) in enumerate(series.items()):
        xs, ys = zip(*[(x, v) for x, v, _ in sorted(pts)])
        c = T["series"][i % len(T["series"])]
        ax.plot(xs, ys, color=c, lw=2.2, marker="o", ms=5, ls="--" if name in dashed else "-", label=name)
        if all(abs(ys[-1] - e) > 0.04 * ys[-1] for e in ends):   # one label for end points that sit together
            ax.annotate(f"{ys[-1]:,.0f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=11, color=T["ink"])
        ends.append(ys[-1])
    ax.set_xticks(list(xticks))
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.legend(loc=legend, fontsize=10.5, labelcolor=T["ink"])


def decode(data):
    fig, ax = axes()
    lines(ax, data["decode"], "concurrent requests", "aggregate decode tok/s", [1, 2, 4, 8])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlim(0.85, 11)
    return fig


def prefill(data):
    fig, ax = axes()
    s = data["prefill"]
    lines(ax, s, "prompt length (K tokens)", "prefill tok/s", [4, 32, 128],
          dashed=[k for k in s if "C4" in k], legend="lower right")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 32, 128])
    ax.set_xticklabels(["4K", "32K", "128K"])
    ax.minorticks_off()
    ax.set_xlim(3.4, 190)
    return fig


def history(data):
    s = data["history"]
    fig, axs = plt.subplots(1, len(s), figsize=(7.2, 2.9))
    for ax, (name, pts) in zip(axs, s.items()):
        dates = [r["date"][5:] for _, _, r in pts]
        vals = [v for _, v, _ in pts]
        cols = [T["series"][3]] * (len(vals) - 1) + [T["series"][0]]   # older dates grey, the newest in colour
        ax.bar(range(len(vals)), vals, color=cols, width=0.62)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:,.1f}" if v < 100 else f"{v:,.0f}", (i, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=11, color=T["ink"])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(dates)
        ax.set_title(name, fontsize=11.5, color=T["ink"])
        ax.set_yticks([])
        ax.set_ylim(0, max(vals) * 1.2)
        ax.spines["bottom"].set_color(T["axis"])
    fig.tight_layout()
    return fig


CHARTS = {"decode": decode, "prefill": prefill, "history": history}


def main():
    data = load()
    IMG.mkdir(parents=True, exist_ok=True)
    for name, draw in CHARTS.items():
        for mode in ("light", "dark"):
            theme(mode)
            fig = draw(data)
            fig.savefig(IMG / f"{name}-{mode}.svg", bbox_inches="tight", metadata={"Date": None})
            if mode == "light":
                fig.savefig(IMG / f"{name}.png", dpi=160, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        print("wrote", name)


if __name__ == "__main__":
    main()
