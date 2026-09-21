# coding: utf-8
"""One plotting style for every closeout figure.

Figures are written as PDF (vector, for the paper) and PNG (for looking at),
same name, same directory.  The palette is colour-blind safe and every role keeps
one colour across all four figures, so a reader who learns "raw is orange" in
Table 1 does not have to relearn it in Figure 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

#: Role -> colour.  C (entity-constraint-cleaned) is the reference blue, K
#: (raw) the contrasting orange, O (controlled replacement) purple, and the
#: no-SFT reference grey.
ROLE_COLORS = {
    "C": "#3E6FB0",
    "K": "#D2694A",
    "O": "#7B5EA7",
    "reference": "#8A8A8A",
    "control": "#4F9D69",
    "type": "#B08A3E",
    "random": "#B0B0B0",
    "bias": "#5E5E5E",
}

DOSE_COLORS = ("#3E6FB0", "#6E8FC4", "#A87EBB", "#7B5EA7", "#5A4680")


def pyplot():
    """Import matplotlib configured for headless, deterministic output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "lines.linewidth": 1.4,
        "errorbar.capsize": 2.5,
        # A vector figure whose text is outlines cannot be searched or fixed
        # in the camera-ready; keep the glyphs as glyphs.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def save(fig, stem: str | Path) -> list[Path]:
    """Write `<stem>.pdf` and `<stem>.png`, returning what was written."""
    target = Path(stem)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png"):
        path = target.with_suffix(suffix)
        fig.savefig(path)
        written.append(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return written


def bar_with_ci(ax, labels: Sequence[str], values: Sequence[float],
                lows: Sequence[float], highs: Sequence[float],
                colors: Sequence[str], *, ylabel: str = "",
                percent: bool = False, rotation: int = 20) -> None:
    """Point estimates with their paired intervals, never bare bars.

    A rate without its interval invites a reader to compare two numbers that
    the data cannot separate, so the error bar is not optional here.
    """
    import numpy as np

    scale = 100.0 if percent else 1.0
    x = np.arange(len(labels))
    v = np.asarray(values, dtype=float) * scale
    lo = np.asarray(lows, dtype=float) * scale
    hi = np.asarray(highs, dtype=float) * scale
    err = np.vstack([np.maximum(v - lo, 0.0), np.maximum(hi - v, 0.0)])
    err = np.where(np.isfinite(err), err, 0.0)
    ax.bar(x, v, color=list(colors), width=0.66)
    ax.errorbar(x, v, yerr=err, fmt="none", ecolor="#333333", elinewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation,
                       ha="right" if rotation else "center")
    if ylabel:
        ax.set_ylabel(ylabel + (" (%)" if percent else ""))


def diff_panel(ax, labels: Sequence[str], diffs: Sequence[float],
               lows: Sequence[float], highs: Sequence[float],
               *, xlabel: str = "", percent: bool = False,
               color: str = "#3E6FB0") -> None:
    """Horizontal paired differences with a zero reference line."""
    import numpy as np

    scale = 100.0 if percent else 1.0
    y = np.arange(len(labels))
    d = np.asarray(diffs, dtype=float) * scale
    lo = np.asarray(lows, dtype=float) * scale
    hi = np.asarray(highs, dtype=float) * scale
    err = np.vstack([np.maximum(d - lo, 0.0), np.maximum(hi - d, 0.0)])
    err = np.where(np.isfinite(err), err, 0.0)
    ax.axvline(0.0, color="#666666", linewidth=0.9, linestyle="--")
    ax.errorbar(d, y, xerr=err, fmt="o", color=color, markersize=4,
                elinewidth=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    if xlabel:
        ax.set_xlabel(xlabel + (" (pp)" if percent else ""))


def note(fig, text: str) -> None:
    """A one-line caption inside the figure, for what the axes cannot say."""
    fig.text(0.005, -0.02, text, fontsize=6.8, color="#444444", va="top")


def color_for(role: str, fallback: str = "#3E6FB0") -> str:
    return ROLE_COLORS.get(role, fallback)


def series_colors(roles: Sequence[str]) -> list[str]:
    return [color_for(role) for role in roles]


def annotate_missing(ax, message: str = "not yet measured") -> None:
    """An empty panel says so; it never gets a zero or a predicted shape."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color="#888888")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def value_or_nan(row: Any, field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")
