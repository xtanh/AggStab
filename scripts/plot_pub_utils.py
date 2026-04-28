from __future__ import annotations

import matplotlib as mpl


def configure_matplotlib() -> None:
    mpl.use("Agg")
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["axes.titlesize"] = 13
    mpl.rcParams["axes.linewidth"] = 1.0
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4b5563")
    ax.spines["bottom"].set_color("#4b5563")
    ax.tick_params(colors="#374151", width=0.8, length=4)
    ax.grid(alpha=0.16, linewidth=0.6, color="#9ca3af")


def add_stats_box(ax, text: str, x: float = 0.03, y: float = 0.97, ha: str = "left") -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va="top",
        ha=ha,
        fontsize=10.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#d1d5db",
            linewidth=0.9,
            alpha=0.96,
        ),
    )
