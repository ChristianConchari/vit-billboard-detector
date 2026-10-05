"""Shared look for every chart in the project, so figures read as one system.

Each model or configuration keeps one color across all charts: RT-DETR (and
the frozen backbone) blue, Grounding DINO orange, the unfrozen backbone aqua.
The palette was checked for color-vision deficiency and contrast; aqua sits
below 3:1 on the surface, so charts using it carry visible labels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

RTDETR_COLOR = "#2a78d6"
ZERO_SHOT_COLOR = "#eb6834"
UNFROZEN_COLOR = "#1baf7a"
SURFACE_COLOR = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def new_figure(width: float = 8, height: float = 4.5) -> tuple[Figure, Axes]:
    figure, axes = plt.subplots(figsize=(width, height), dpi=150, facecolor=SURFACE_COLOR)
    axes.set_facecolor(SURFACE_COLOR)
    return figure, axes


def style_axes(axes: Axes, title: str, xlabel: str, ylabel: str, grid_axis: str = "y") -> None:
    axes.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12, pad=16)
    axes.set_xlabel(xlabel, color=TEXT_SECONDARY)
    axes.set_ylabel(ylabel, color=TEXT_SECONDARY)
    axes.grid(axis=grid_axis, color=GRID_COLOR, linewidth=1)
    axes.set_axisbelow(True)
    axes.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID_COLOR)


def add_legend(axes: Axes, location: str = "lower right") -> None:
    """Legend whose line samples drop the markers' surface ring, which would make solid
    lines look dashed at legend size."""
    handles, labels = axes.get_legend_handles_labels()
    handles = [_ringless(handle) if isinstance(handle, Line2D) else handle for handle in handles]
    axes.legend(handles, labels, loc=location, frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)


def _ringless(line: Line2D) -> Line2D:
    return Line2D(
        [],
        [],
        color=line.get_color(),
        linewidth=line.get_linewidth(),
        linestyle=line.get_linestyle(),
        marker=line.get_marker(),
        markersize=line.get_markersize(),
        markeredgewidth=0,
    )


def save(figure: Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, facecolor=SURFACE_COLOR)
    plt.close(figure)
