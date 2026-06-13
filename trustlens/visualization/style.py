from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, field
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt

BRAND_COLORS = {
    "trustlens_blue": "#0052cc",
    "trustlens_green": "#00b300",
    "trustlens_orange": "#ff6600",
    "trustlens_red": "#cc0000",
    "trustlens_gray": "#666666",
    "trustlens_light_gray": "#999999",
    "trustlens_pale_gray": "#cccccc",
}

TYPOGRAPHY = {
    "font_family": "DejaVu Sans",
    "title_size": 13,
    "label_size": 11,
    "tick_size": 10,
    "annotation_size": 10,
}

GRID = {
    "alpha": 0.4,
    "alpha_minor": 0.2,
    "linewidth": 0.8,
}

FIG_DEFAULTS = {
    "figsize": (8, 6),
    "facecolor": "#FFFFFF",
    "dpi": 100,
    "savefig_dpi": 150,
}

SPACING = {
    "bbox_pad": 0.4,
    "bbox_alpha": 0.8,
    "bar_edge_width": 1.0,
    "bar_alpha": 0.85,
}

PALETTE = [
    BRAND_COLORS["trustlens_blue"],
    BRAND_COLORS["trustlens_orange"],
    BRAND_COLORS["trustlens_green"],
    BRAND_COLORS["trustlens_red"],
    BRAND_COLORS["trustlens_gray"],
    BRAND_COLORS["trustlens_light_gray"],
    BRAND_COLORS["trustlens_pale_gray"],
]

COLORBLIND_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
]

SEMANTIC_COLORS = {
    "severity": {
        "acceptable": "#00b300",
        "moderate": "#ff6600",
        "severe": "#cc0000",
        "unknown": "#999999",
    },
    "verdict": {
        "deploy": "#0052cc",
        "caution": "#ff6600",
        "do_not_deploy": "#cc0000",
    },
    "grade": {
        "A": "#0052cc",
        "B": "#00b300",
        "C": "#ff6600",
        "D": "#cc0000",
    },
}

COLORBLIND_SEMANTIC_COLORS = {
    "severity": {
        "acceptable": "#0072B2",
        "moderate": "#E69F00",
        "severe": "#D55E00",
        "unknown": "#999999",
    }
}


@dataclass(frozen=True)
class Theme:
    name: str = "default"
    palette: list[str] = field(default_factory=lambda: list(PALETTE))
    semantic: dict[str, dict[str, str]] = field(
        default_factory=lambda: deepcopy(SEMANTIC_COLORS)
    )
    fig_defaults: dict[str, object] = field(
        default_factory=lambda: deepcopy(FIG_DEFAULTS)
    )
    typography: dict[str, object] = field(
        default_factory=lambda: deepcopy(TYPOGRAPHY)
    )
    grid: dict[str, object] = field(default_factory=lambda: deepcopy(GRID))
    spacing: dict[str, object] = field(default_factory=lambda: deepcopy(SPACING))
    base_style: str | None = None

    def __repr__(self) -> str:
        return f"<Theme {self.name}>"


DEFAULT_THEME = Theme()

COLORBLIND_THEME = Theme(
    name="colorblind",
    palette=list(COLORBLIND_PALETTE),
    semantic=deepcopy(COLORBLIND_SEMANTIC_COLORS),
)


@contextmanager
def apply_style(theme: Theme = DEFAULT_THEME):
    original_rc = mpl.rcParams.copy()
    try:
        if theme.base_style is not None:
            try:
                mpl.style.use(theme.base_style)
            except Exception:
                warnings.warn(
                    f"Style {theme.base_style!r} could not be loaded",
                    UserWarning,
                )

        mpl.rcParams["font.family"] = [theme.typography["font_family"]]
        mpl.rcParams["font.size"] = theme.typography["label_size"]
        yield theme
    finally:
        mpl.rcParams.clear()
        mpl.rcParams.update(original_rc)


def styled_figure(
    theme: Theme = DEFAULT_THEME,
    figsize: tuple[float, float] | None = None,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs,
):
    if figsize is None:
        figsize = tuple(theme.fig_defaults["figsize"])

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        dpi=theme.fig_defaults["dpi"],
        facecolor=theme.fig_defaults["facecolor"],
        **kwargs,
    )

    if isinstance(axes, Iterable):
        for ax in axes.flat:
            ax.set_facecolor(theme.fig_defaults["facecolor"])
    else:
        axes.set_facecolor(theme.fig_defaults["facecolor"])

    return fig, axes


def get_categorical_colors(n: int, theme: Theme = DEFAULT_THEME) -> list[str]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if len(theme.palette) == 0:
        raise ValueError("theme.palette must be non-empty")
    if n == 0:
        return []
    if n <= len(theme.palette):
        return theme.palette[:n]
    times = (n // len(theme.palette)) + 1
    return (theme.palette * times)[:n]


__all__ = [
    "BRAND_COLORS",
    "PALETTE",
    "SEMANTIC_COLORS",
    "TYPOGRAPHY",
    "GRID",
    "FIG_DEFAULTS",
    "SPACING",
    "Theme",
    "DEFAULT_THEME",
    "COLORBLIND_THEME",
    "apply_style",
    "styled_figure",
    "get_categorical_colors",
]
