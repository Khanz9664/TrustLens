"""
trustlens.visualization.style.
==============================
Centralized styling for all TrustLens visualization modules.

Architecture & Centralized Styling
----------------------------------
TrustLens uses a unified styling architecture to ensure visual parity across
all plots (calibration, bias, failure, representation). Centralized styling exists
to prevent fragmented UI experiences and to enforce accessibility standards,
making it trivial to adapt to dark mode or colorblind themes in the future.

How Visual Parity is Maintained
-------------------------------
Visual parity is maintained by prohibiting hard-coded colors, fonts, or grids
in individual plotting functions. Instead, all plots must rely on:
1. `Theme` properties for resolving colors and typography.
2. `styled_figure` for instantiating axes with consistent dimensions and backgrounds.
3. `apply_style` for applying global Matplotlib configurations temporarily.

Implementing New Visualizations
-------------------------------
When building a new plot in `trustlens/visualization/`:
1. **Never mutate `matplotlib.rcParams` globally**. Use `with apply_style() as theme:`.
2. **Never hard-code colors**. Use `theme.brand` or `theme.semantic`.
3. **Always use semantic mappings** (`SEMANTIC_COLORS`) when representing statuses like 'pass' or 'fail'.
4. **Use `get_categorical_colors`** when plotting multiple arbitrary categories.

Constants
---------
* `BRAND_COLORS`: The core TrustLens color palette. Used for structural elements, titles, and fills.
* `SEMANTIC_COLORS`: Meaning-bearing colors mapping specific logic (severity, verdicts, grades) to a visual representation. These must be used for all diagnostic outputs.

It also exposes:
* :func:`apply_style` — a context manager that scopes ``rcParams`` mutations to
  a ``with`` block, so the library never permanently mutates user matplotlib state.
* :func:`styled_figure` — a thin helper around :func:`matplotlib.pyplot.subplots`.
* :func:`get_categorical_colors` — return ``n`` colors from the categorical palette.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Brand colors — name lookup for non-categorical use (titles, fills, etc.)
# ---------------------------------------------------------------------------

BRAND_COLORS: dict[str, str] = {
    "blue": "#4B8BF5",
    "orange": "#F5784B",
    "green": "#34C759",
    "red": "#FF3B30",
    "amber": "#FF9F0A",
    "purple": "#AF52DE",
    "pink": "#FF2D55",
    "cyan": "#5AC8FA",
    "deep_orange": "#FF6B35",
    "gray": "#8E8E93",
    "light_gray": "#CCCCCC",
    "muted_gray": "#AAAAAA",
    "text_dark": "#444444",
    "text_muted": "#666666",
    "text_subtle": "#888888",
    "light": "#F2F2F7",
    "white": "#FFFFFF",
    "dark": "#1C1C1E",
}

# ---------------------------------------------------------------------------
# Categorical palette — ordered list for plots with N groups/classes
# ---------------------------------------------------------------------------

PALETTE: list[str] = [
    BRAND_COLORS["blue"],
    BRAND_COLORS["orange"],
    BRAND_COLORS["green"],
    BRAND_COLORS["purple"],
    BRAND_COLORS["amber"],
    BRAND_COLORS["pink"],
    BRAND_COLORS["cyan"],
    BRAND_COLORS["deep_orange"],
]

# ---------------------------------------------------------------------------
# Semantic colors — meaning-bearing colors that should never be reused
# arbitrarily. Severity is diagnostic; verdict is operational.
# ---------------------------------------------------------------------------

SEMANTIC_COLORS: dict[str, dict[str, str]] = {
    "severity": {
        "acceptable": BRAND_COLORS["green"],
        "moderate": BRAND_COLORS["amber"],
        "severe": BRAND_COLORS["pink"],
        "unknown": BRAND_COLORS["light_gray"],
    },
    "verdict": {
        "deploy": BRAND_COLORS["green"],
        "caution": BRAND_COLORS["amber"],
        "do_not_deploy": BRAND_COLORS["red"],
    },
    "grade": {
        "A": BRAND_COLORS["green"],
        "B": BRAND_COLORS["blue"],
        "C": BRAND_COLORS["amber"],
        "D": BRAND_COLORS["red"],
    },
    "direction": {
        "positive": BRAND_COLORS["blue"],
        "negative": BRAND_COLORS["orange"],
    },
    "neutral": {
        "reference": BRAND_COLORS["muted_gray"],
        "edge": BRAND_COLORS["white"],
        "annotation_edge": BRAND_COLORS["light_gray"],
        "annotation_face": BRAND_COLORS["white"],
    },
}

# ---------------------------------------------------------------------------
# Typography / Grid / Figure defaults / Spacing
# ---------------------------------------------------------------------------

TYPOGRAPHY: dict[str, Any] = {
    "font_family": "DejaVu Sans",
    "title_size": 13,
    "label_size": 11,
    "tick_size": 10,
    "annotation_size": 9,
    "title_weight": "bold",
}

GRID: dict[str, Any] = {
    "alpha": 0.4,
    "alpha_minor": 0.3,
    "linewidth": 0.8,
}

FIG_DEFAULTS: dict[str, Any] = {
    "figsize": (8, 6),
    "facecolor": BRAND_COLORS["white"],
    "dpi": 100,
    "savefig_dpi": 150,
}

SPACING: dict[str, Any] = {
    "bbox_pad": 0.4,
    "bbox_alpha": 0.9,
    "bar_edge_width": 1.2,
    "bar_alpha": 0.85,
}

COLORBLIND_PALETTE: list[str] = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
]

COLORBLIND_SEMANTIC_COLORS: dict[str, dict[str, str]] = {
    "severity": {
        "acceptable": "#0072B2",
        "moderate": "#E69F00",
        "severe": "#D55E00",
        "unknown": "#999999",
    }
}

# ---------------------------------------------------------------------------
# Theme — frozen dataclass bundling everything above for future extensibility
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    """
    Immutable bundle of style constants for a single named theme.

    A theme groups every visual constant needed to render TrustLens plots so
    that future themes (``"dark"``, ``"colorblind"``, ``"publication"``) can be
    added by registering a new :class:`Theme` instance without touching
    plotting code.
    """

    name: str = "default"
    base_style: str = "seaborn-v0_8-whitegrid"
    palette: list[str] = field(default_factory=lambda: list(PALETTE))
    brand: dict[str, str] = field(default_factory=lambda: dict(BRAND_COLORS))
    semantic: dict[str, dict[str, str]] = field(
        default_factory=lambda: {k: dict(v) for k, v in SEMANTIC_COLORS.items()}
    )
    typography: dict[str, Any] = field(default_factory=lambda: dict(TYPOGRAPHY))
    grid: dict[str, Any] = field(default_factory=lambda: dict(GRID))
    fig_defaults: dict[str, Any] = field(default_factory=lambda: dict(FIG_DEFAULTS))
    spacing: dict[str, Any] = field(default_factory=lambda: dict(SPACING))


DEFAULT_THEME: Theme = Theme()

COLORBLIND_THEME: Theme = Theme(
    name="colorblind",
    palette=list(COLORBLIND_PALETTE),
    semantic=deepcopy(COLORBLIND_SEMANTIC_COLORS),
)


# ---------------------------------------------------------------------------
# Context manager — scoped rcParams mutations
# ---------------------------------------------------------------------------


@contextmanager
def apply_style(theme: Theme | None = None) -> Iterator[Theme]:
    """
    Apply a TrustLens theme inside a ``with`` block, restoring state on exit.
    """
    active = theme if theme is not None else DEFAULT_THEME
    previous = mpl.rcParams.copy()
    try:
        try:
            plt.style.use(active.base_style)
        except OSError:
            warnings.warn(
                f"TrustLens base style {active.base_style!r} could not be loaded; "
                "continuing with rcParams updates only.",
                stacklevel=2,
            )
        mpl.rcParams.update(
            {
                "font.family": active.typography["font_family"],
                "axes.titlesize": active.typography["title_size"],
                "axes.labelsize": active.typography["label_size"],
                "xtick.labelsize": active.typography["tick_size"],
                "ytick.labelsize": active.typography["tick_size"],
                "axes.titleweight": active.typography["title_weight"],
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.facecolor": active.fig_defaults["facecolor"],
                "axes.facecolor": active.fig_defaults["facecolor"],
                "savefig.dpi": active.fig_defaults["savefig_dpi"],
            }
        )
        yield active
    finally:
        mpl.rcParams.update(previous)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def styled_figure(
    figsize: tuple[float, float] | None = None,
    nrows: int = 1,
    ncols: int = 1,
    theme: Theme | None = None,
    grid: bool = True,
    **subplots_kwargs: Any,
) -> tuple[plt.Figure, Any]:
    """
    Create a figure and axes pre-configured with TrustLens styling.
    """
    active = theme if theme is not None else DEFAULT_THEME
    size = figsize if figsize is not None else active.fig_defaults["figsize"]

    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=size,
        facecolor=active.fig_defaults["facecolor"],
        **subplots_kwargs,
    )

    axes_iter = ax.flat if hasattr(ax, "flat") else [ax]
    for single_ax in axes_iter:
        single_ax.set_facecolor(active.fig_defaults["facecolor"])
        if grid:
            single_ax.grid(True, alpha=active.grid["alpha"], linewidth=active.grid["linewidth"])

    return fig, ax


def get_categorical_colors(n: int, theme: Theme | None = None) -> list[str]:
    """
    Return ``n`` categorical colors from the active theme palette.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    active = theme if theme is not None else DEFAULT_THEME
    palette = active.palette
    if not palette:
        raise ValueError("theme palette must be non-empty")
    return [palette[i % len(palette)] for i in range(n)]


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
