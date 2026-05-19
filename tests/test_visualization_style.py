"""
tests/test_visualization_style.py.
==================================
Tests for the centralized visualization style module.
"""

from __future__ import annotations

import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from trustlens.visualization import style
from trustlens.visualization.style import (
    BRAND_COLORS,
    DEFAULT_THEME,
    FIG_DEFAULTS,
    GRID,
    PALETTE,
    SEMANTIC_COLORS,
    SPACING,
    TYPOGRAPHY,
    Theme,
    apply_style,
    get_categorical_colors,
    styled_figure,
)

HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


class TestColorConstants:
    """Validate the shape and contents of color constants."""

    def test_brand_colors_are_valid_hex(self) -> None:
        for name, value in BRAND_COLORS.items():
            assert HEX_RE.match(value), f"BRAND_COLORS[{name!r}] = {value!r} is not 6-digit hex"

    def test_palette_is_non_empty_list_of_hex(self) -> None:
        assert isinstance(PALETTE, list)
        assert len(PALETTE) > 0
        for value in PALETTE:
            assert HEX_RE.match(value), f"{value!r} is not 6-digit hex"

    def test_palette_entries_are_unique(self) -> None:
        assert len(set(PALETTE)) == len(PALETTE), "PALETTE should not contain duplicates"

    def test_palette_entries_come_from_brand_colors(self) -> None:
        brand_values = set(BRAND_COLORS.values())
        for value in PALETTE:
            assert value in brand_values, f"{value} in PALETTE not registered in BRAND_COLORS"

    def test_semantic_severity_has_required_keys(self) -> None:
        expected = {"acceptable", "moderate", "severe", "unknown"}
        assert set(SEMANTIC_COLORS["severity"].keys()) == expected

    def test_semantic_verdict_has_required_keys(self) -> None:
        expected = {"deploy", "caution", "do_not_deploy"}
        assert set(SEMANTIC_COLORS["verdict"].keys()) == expected

    def test_semantic_grade_has_required_keys(self) -> None:
        expected = {"A", "B", "C", "D"}
        assert set(SEMANTIC_COLORS["grade"].keys()) == expected

    def test_semantic_groups_use_hex_colors(self) -> None:
        for group, colors in SEMANTIC_COLORS.items():
            for key, value in colors.items():
                assert HEX_RE.match(value), f"SEMANTIC_COLORS[{group!r}][{key!r}] = {value!r}"


class TestStaticConfigs:
    """Validate non-color constants."""

    def test_typography_has_required_keys(self) -> None:
        required = {"font_family", "title_size", "label_size", "tick_size", "annotation_size"}
        assert required.issubset(TYPOGRAPHY.keys())

    def test_grid_alpha_in_valid_range(self) -> None:
        assert 0.0 <= GRID["alpha"] <= 1.0
        assert 0.0 <= GRID["alpha_minor"] <= 1.0

    def test_fig_defaults_figsize_is_tuple(self) -> None:
        figsize = FIG_DEFAULTS["figsize"]
        assert isinstance(figsize, tuple)
        assert len(figsize) == 2
        assert all(isinstance(v, (int, float)) and v > 0 for v in figsize)

    def test_spacing_has_required_keys(self) -> None:
        required = {"bbox_pad", "bbox_alpha", "bar_edge_width", "bar_alpha"}
        assert required.issubset(SPACING.keys())


class TestTheme:
    """Validate the Theme dataclass behavior."""

    def test_default_theme_name(self) -> None:
        assert DEFAULT_THEME.name == "default"

    def test_theme_is_frozen(self) -> None:
        # dataclasses.FrozenInstanceError is a subclass of AttributeError.
        with pytest.raises(AttributeError):
            DEFAULT_THEME.name = "modified"  # type: ignore[misc]

    def test_two_default_themes_have_independent_mutable_fields(self) -> None:
        a = Theme()
        b = Theme()
        a.palette.append("#000000")
        assert "#000000" not in b.palette, "Theme instances must not share mutable state"

    def test_theme_palette_matches_module_palette(self) -> None:
        assert DEFAULT_THEME.palette == PALETTE

    def test_theme_semantic_matches_module_semantic(self) -> None:
        assert DEFAULT_THEME.semantic == SEMANTIC_COLORS


class TestApplyStyle:
    """Validate that apply_style scopes rcParams mutations correctly."""

    def test_rcparams_restored_after_context(self) -> None:
        before = mpl.rcParams.copy()
        with apply_style():
            pass
        for key, value in before.items():
            assert mpl.rcParams[key] == value, f"rcParams[{key!r}] not restored"

    def test_font_family_mutated_inside_context(self) -> None:
        with apply_style() as theme:
            assert mpl.rcParams["font.family"] == [theme.typography["font_family"]]

    def test_rcparams_restored_after_exception(self) -> None:
        before_font = mpl.rcParams["font.family"]
        with pytest.raises(RuntimeError, match="boom"):
            with apply_style():
                raise RuntimeError("boom")
        assert mpl.rcParams["font.family"] == before_font

    def test_apply_style_yields_theme(self) -> None:
        with apply_style() as theme:
            assert isinstance(theme, Theme)
            assert theme is DEFAULT_THEME

    def test_apply_style_accepts_explicit_theme(self) -> None:
        custom = Theme(name="custom")
        with apply_style(custom) as yielded:
            assert yielded is custom
            assert yielded.name == "custom"

    def test_nested_apply_style_restores_outer_state(self) -> None:
        original = mpl.rcParams["font.family"]
        with apply_style():
            mid = mpl.rcParams["font.family"]
            with apply_style():
                pass
            assert mpl.rcParams["font.family"] == mid
        assert mpl.rcParams["font.family"] == original

    def test_apply_style_warns_on_unknown_base_style(self) -> None:
        custom = Theme(name="bad", base_style="this-style-does-not-exist")
        with pytest.warns(UserWarning, match="could not be loaded"):
            with apply_style(custom):
                pass


class TestStyledFigure:
    """Validate the styled_figure helper."""

    def test_returns_figure_and_axes(self) -> None:
        fig, ax = styled_figure()
        assert isinstance(fig, plt.Figure)
        assert ax is not None
        plt.close(fig)

    def test_default_figsize_matches_theme(self) -> None:
        fig, _ax = styled_figure()
        assert tuple(fig.get_size_inches()) == tuple(DEFAULT_THEME.fig_defaults["figsize"])
        plt.close(fig)

    def test_custom_figsize_is_respected(self) -> None:
        fig, _ax = styled_figure(figsize=(12, 4))
        assert tuple(fig.get_size_inches()) == (12.0, 4.0)
        plt.close(fig)

    def test_multiple_subplots_returns_array(self) -> None:
        fig, axes = styled_figure(nrows=2, ncols=2)
        assert hasattr(axes, "flat")
        assert sum(1 for _ in axes.flat) == 4
        plt.close(fig)

    def test_facecolor_applied_to_figure_and_axes(self) -> None:
        fig, ax = styled_figure()
        expected = DEFAULT_THEME.fig_defaults["facecolor"]
        # matplotlib normalizes hex to lowercase + may add alpha; compare via to_hex
        from matplotlib.colors import to_hex

        assert to_hex(fig.get_facecolor()).lower() == expected.lower()
        assert to_hex(ax.get_facecolor()).lower() == expected.lower()
        plt.close(fig)


class TestGetCategoricalColors:
    """Validate the get_categorical_colors helper."""

    def test_returns_requested_count(self) -> None:
        assert len(get_categorical_colors(3)) == 3
        assert len(get_categorical_colors(8)) == 8

    def test_zero_returns_empty_list(self) -> None:
        assert get_categorical_colors(0) == []

    def test_cycles_when_n_exceeds_palette(self) -> None:
        palette_len = len(PALETTE)
        colors = get_categorical_colors(palette_len + 2)
        assert colors[0] == colors[palette_len]
        assert colors[1] == colors[palette_len + 1]

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            get_categorical_colors(-1)

    def test_uses_provided_theme(self) -> None:
        custom = Theme(palette=["#000000", "#FFFFFF"])
        assert get_categorical_colors(3, theme=custom) == ["#000000", "#FFFFFF", "#000000"]

    def test_empty_palette_raises(self) -> None:
        custom = Theme(palette=[])
        with pytest.raises(ValueError, match="non-empty"):
            get_categorical_colors(3, theme=custom)


class TestModuleSurface:
    """Ensure the module exposes its documented internal surface."""

    def test_top_level_names_present(self) -> None:
        required = {
            "BRAND_COLORS",
            "PALETTE",
            "SEMANTIC_COLORS",
            "TYPOGRAPHY",
            "GRID",
            "FIG_DEFAULTS",
            "SPACING",
            "Theme",
            "DEFAULT_THEME",
            "apply_style",
            "styled_figure",
            "get_categorical_colors",
        }
        for name in required:
            assert hasattr(style, name), f"style.{name} missing"
