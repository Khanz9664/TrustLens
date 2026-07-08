"""
tests/test_conformal.py.
========================
Unit tests for trustlens.metrics.conformal.

All fixtures are hand-built prediction sets with coverage numbers computed by
hand, so every assertion pins an exact known value rather than a property.
"""

import numpy as np
import pytest

from trustlens.metrics.conformal import (
    class_conditional_coverage,
    conformal_diagnostics,
    marginal_coverage,
    set_size_summary,
    size_stratified_coverage,
    to_membership_matrix,
)

# ---------------------------------------------------------------------------
# Shared construction
# ---------------------------------------------------------------------------


def _headline_case():
    """A 100-sample K=3 case where marginal coverage 0.90 hides conditional failure.

    Group A (80 samples, true label 0, size-1 sets): 76 cover, 4 miss -> 0.95.
    Group B (20 samples, true label 1, size-2 sets): 14 cover, 6 miss  -> 0.70.
    Marginal = (76 + 14) / 100 = 0.90.

    class-conditional: class 0 -> 0.95, class 1 -> 0.70 (worst).
    size-stratified:   size 1 -> 0.95, size 2 -> 0.70 (worst, both eligible).
    sizes: 80 singletons + 20 doubletons -> avg 1.2, singleton_rate 0.8.
    """
    y_true = [0] * 80 + [1] * 20
    sets = (
        [[0]] * 76  # true 0, covered, size 1
        + [[1]] * 4  # true 0, miss, size 1
        + [[1, 2]] * 14  # true 1, covered, size 2
        + [[0, 2]] * 6  # true 1, miss, size 2
    )
    return y_true, sets


# ---------------------------------------------------------------------------
# to_membership_matrix
# ---------------------------------------------------------------------------


class TestToMembershipMatrix:
    def test_label_lists_round_trip(self):
        S = to_membership_matrix([[0, 2], [1], []])
        assert S.dtype == bool
        assert S.shape == (3, 3)
        expected = np.array([[True, False, True], [False, True, False], [False, False, False]])
        np.testing.assert_array_equal(S, expected)

    def test_label_lists_respect_n_classes(self):
        S = to_membership_matrix([[0], [1]], n_classes=4)
        assert S.shape == (2, 4)

    def test_matrix_passes_through_as_bool(self):
        m = np.array([[1, 0, 1], [0, 1, 0]])
        S = to_membership_matrix(m)
        assert S.dtype == bool
        assert S.shape == (2, 3)
        np.testing.assert_array_equal(S, m.astype(bool))

    def test_bool_matrix_passes_through(self):
        m = np.array([[True, False], [False, True]])
        S = to_membership_matrix(m)
        assert S.dtype == bool
        np.testing.assert_array_equal(S, m)

    def test_rectangular_01_native_list_is_ambiguous_and_raises(self):
        # A rectangular all-0/1 native list could be a matrix OR label lists;
        # without n_classes (or a numpy array) it must raise rather than guess.
        with pytest.raises(ValueError, match="Ambiguous input"):
            to_membership_matrix([[1, 0], [0, 1]])

    def test_rectangular_01_ndarray_is_unambiguously_matrix(self):
        # A 2D numpy array is unambiguous -> matrix, no n_classes needed.
        S = to_membership_matrix(np.array([[1, 0], [0, 1]]))
        assert S.shape == (2, 2)
        np.testing.assert_array_equal(S, np.array([[True, False], [False, True]]))

    def test_rectangular_01_list_matrix_via_matching_n_classes(self):
        # width == n_classes -> read as a matrix.
        S = to_membership_matrix([[1, 0], [0, 1]], n_classes=2)
        assert S.shape == (2, 2)
        np.testing.assert_array_equal(S, np.array([[True, False], [False, True]]))

    def test_rectangular_01_list_label_lists_via_nonmatching_n_classes(self):
        # width (1) != n_classes (3) -> the same 0/1 list is read as label lists:
        # sample 0 = {0}, sample 1 = {1}.
        S = to_membership_matrix([[0], [1]], n_classes=3)
        assert S.shape == (2, 3)
        np.testing.assert_array_equal(S, np.array([[True, False, False], [False, True, False]]))

    def test_invalid_label_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            to_membership_matrix([[0, 3]], n_classes=2)

    def test_negative_label_raises(self):
        with pytest.raises(ValueError, match="negative"):
            to_membership_matrix([[-1], [0]])

    def test_non_integer_label_raises(self):
        with pytest.raises(ValueError, match="integer"):
            to_membership_matrix([[1.5], [0]])

    def test_matrix_with_non_binary_value_is_label_lists(self):
        # Equal-length rows but a value of 2 -> read as label lists, inferred K=3.
        S = to_membership_matrix([[0, 2], [1, 2]])
        assert S.shape == (2, 3)
        np.testing.assert_array_equal(S, np.array([[True, False, True], [False, True, True]]))

    def test_all_empty_without_n_classes_raises(self):
        with pytest.raises(ValueError, match="infer n_classes"):
            to_membership_matrix([[], []])

    def test_empty_input_without_n_classes_raises(self):
        with pytest.raises(ValueError, match="empty input"):
            to_membership_matrix([])


# ---------------------------------------------------------------------------
# marginal_coverage
# ---------------------------------------------------------------------------


class TestMarginalCoverage:
    def test_known_fraction(self):
        # 3 covered out of 4 -> 0.75.
        y_true = [0, 1, 2, 0]
        sets = [[0], [1], [2], [1]]  # last misses (true 0 not in {1})
        assert marginal_coverage(y_true, sets) == pytest.approx(0.75)

    def test_empty_set_is_a_miss(self):
        y_true = [0, 1, 2]
        sets = [[0], [1], []]  # third is empty -> miss
        assert marginal_coverage(y_true, sets) == pytest.approx(2 / 3)

    def test_perfect_coverage(self):
        y_true = [0, 1, 2]
        sets = [[0, 1], [1], [0, 2]]
        assert marginal_coverage(y_true, sets) == pytest.approx(1.0)

    def test_headline_case_is_0_90(self):
        y_true, sets = _headline_case()
        assert marginal_coverage(y_true, sets) == pytest.approx(0.90)

    def test_returns_float(self):
        # n_classes=2 disambiguates the 0/1 list as label lists ({0}, {1}).
        assert isinstance(marginal_coverage([0, 1], [[0], [1]], n_classes=2), float)

    def test_shape_mismatch_raises(self):
        # [[0], [2]] is unambiguously label lists (contains a 2) -> 2 samples.
        with pytest.raises(ValueError, match="Length mismatch"):
            marginal_coverage([0, 1, 2], [[0], [2]])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            marginal_coverage([], [])


# ---------------------------------------------------------------------------
# class_conditional_coverage
# ---------------------------------------------------------------------------


class TestClassConditionalCoverage:
    def test_one_class_under_covers(self):
        # class 0: 4 samples all covered -> 1.0
        # class 1: 4 samples, 2 covered   -> 0.5 (worst)
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        sets = [[0], [0], [0], [0], [1], [1], [0], [2]]
        out = class_conditional_coverage(y_true, sets)
        assert out["per_class"][0] == pytest.approx(1.0)
        assert out["per_class"][1] == pytest.approx(0.5)
        assert out["worst_class"] == 1
        assert out["worst_coverage"] == pytest.approx(0.5)

    def test_zero_sample_classes_omitted(self):
        # n_classes=3 but class 2 never appears as a true label.
        y_true = [0, 1]
        sets = [[0], [1]]
        out = class_conditional_coverage(y_true, sets, n_classes=3)
        assert set(out["per_class"].keys()) == {0, 1}

    def test_headline_worst_class(self):
        y_true, sets = _headline_case()
        out = class_conditional_coverage(y_true, sets)
        assert out["per_class"][0] == pytest.approx(0.95)
        assert out["per_class"][1] == pytest.approx(0.70)
        assert out["worst_class"] == 1
        assert out["worst_coverage"] == pytest.approx(0.70)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            class_conditional_coverage([0, 1, 0], [[0], [2]])

    def test_class_never_in_any_set_is_zero_coverage(self):
        # Class 2 is a true label but appears in no prediction set (inferred K=2).
        # Its true samples cannot be covered -> surfaced as 0.0 coverage, not raised.
        y_true = [0, 1, 2, 2]
        sets = [[0, 1], [1], [0], [1]]  # ragged -> label lists; no set contains class 2
        out = class_conditional_coverage(y_true, sets)
        assert out["per_class"][2] == pytest.approx(0.0)
        assert out["worst_class"] == 2
        assert out["worst_coverage"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# size_stratified_coverage
# ---------------------------------------------------------------------------


class TestSizeStratifiedCoverage:
    def test_marginal_hides_stratum_failure(self):
        # Marginal 0.90 overall, but the size-2 stratum sits at exactly 0.70.
        y_true, sets = _headline_case()
        assert marginal_coverage(y_true, sets) == pytest.approx(0.90)
        out = size_stratified_coverage(y_true, sets, min_stratum=20)
        assert out["per_size"][1]["coverage"] == pytest.approx(0.95)
        assert out["per_size"][1]["count"] == 80
        assert out["per_size"][2]["coverage"] == pytest.approx(0.70)
        assert out["per_size"][2]["count"] == 20
        assert out["worst_stratum_coverage"] == pytest.approx(0.70)
        assert out["worst_stratum_size"] == 2

    def test_tiny_stratum_flagged_and_excluded(self):
        # size 1: 20 samples @ 0.90 (eligible)
        # size 2: 20 samples @ 0.70 (eligible, worst among eligible)
        # size 3: 5 samples  @ 0.00 (tiny -> flagged, excluded from worst)
        y_true = [0] * 20 + [1] * 20 + [3] * 5
        sets = (
            [[0]] * 18  # size 1 cover
            + [[1]] * 2  # size 1 miss
            + [[1, 2]] * 14  # size 2 cover
            + [[0, 2]] * 6  # size 2 miss
            + [[0, 1, 2]] * 5  # size 3, true label 3 absent -> miss
        )
        out = size_stratified_coverage(y_true, sets, min_stratum=20)
        assert out["per_size"][1]["low_confidence"] is False
        assert out["per_size"][2]["low_confidence"] is False
        assert out["per_size"][3]["low_confidence"] is True
        assert out["per_size"][3]["coverage"] == pytest.approx(0.0)
        assert out["per_size"][3]["count"] == 5
        # The 0.00 tiny stratum is excluded; worst eligible is the size-2 0.70.
        assert out["worst_stratum_coverage"] == pytest.approx(0.70)
        assert out["worst_stratum_size"] == 2

    def test_no_eligible_stratum_returns_none(self):
        # Every stratum below the (large) threshold -> no eligible worst.
        y_true = [0, 1, 2]
        sets = [[0], [1], [2]]
        out = size_stratified_coverage(y_true, sets, min_stratum=20)
        assert out["worst_stratum_coverage"] is None
        assert out["worst_stratum_size"] is None
        assert out["per_size"][1]["low_confidence"] is True


# ---------------------------------------------------------------------------
# set_size_summary
# ---------------------------------------------------------------------------


class TestSetSizeSummary:
    def test_all_singleton(self):
        out = set_size_summary([[0], [1], [2], [0]], n_classes=3)
        assert out["avg_size"] == pytest.approx(1.0)
        assert out["singleton_rate"] == pytest.approx(1.0)
        assert out["empty_rate"] == pytest.approx(0.0)
        assert out["size_efficiency"] == pytest.approx(1.0)

    def test_all_full_sets_zero_efficiency(self):
        # Every set is the full label space -> efficiency 0.0.
        out = set_size_summary([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        assert out["avg_size"] == pytest.approx(3.0)
        assert out["size_efficiency"] == pytest.approx(0.0)

    def test_mixed_exact_avg(self):
        # sizes 1, 2, 3 with K=3 -> avg 2.0, efficiency 1 - (2-1)/(3-1) = 0.5.
        out = set_size_summary([[0], [0, 1], [0, 1, 2]])
        assert out["avg_size"] == pytest.approx(2.0)
        assert out["singleton_rate"] == pytest.approx(1 / 3)
        assert out["size_efficiency"] == pytest.approx(0.5)

    def test_empty_rate(self):
        # sizes 0, 1, 2 -> empty_rate 1/3.
        out = set_size_summary([[], [0], [1, 2]], n_classes=3)
        assert out["empty_rate"] == pytest.approx(1 / 3)
        assert out["avg_size"] == pytest.approx(1.0)

    def test_single_class_efficiency_is_one(self):
        # K == 1 edge: efficiency defined as 1.0 regardless of sizes.
        out = set_size_summary([[0], [], [0]], n_classes=1)
        assert out["size_efficiency"] == pytest.approx(1.0)
        assert out["singleton_rate"] == pytest.approx(2 / 3)
        assert out["empty_rate"] == pytest.approx(1 / 3)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            set_size_summary([], n_classes=3)


# ---------------------------------------------------------------------------
# conformal_diagnostics (orchestrator)
# ---------------------------------------------------------------------------


class TestConformalDiagnostics:
    def test_end_to_end_all_fields(self):
        y_true, sets = _headline_case()
        out = conformal_diagnostics(y_true, sets, nominal_coverage=0.90, n_classes=3)

        assert out["nominal"] == pytest.approx(0.90)
        assert out["marginal_coverage"] == pytest.approx(0.90)
        assert out["coverage_gap"] == pytest.approx(0.0)
        assert out["worst_class_coverage"] == pytest.approx(0.70)
        assert out["worst_class_gap"] == pytest.approx(0.20)
        assert out["ssc_violation"] == pytest.approx(0.20)
        assert out["avg_set_size"] == pytest.approx(1.2)
        assert out["singleton_rate"] == pytest.approx(0.8)
        assert out["size_efficiency"] == pytest.approx(0.9)
        assert out["empty_rate"] == pytest.approx(0.0)
        assert out["n_classes"] == 3
        assert out["n_samples"] == 100

    def test_nominal_none_leaves_gaps_none(self):
        y_true, sets = _headline_case()
        out = conformal_diagnostics(y_true, sets, nominal_coverage=None, n_classes=3)

        # Gap/violation fields are None, never a silent 0.
        assert out["nominal"] is None
        assert out["coverage_gap"] is None
        assert out["worst_class_gap"] is None
        assert out["ssc_violation"] is None
        # Coverage and size numbers are still reported.
        assert out["marginal_coverage"] == pytest.approx(0.90)
        assert out["worst_class_coverage"] == pytest.approx(0.70)
        assert out["avg_set_size"] == pytest.approx(1.2)
        assert out["size_efficiency"] == pytest.approx(0.9)

    def test_ssc_violation_none_when_no_eligible_stratum(self):
        # Fewer than min_stratum (20) samples per size -> no eligible stratum,
        # so ssc_violation is None even with a nominal target supplied.
        y_true = [0, 1, 2]
        sets = [[0], [1], [2]]
        out = conformal_diagnostics(y_true, sets, nominal_coverage=0.9)
        assert out["ssc_violation"] is None

    def test_invalid_nominal_raises(self):
        with pytest.raises(ValueError, match="nominal_coverage"):
            conformal_diagnostics([0, 1], [[0], [1]], nominal_coverage=1.5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            conformal_diagnostics([0, 1, 2], [[0], [2]])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            conformal_diagnostics([], [], n_classes=3)


# ---------------------------------------------------------------------------
# Explicit n_classes: declared class space is a contract, not a diagnostic
# ---------------------------------------------------------------------------


class TestExplicitNClassesContract:
    """With an explicit n_classes, a y_true label outside [0, n_classes) is a
    data-consistency error and raises; with inferred K it stays a coverage miss."""

    def test_marginal_raises_on_out_of_range_label(self):
        with pytest.raises(ValueError, match="outside the declared class space"):
            marginal_coverage([0, 1, 5], [[0], [1], [1]], n_classes=3)

    def test_class_conditional_raises_on_out_of_range_label(self):
        with pytest.raises(ValueError, match="outside the declared class space"):
            class_conditional_coverage([0, 1, 5], [[0], [1], [1]], n_classes=3)

    def test_size_stratified_raises_on_out_of_range_label(self):
        with pytest.raises(ValueError, match="outside the declared class space"):
            size_stratified_coverage([0, 1, 5], [[0], [1], [1]], n_classes=3)

    def test_conformal_diagnostics_raises_on_out_of_range_label(self):
        with pytest.raises(ValueError, match="outside the declared class space"):
            conformal_diagnostics([0, 1, 5], [[0], [1], [1]], n_classes=3)

    def test_inferred_k_keeps_miss_behavior(self):
        # Same class-overflow shape, but with inferred K it is a miss (0.0), not
        # an error: class 2 is simply never predicted (ragged -> unambiguous,
        # inferred K=2, so true label 2 is out-of-span and counts as a miss).
        out = class_conditional_coverage([0, 0, 2, 2], [[0, 1], [1], [0], [1]])
        assert out["per_class"][2] == pytest.approx(0.0)  # class 2 never covered

    def test_marginal_accepts_n_classes_passthrough(self):
        # n_classes wider than any observed label is a valid, consistent contract.
        cov = marginal_coverage([0, 1], [[0], [1]], n_classes=5)
        assert cov == pytest.approx(1.0)

    def test_size_stratified_accepts_n_classes_passthrough(self):
        out = size_stratified_coverage([0, 1], [[0], [1]], min_stratum=1, n_classes=5)
        assert out["per_size"][1]["coverage"] == pytest.approx(1.0)

    def test_label_exactly_at_boundary_raises(self):
        # n_classes=3 -> valid labels 0..2; label 3 is the first illegal one.
        with pytest.raises(ValueError, match="outside the declared class space"):
            marginal_coverage([0, 3], [[0], [0]], n_classes=3)
