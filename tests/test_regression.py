"""
tests/test_regression.py.
=========================
Unit tests for trustlens.metrics.regression.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from trustlens.metrics import regression
from trustlens.metrics.regression import (
    crps_decomposition,
    crps_from_intervals,
    error_distribution,
    error_variance_correlation,
    multilevel_interval_coverage,
    prediction_interval_coverage,
)


@pytest.fixture
def regression_data():
    """A small, deterministic regression fit via make_regression + LinearRegression."""
    X, y = make_regression(n_samples=200, n_features=5, noise=10.0, random_state=42)
    model = LinearRegression().fit(X, y)
    return np.asarray(y, dtype=float), model.predict(X)


class TestErrorDistribution:
    def test_perfect_predictor_zero_error(self):
        y = np.arange(10, dtype=float)
        dist = error_distribution(y, y.copy())
        assert dist["median_absolute_error"] == pytest.approx(0.0)
        assert dist["p90_absolute_error"] == pytest.approx(0.0)
        assert dist["max_error"] == pytest.approx(0.0)
        assert dist["rmse"] == pytest.approx(0.0)

    def test_medae_and_p90_known_values(self):
        # abs errors are exactly 0,1,...,9 -> MedAE=4.5, p90=8.1, max=9
        y_true = np.zeros(10, dtype=float)
        y_pred = -np.arange(10, dtype=float)
        dist = error_distribution(y_true, y_pred)
        assert dist["median_absolute_error"] == pytest.approx(4.5)
        assert dist["p90_absolute_error"] == pytest.approx(8.1)
        assert dist["max_error"] == pytest.approx(9.0)

    def test_on_make_regression_returns_finite(self, regression_data):
        y_true, y_pred = regression_data
        dist = error_distribution(y_true, y_pred)
        for key in (
            "median_absolute_error",
            "p90_absolute_error",
            "max_error",
            "mean_absolute_error",
            "rmse",
        ):
            assert isinstance(dist[key], float)
            assert np.isfinite(dist[key])
        assert dist["n_samples"] == len(y_true)
        assert dist["error_hist"].sum() == len(y_true)

    def test_shape_mismatch_raises_clear_error(self):
        with pytest.raises(ValueError, match="same shape"):
            error_distribution(np.zeros(5), np.zeros(4))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            error_distribution(np.array([]), np.array([]))

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError, match="n_bins"):
            error_distribution(np.zeros(5), np.zeros(5), n_bins=0)


class TestPredictionIntervalCoverage:
    def test_skips_when_no_intervals(self):
        result = prediction_interval_coverage(np.arange(10, dtype=float))
        assert result["status"] == "skipped"
        assert result["reason"] == "missing_intervals"

    def test_full_coverage_is_under_confident(self):
        y = np.arange(50, dtype=float)
        result = prediction_interval_coverage(y, y - 1e9, y + 1e9, confidence_level=0.95)
        assert result["picp"] == pytest.approx(1.0)
        assert result["verdict"] == "under-confident"

    def test_zero_width_intervals_are_over_confident(self):
        y = np.arange(50, dtype=float)
        point = y + 1.0  # point estimate never equals the truth
        result = prediction_interval_coverage(y, point, point, confidence_level=0.95)
        assert result["picp"] == pytest.approx(0.0)
        assert result["verdict"] == "over-confident"

    def test_well_calibrated_normal_intervals(self):
        rng = np.random.default_rng(7)
        y = rng.normal(0.0, 1.0, 4000)
        lo = np.full_like(y, -1.96)
        hi = np.full_like(y, 1.96)
        result = prediction_interval_coverage(y, lo, hi, confidence_level=0.95)
        assert result["picp"] == pytest.approx(0.95, abs=0.03)
        assert result["verdict"] == "well-calibrated"

    def test_lower_above_upper_raises(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="lower bound"):
            prediction_interval_coverage(y, y + 1.0, y - 1.0)

    def test_invalid_confidence_level_raises(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="confidence_level"):
            prediction_interval_coverage(y, y - 1.0, y + 1.0, confidence_level=1.5)


class TestMultilevelIntervalCoverage:
    def _well_calibrated_intervals(self, rng, n=4000):
        """Gaussian targets with exact theoretical central intervals at each level."""
        from scipy.stats import norm

        y = rng.normal(0.0, 1.0, n)
        levels = [0.5, 0.8, 0.95]
        intervals = {}
        for tau in levels:
            z = norm.ppf(0.5 + tau / 2.0)
            intervals[tau] = (np.full(n, -z), np.full(n, z))
        return y, intervals

    def test_skips_when_no_intervals(self):
        result = multilevel_interval_coverage(np.arange(10, dtype=float), None)
        assert result["status"] == "skipped"
        assert result["reason"] == "missing_intervals"

    def test_well_calibrated_has_low_ice(self):
        rng = np.random.default_rng(0)
        y, intervals = self._well_calibrated_intervals(rng)
        result = multilevel_interval_coverage(y, intervals, tolerance=0.05)
        assert result["ice"] < 0.05
        assert result["verdict"] == "well-calibrated"
        assert result["n_levels"] == 3
        assert result["n_calibrated_levels"] == 3

    def test_overconfident_levels_raise_ice_and_are_excluded_from_sharpness(self):
        # Narrow intervals (half the calibrated width) under-cover -> fail the gate.
        rng = np.random.default_rng(1)
        y, calibrated = self._well_calibrated_intervals(rng)
        narrow = {tau: (lo / 2.0, hi / 2.0) for tau, (lo, hi) in calibrated.items()}
        result = multilevel_interval_coverage(y, narrow, tolerance=0.05)
        assert result["ice"] > 0.05
        assert result["verdict"] == "over-confident"
        assert result["worst_calibration_error"] < -0.05
        # No level passes calibration -> the sharpness proxy is undefined, so the
        # artificially-narrow widths cannot inflate it.
        assert result["n_calibrated_levels"] == 0
        assert result["sharpness_skill"] is None

    def test_sharpness_skill_positive_when_sharper_than_climatology(self):
        # Intervals centered on the conditional mean span only the noise, so they
        # are genuinely tighter than the marginal (signal + noise) spread while
        # staying calibrated -> robustly positive skill (not a sampling-noise pass).
        from scipy.stats import norm

        rng = np.random.default_rng(2)
        n = 4000
        mu = rng.normal(0.0, 5.0, n)  # signal: wide marginal spread
        y = mu + rng.normal(0.0, 1.0, n)  # small unit noise around the mean
        intervals = {}
        for tau in (0.5, 0.8, 0.95):
            z = norm.ppf(0.5 + tau / 2.0)
            intervals[tau] = (mu - z, mu + z)  # width 2z from unit noise -> calibrated
        result = multilevel_interval_coverage(y, intervals, tolerance=0.05)
        assert result["sharpness_skill"] is not None
        assert result["n_calibrated_levels"] == 3
        # model width (~2z) << climatology width (~2z * 5.1) -> skill ~0.8.
        assert result["sharpness_skill"] > 0.3

    def test_single_level_emits_backcompat_picp_fields(self):
        rng = np.random.default_rng(3)
        y, intervals = self._well_calibrated_intervals(rng)
        one = {0.95: intervals[0.95]}
        result = multilevel_interval_coverage(y, one)
        assert result["n_levels"] == 1
        assert "picp" in result and "calibration_error" in result
        assert result["target_coverage"] == 0.95

    def test_shape_mismatch_raises(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="same shape"):
            multilevel_interval_coverage(y, {0.9: (np.zeros(4), np.ones(4))})

    def test_lower_above_upper_raises(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="lower bound"):
            multilevel_interval_coverage(y, {0.9: (y + 1.0, y - 1.0)})

    def test_invalid_level_raises(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="level"):
            multilevel_interval_coverage(y, {1.5: (y - 1.0, y + 1.0)})


class TestCrpsFromIntervals:
    @staticmethod
    def _gaussian_intervals(mu, sigma, n, levels):
        """Per-sample constant central intervals for a shared N(mu, sigma) forecast."""
        from scipy.stats import norm

        intervals = {}
        for tau in levels:
            z_lo = norm.ppf((1.0 - tau) / 2.0)
            z_hi = norm.ppf((1.0 + tau) / 2.0)
            intervals[float(tau)] = (
                np.full(n, mu + sigma * z_lo),
                np.full(n, mu + sigma * z_hi),
            )
        return intervals

    @staticmethod
    def _closed_form_gaussian_crps(mu, sigma, y):
        """Exact CRPS of N(mu, sigma^2) vs y (Gneiting & Raftery 2007)."""
        from scipy.stats import norm

        w = (y - mu) / sigma
        return sigma * (w * (2 * norm.cdf(w) - 1) + 2 * norm.pdf(w) - 1 / np.sqrt(np.pi))

    def test_skips_when_no_intervals(self):
        result = crps_from_intervals(np.arange(10, dtype=float), None)
        assert result["status"] == "skipped"
        assert result["reason"] == "missing_intervals"

    def test_skips_when_empty_intervals_dict(self):
        # Empty mapping degrades like None (guards the `if not intervals` gate
        # against a refactor to `if intervals is None`).
        result = crps_from_intervals(np.arange(10, dtype=float), {})
        assert result["status"] == "skipped"
        assert result["reason"] == "missing_intervals"

    def test_single_level_is_sufficient_to_integrate(self):
        # One central level already supplies two quantiles, enough to integrate.
        y = np.zeros(20, dtype=float)
        result = crps_from_intervals(y, self._gaussian_intervals(0.0, 1.0, 20, [0.9]))
        assert result["n_quantile_levels"] == 2
        assert np.isfinite(result["mean_crps"])

    def test_matches_closed_form_gaussian_on_dense_grid(self):
        rng = np.random.default_rng(0)
        mu, sigma, n = 1.3, 0.8, 3000
        y = rng.normal(mu, sigma, n)
        levels = np.round(np.arange(0.05, 0.96, 0.05), 3)  # 19 interval levels
        result = crps_from_intervals(y, self._gaussian_intervals(mu, sigma, n, levels))
        expected = float(self._closed_form_gaussian_crps(mu, sigma, y).mean())
        # 19-level grid is characterised at <1% mean rel. error; allow 2% margin.
        assert result["mean_crps"] == pytest.approx(expected, rel=0.02)
        assert result["n_quantile_levels"] == 38
        assert result["n_samples"] == n

    def test_denser_grid_reduces_truncation_bias(self):
        rng = np.random.default_rng(1)
        mu, sigma, n = 1.3, 0.8, 3000
        y = rng.normal(mu, sigma, n)
        exact = float(self._closed_form_gaussian_crps(mu, sigma, y).mean())
        coarse = crps_from_intervals(y, self._gaussian_intervals(mu, sigma, n, [0.5, 0.8, 0.95]))[
            "mean_crps"
        ]
        dense = crps_from_intervals(
            y, self._gaussian_intervals(mu, sigma, n, np.round(np.arange(0.05, 0.96, 0.05), 3))
        )["mean_crps"]
        assert abs(dense - exact) < abs(coarse - exact)

    def test_sharper_forecast_has_lower_crps(self):
        # Same observations at the forecast mean; the tighter forecast scores lower.
        rng = np.random.default_rng(2)
        mu, n = 0.0, 3000
        y = rng.normal(mu, 0.05, n)  # observations essentially at the mean
        levels = np.round(np.arange(0.05, 0.96, 0.05), 3)
        sharp = crps_from_intervals(y, self._gaussian_intervals(mu, 0.5, n, levels))["mean_crps"]
        broad = crps_from_intervals(y, self._gaussian_intervals(mu, 2.0, n, levels))["mean_crps"]
        assert sharp < broad

    def test_point_mass_on_truth_scores_near_zero(self):
        # Degenerate intervals collapsed onto y at every level -> CRPS ~ 0.
        y = np.linspace(-3.0, 3.0, 50)
        intervals = {tau: (y.copy(), y.copy()) for tau in (0.5, 0.8, 0.95)}
        assert crps_from_intervals(y, intervals)["mean_crps"] == pytest.approx(0.0, abs=1e-9)

    def test_quantile_crossing_is_rearranged_not_crashed(self):
        # Non-nested (crossed) intervals must not crash and must equal the score of
        # the same quantiles sorted into a valid non-decreasing forecast.
        n = 200
        y = np.zeros(n, dtype=float)
        crossed = {
            0.5: (np.full(n, 1.0), np.full(n, 2.0)),  # inner band placed *outside*
            0.9: (np.full(n, -0.5), np.full(n, 0.5)),  # outer band placed *inside*
        }
        result = crps_from_intervals(y, crossed)
        assert np.isfinite(result["mean_crps"])
        # Rearranged reference: sort the four quantile values ascending per sample.
        vals = np.sort(np.array([1.0, 2.0, -0.5, 0.5]))
        rearranged = {
            0.5: (np.full(n, vals[1]), np.full(n, vals[2])),
            0.9: (np.full(n, vals[0]), np.full(n, vals[3])),
        }
        assert result["mean_crps"] == pytest.approx(crps_from_intervals(y, rearranged)["mean_crps"])

    def test_grid_metadata_reported(self):
        y = np.zeros(20, dtype=float)
        result = crps_from_intervals(y, self._gaussian_intervals(0.0, 1.0, 20, [0.5, 0.9]))
        assert result["n_quantile_levels"] == 4
        lo, hi = result["quantile_level_span"]
        assert lo == pytest.approx(0.05) and hi == pytest.approx(0.95)

    def test_shape_mismatch_raises(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="same shape"):
            crps_from_intervals(y, {0.9: (np.zeros(4), np.ones(4))})

    def test_lower_above_upper_raises(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="lower bound"):
            crps_from_intervals(y, {0.9: (y + 1.0, y - 1.0)})

    def test_invalid_level_raises(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="level"):
            crps_from_intervals(y, {1.5: (y - 1.0, y + 1.0)})

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            crps_from_intervals(np.array([]), {0.9: (np.array([]), np.array([]))})


class TestCrpsDecomposition:
    @staticmethod
    def _conditional_gaussian(rng, n=6000, sd=1.0):
        """Wide-signal conditional Gaussian: obs ~ N(mu, 1), forecast = N(mu, sd)."""
        from scipy.stats import norm

        levels = np.round(np.arange(0.05, 0.96, 0.05), 3)
        mu = rng.normal(0.0, 5.0, n)
        y = mu + rng.normal(0.0, 1.0, n)
        intervals = {}
        for tau in levels:
            z = norm.ppf((1 + tau) / 2)
            intervals[float(tau)] = (mu - z * sd, mu + z * sd)
        return y, intervals

    def test_skips_when_no_intervals(self):
        result = crps_decomposition(np.arange(10, dtype=float), None)
        assert result["status"] == "skipped"
        assert result["reason"] == "missing_intervals"

    def test_identity_reliability_plus_potential_equals_crps(self):
        rng = np.random.default_rng(0)
        y, intervals = self._conditional_gaussian(rng)
        d = crps_decomposition(y, intervals)
        assert d["reliability"] + d["crps_potential"] == pytest.approx(d["crps"], abs=1e-3)

    def test_three_term_identity_crps_equals_rel_minus_res_plus_unc(self):
        rng = np.random.default_rng(1)
        y, intervals = self._conditional_gaussian(rng)
        d = crps_decomposition(y, intervals)
        assert d["reliability"] - d["resolution"] + d["uncertainty"] == pytest.approx(
            d["crps"], abs=1e-3
        )

    def test_crps_matches_estimator_within_grid_tolerance(self):
        rng = np.random.default_rng(2)
        y, intervals = self._conditional_gaussian(rng)
        d = crps_decomposition(y, intervals)
        est = crps_from_intervals(y, intervals)["mean_crps"]
        assert d["crps"] == pytest.approx(est, rel=0.05)

    def test_reliability_nonnegative_and_small_when_calibrated(self):
        rng = np.random.default_rng(3)
        y, intervals = self._conditional_gaussian(rng, sd=1.0)  # forecast sd == true sd
        d = crps_decomposition(y, intervals)
        assert d["reliability"] >= 0.0
        assert d["reliability"] < 0.02

    def test_reliability_rises_when_miscalibrated(self):
        from scipy.stats import norm

        rng = np.random.default_rng(4)
        n = 6000
        mu = rng.normal(0.0, 5.0, n)
        y = mu + rng.normal(0.0, 1.0, n)
        levels = np.round(np.arange(0.05, 0.96, 0.05), 3)

        def ivs(sd):
            return {
                float(t): (mu - norm.ppf((1 + t) / 2) * sd, mu + norm.ppf((1 + t) / 2) * sd)
                for t in levels
            }

        rel_calibrated = crps_decomposition(y, ivs(1.0))["reliability"]
        rel_overconfident = crps_decomposition(y, ivs(0.5))["reliability"]
        assert rel_overconfident > rel_calibrated

    def test_resolution_positive_for_skillful_forecast(self):
        rng = np.random.default_rng(5)
        y, intervals = self._conditional_gaussian(rng, sd=1.0)
        d = crps_decomposition(y, intervals)
        assert d["resolution"] > 0.0
        assert d["uncertainty"] > d["crps_potential"]

    def test_climatology_forecast_has_near_zero_resolution(self):
        # A forecast equal to climatology (marginal quantiles, constant across
        # samples) has no skill over climatology -> resolution ~ 0.
        rng = np.random.default_rng(6)
        y = rng.normal(0.0, 3.0, 6000)
        levels = np.round(np.arange(0.05, 0.96, 0.05), 3)
        clim = {}
        for tau in levels:
            lo = float(np.quantile(y, (1 - tau) / 2))
            hi = float(np.quantile(y, (1 + tau) / 2))
            clim[float(tau)] = (np.full(y.size, lo), np.full(y.size, hi))
        d = crps_decomposition(y, clim)
        assert abs(d["resolution"]) < 0.02

    def test_custom_climatology_wrong_levels_raises(self):
        rng = np.random.default_rng(7)
        y, intervals = self._conditional_gaussian(rng)
        bad_clim = {0.5: (np.full(y.size, -1.0), np.full(y.size, 1.0))}  # different grid
        with pytest.raises(ValueError, match="same interval levels"):
            crps_decomposition(y, intervals, climatology=bad_clim)

    def test_shape_mismatch_raises(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="same shape"):
            crps_decomposition(y, {0.9: (np.zeros(4), np.ones(4))})

    def test_invalid_level_raises(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="level"):
            crps_decomposition(y, {1.5: (y - 1.0, y + 1.0)})


class TestErrorVarianceCorrelation:
    def test_skips_when_no_variance(self):
        y = np.arange(10, dtype=float)
        result = error_variance_correlation(y, y.copy())
        assert result["status"] == "skipped"
        assert result["reason"] == "missing_variance"

    def test_informative_when_variance_tracks_error(self):
        rng = np.random.default_rng(1)
        y_true = rng.normal(0.0, 1.0, 500)
        abs_err = rng.uniform(0.0, 5.0, 500)
        # construct predictions with the chosen error magnitude, variance ~ error
        y_pred = y_true + abs_err
        predicted_variance = abs_err + rng.normal(0.0, 0.1, 500)
        result = error_variance_correlation(y_true, y_pred, predicted_variance)
        assert result["spearman"] > 0.8
        assert result["verdict"] == "informative"

    def test_uninformative_when_variance_is_random(self):
        rng = np.random.default_rng(2)
        y_true = rng.normal(0.0, 1.0, 500)
        y_pred = y_true + rng.normal(0.0, 1.0, 500)
        predicted_variance = rng.uniform(0.0, 1.0, 500)  # independent of error
        result = error_variance_correlation(y_true, y_pred, predicted_variance)
        assert abs(result["spearman"]) < 0.2
        assert result["verdict"] == "uninformative"

    def test_constant_variance_returns_zero(self):
        y_true = np.arange(20, dtype=float)
        y_pred = y_true + 1.0
        predicted_variance = np.ones(20)
        result = error_variance_correlation(y_true, y_pred, predicted_variance)
        assert result["pearson"] == 0.0
        assert result["spearman"] == 0.0

    def test_shape_mismatch_raises_clear_error(self):
        with pytest.raises(ValueError, match="same shape"):
            error_variance_correlation(np.zeros(5), np.zeros(5), np.zeros(4))


def test_regression_module_exports_match_all():
    # mirror tests/test_metrics_all_exports.py's guard for the new module
    assert set(regression.__all__) == {
        "error_distribution",
        "prediction_interval_coverage",
        "multilevel_interval_coverage",
        "crps_from_intervals",
        "crps_decomposition",
        "error_variance_correlation",
    }
