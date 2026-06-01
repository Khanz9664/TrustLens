"""
trustlens.report.
=================
TrustReport — the central result container returned by ``analyze()``.

Responsibilities
----------------
* Store all computed metric results in a structured dictionary.
* Compute and expose the Trust Score (0–100).
* Provide human-readable console summaries via ``show()``.
* Render the summary dashboard via ``summary_plot()``.
* Surface critical failures via ``show_failures()``.
* Render per-module plots via ``plot()``.
* Persist the full report to disk via ``save()``.
* Support JSON serialization for downstream consumption.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

from ._version import __version__

logger = logging.getLogger(__name__)


class TrustReport:
    """
    Container for all TrustLens analysis results.

    Attributes
    ----------
    results : dict
      Nested dictionary keyed by module name.
    trust_score : TrustScoreResult
      Composite 0–100 trust score with sub-scores and grade.
    metadata : dict
      Run-level metadata (timestamp, library version, etc.).
    model : Any
      Reference to the analysed model (not serialized to JSON).
    """

    def __init__(
        self,
        results: dict[str, Any],
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        embeddings: np.ndarray | None = None,
        framework: str | None = None,
        backend_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.results = results
        self.model = model
        self.X = X
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.embeddings = embeddings
        self.framework = framework
        self.backend_metadata = backend_metadata or {}
        self.metadata = self._build_metadata()

        # Compute Trust Score immediately so it's always available
        from trustlens.trust_score import compute_trust_score

        self.trust_score = compute_trust_score(results)
        self._patterns: list[str] = []
        self._compute_patterns()

    @property
    def patterns(self) -> list[str]:
        """Detected behavioral patterns (e.g. 'Confidently Wrong')."""
        return self._patterns

    def _compute_patterns(self) -> None:
        """Derive patterns directly from metrics."""
        failure_score = self.trust_score.sub_scores.get("failure", 100.0)
        ece = self.results.get("calibration", {}).get("ece", 0.0)
        conf_gap = self.results.get("failure", {}).get("confidence_gap", {}).get("gap", 0.0)

        # High-confidence errors mean avg confidence of mistakes is high
        avg_err_conf = (
            self.results.get("failure", {})
            .get("confidence_gap", {})
            .get("incorrect_confidence_mean", 0.0)
        )

        # 1. Confidently Wrong
        if (failure_score < 40 or avg_err_conf > 0.65) and conf_gap < 0.1:
            self._patterns.append("Confidently Wrong")

        # 2. Safe Failures
        if failure_score < 60 and avg_err_conf < 0.5 and conf_gap > 0.15:
            self._patterns.append("Safe Failures")

        # 3. Calibration Drift
        if ece > 0.1 or (failure_score > 70 and ece > 0.08):
            self._patterns.append("Calibration Drift")

    def _format_score_explanation(self) -> list[str]:
        """Rank and format top penalties for explanation."""
        penalties = self.trust_score.penalties_applied
        if not penalties:
            return []

        # Sort by magnitude descending
        sorted_p = sorted(penalties.items(), key=lambda x: x[1], reverse=True)
        top_p = sorted_p[:3]

        lines = ["Score Explanation:"]
        labels = ["Dominant Issue", "Secondary Issue", "Minor Impact"]

        for i, (name, val) in enumerate(top_p):
            if val > 0:
                label = labels[i] if i < len(labels) else "Other Impact"
                lines.append(f"  - {label:<16}: {name} (-{val:.1f})")
        return lines

    @property
    def deployment_explanation(self) -> dict[str, Any]:
        """Provide a structured explanation for the deployment verdict."""
        ts = self.trust_score
        grade_map = {"A": "PASS", "B": "CAUTION", "C": "CAUTION", "D": "BLOCK"}
        verdict = "BLOCK" if ts.is_blocked else grade_map.get(ts.grade, "PASS")

        reasons = []
        recommendations = []
        penalties = getattr(ts, "penalties_applied", {})

        # Recommendations mapping
        rec_map = {
            "calibration": "Consider probability calibration techniques such as temperature scaling or isotonic regression before deployment.",
            "fairness": "Investigate subgroup performance disparities and consider fairness constraints before deployment.",
            "failure": "Inspect high-confidence misclassifications and verify confidence-weighted error distributions.",
            "representation": "Review latent embedding quality; poor representation may indicate failure to capture fundamental class separation.",
        }

        # Map dimension to standard penalty name logic
        dim_to_penalty_key = {
            "calibration": "Calibration",
            "failure": "Failure",
            "bias": "Fairness",
            "representation": "Representation",
        }

        # Case-insensitive penalty key matching
        penalties_lower = {k.lower(): (k, v) for k, v in penalties.items()}

        for dim in self.trust_score.sub_scores.keys():
            expected_penalty_key = dim_to_penalty_key.get(dim, dim.title())

            if expected_penalty_key.lower() in penalties_lower:
                actual_key, _ = penalties_lower[expected_penalty_key.lower()]
                reasons.append({"status": "fail", "message": f"{actual_key} penalty applied"})
                # Map back to our recommendation keys
                rec_key = "fairness" if dim == "bias" else dim
                if rec_key in rec_map:
                    recommendations.append(rec_map[rec_key])
            else:
                reasons.append(
                    {"status": "pass", "message": f"{expected_penalty_key} assessment completed"}
                )

        if not recommendations:
            recommendations.append("Model meets all trustworthiness criteria for deployment.")

        # Determine primary risk
        primary_risk = None
        if penalties:
            # Largest penalty
            matched_key = max(penalties.items(), key=lambda x: x[1])[0]
            primary_risk = {"metric": matched_key, "value": penalties[matched_key]}
        elif ts.sub_scores:
            # Lowest sub-score
            lowest_dim = min(ts.sub_scores.items(), key=lambda x: x[1])[0]
            primary_risk = {
                "metric": dim_to_penalty_key.get(lowest_dim, lowest_dim.title()),
                "value": ts.sub_scores[lowest_dim],
            }

        return {
            "verdict": verdict,
            "reasons": reasons,
            "primary_risk": primary_risk,
            "recommendations": recommendations,
        }

    @property
    def deployment_summary(self) -> str:
        """Format the deployment explanation into a human-readable string."""
        exp = self.deployment_explanation
        lines = [f"Deployment Verdict: {exp['verdict']}", "", "Reasons:"]
        for reason in exp["reasons"]:
            icon = "✗" if reason["status"] == "fail" else "✓"
            lines.append(f"{icon} {reason['message']}")

        if exp["primary_risk"]:
            lines.extend(["", "Primary Risk:", str(exp["primary_risk"]["metric"])])

        lines.extend(["", "Recommendations:"])
        for rec in exp["recommendations"]:
            lines.append(f"• {rec}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_metadata(self) -> dict[str, Any]:
        """Collect run-level metadata."""
        from trustlens import __version__

        meta = {
            "trustlens_version": __version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_samples": int(len(self.y_true)),
            "n_classes": int(len(np.unique(self.y_true))),
            "model_class": type(self.model).__name__ if self.model is not None else "Manual",
            "modules_run": list(self.results.keys()),
        }

        if self.framework:
            meta["framework"] = self.framework

        if self.backend_metadata:
            meta["backend"] = self.backend_metadata

        return meta

    def _print_score_methodology(self) -> None:
        """Display the mathematical composition and notes section."""
        ts = self.trust_score
        print("\n[ SCORE METHODOLOGY ]")
        weights_str = " + ".join(
            [f"{k.capitalize()} ({int(v * 100)}%)" for k, v in ts.weights_used.items()]
        )
        print(f"  Formula     : {weights_str}")
        print("  Definitions :")
        print("    - Failure Score     : Reflects confidence-weighted errors, not raw error rate.")
        print(
            "    - Calibration       : Measures probability reliability via Expected Calibration Error (ECE)."
        )
        print("    - Fairness Margin   : Distance from the acceptable disparity threshold (0.10).")
        print("    - Penalties         : Deductions applied for critical diagnostic risks.")

    def _max_confidence(self) -> np.ndarray:
        """Return per-sample max predicted confidence."""
        if self.y_prob is None:
            return np.zeros(len(self.y_true))
        yp = np.asarray(self.y_prob)
        return cast(np.ndarray, yp.max(axis=1) if yp.ndim == 2 else yp)

    # ------------------------------------------------------------------
    # show()
    # ------------------------------------------------------------------

    def show(self, verbose: bool = False) -> None:
        """
        Print a rich, structured summary of all analysis results to stdout.

        Displays the Trust Score prominently at the top, followed by
        key insights and then delimited per-module metric summaries.
        """
        print("\nTrustLens Analysis Report")
        print(f"Timestamp : {self.metadata['timestamp']}")
        print(f"Model     : {self.metadata['model_class']}")
        print(f"Samples   : {self.metadata['n_samples']:,}")
        print(f"Classes   : {self.metadata['n_classes']}")

        # Trust Score section
        ts = self.trust_score
        print(f"\nTRUST SCORE: {ts.score}/100 [{ts.grade}]")
        print(f"Assessment : {ts.verdict}")

        if getattr(ts, "penalties_applied", None):
            print("\nScore Summary:")
            print(f"  Base Score        : {ts.base_score}")
            penalties_str = ", ".join([f"{k} (-{v})" for k, v in ts.penalties_applied.items()])
            print(
                f"  Penalties Applied : -{sum(ts.penalties_applied.values()):.1f} [{penalties_str}]"
            )
            print(f"  Final Score       : {ts.score}")

        explanation = self._format_score_explanation()
        if explanation:
            print()
            for line in explanation:
                print(line)

        print(f"\n{self.deployment_summary}")

        # Print Key Observations/Insights
        print("\nKey Observations:")
        insights = self._generate_insights()
        if not insights:
            print("- No critical issues found.")
        for insight in insights:
            print(f"- {insight}")

        print("\nDimension Breakdown:")
        for dim, score in ts.sub_scores.items():
            print(f"- {dim.capitalize() + ' Score':<18}: {score:5.1f}/100")

        for module_name, module_data in self.results.items():
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                self._print_module(module_data, indent=0, verbose=verbose)
            out = f.getvalue().strip()
            if out:
                print(f"\n{module_name.title()} Analysis")
                print(out)

        conclusion = self._generate_conclusion()
        print(f"\nConclusion:\n{conclusion}")

        # Methodology section at the end
        self._print_score_methodology()
        print()

    def _generate_text_report(self, verbose: bool = False) -> str:
        """
        Generate a clean, human-readable text report without ANSI colors.
        Mirroring the structure of show().
        """
        lines = []
        lines.append("TrustLens Analysis Report")
        lines.append(f"Timestamp : {self.metadata['timestamp']}")
        lines.append(f"Model     : {self.metadata['model_class']}")
        lines.append(f"Samples   : {self.metadata['n_samples']:,}")
        lines.append(f"Classes   : {self.metadata['n_classes']}")

        ts = self.trust_score
        lines.append(f"\nTRUST SCORE: {ts.score}/100 [{ts.grade}]")
        lines.append(f"Assessment : {ts.verdict}")
        if getattr(ts, "penalties_applied", None):
            lines.append("\nScore Summary:")
            lines.append(f"  Base Score        : {ts.base_score}")
            penalties_str = ", ".join([f"{k} (-{v})" for k, v in ts.penalties_applied.items()])
            lines.append(
                f"  Penalties Applied : -{sum(ts.penalties_applied.values()):.1f} [{penalties_str}]"
            )
            lines.append(f"  Final Score       : {ts.score}")
        explanation = self._format_score_explanation()
        if explanation:
            lines.append("")
            lines.extend(explanation)

        lines.append("")
        lines.extend(self.deployment_summary.split("\n"))

        lines.append("\nKey Observations:")
        insights = self._generate_insights()
        if not insights:
            lines.append("- No critical issues found.")
        else:
            for insight in insights:
                lines.append(f"- {insight}")

        lines.append("\nDimension Breakdown:")
        for dim, score in ts.sub_scores.items():
            lines.append(f"- {dim.capitalize() + ' Score':<18}: {score:5.1f}/100")

        for module_name, module_data in self.results.items():
            line_buf: list[str] = []
            self._get_module_text_lines(module_data, line_buf, indent=0, verbose=verbose)
            if line_buf:
                lines.append(f"\n{module_name.title()} Analysis")
                lines.extend(line_buf)

        lines.append(f"\nConclusion:\n{self._generate_conclusion()}")

        # Text methodology lines
        lines.append("\n[ SCORE METHODOLOGY ]")
        weights_str = " + ".join(
            [f"{k.capitalize()} ({int(v * 100)}%)" for k, v in ts.weights_used.items()]
        )
        lines.append(f"  Formula     : {weights_str}")
        lines.append("  Definitions :")
        lines.append(
            "    - Failure Score     : Reflects confidence-weighted errors, not raw error rate."
        )
        lines.append(
            "    - Calibration       : Measures probability reliability via Expected Calibration Error (ECE)."
        )
        lines.append(
            "    - Fairness Margin   : Distance from the acceptable disparity threshold (0.10)."
        )
        lines.append("    - Penalties         : Deductions applied for critical diagnostic risks.")
        return "\n".join(lines)

    def _get_module_text_lines(
        self, data: Any, buf: list[str], indent: int = 0, verbose: bool = False
    ) -> None:
        """Helper for _generate_text_report to recursively collect lines."""
        prefix = " " * indent
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and key.startswith("__") and key.endswith("__"):
                    continue
                display_key = str(key).replace("_", " ").title()

                if isinstance(value, dict):
                    if verbose:
                        buf.append(f"{prefix}- {display_key}:")
                        self._get_module_text_lines(value, buf, indent + 2, verbose)
                elif isinstance(value, (list, np.ndarray, tuple)):
                    if verbose:
                        buf.append(
                            f"{prefix}- {display_key}: [data structure of size {len(value)}]"
                        )
                elif isinstance(value, float):
                    buf.append(f"{prefix}- {display_key}: {value:.4f}")
                else:
                    buf.append(f"{prefix}- {display_key}: {value}")
        else:
            if verbose:
                buf.append(f"{prefix}- {data}")

    def _generate_conclusion(self) -> str:
        """Generate a short 1-2 line conclusion based on the scores."""
        failure_score = self.trust_score.sub_scores.get("failure", 100.0)
        ece = self.results.get("calibration", {}).get("ece", 0.0)
        conf_gap = self.results.get("failure", {}).get("confidence_gap", {}).get("gap", 0.0)

        # Cross-dimension pattern check
        is_confidently_wrong = failure_score < 50 and ece > 0.15 and conf_gap < 0.05

        # Fairness risk check
        bias_has_severe_violation = False
        bias_module = self.results.get("bias", {})
        for feat_data in bias_module.get("subgroup_performance", {}).values():
            if feat_data.get("__summary__", {}).get("performance_gap", 0.0) > 0.15:
                bias_has_severe_violation = True
                break
        if not bias_has_severe_violation:
            for val in bias_module.get("equalized_odds", {}).values():
                if not isinstance(val, dict):
                    continue
                summary = val.get("__summary__", {})
                if (
                    summary.get("tpr_violation") == "severe"
                    or summary.get("fpr_violation") == "severe"
                ):
                    bias_has_severe_violation = True
                    break

        if is_confidently_wrong:
            return (
                "Model exhibits 'confidently wrong' behavior and high failure risk. Do not deploy."
            )
        if failure_score < 40:
            return "Model shows high failure risk and is not ready for deployment."
        if bias_has_severe_violation:
            return "Model exhibits severe fairness violations and is not ready for deployment."
        if ece > 0.1:
            return "Model requires calibration before deployment."

        grade = self.trust_score.grade
        if grade == "A":
            return "Model demonstrates strong reliability across all measured dimensions. Ready for production."
        elif grade == "B":
            return "Model is generally reliable, but minor issues should be addressed before broad deployment."
        elif grade == "C":
            return "Model shows moderate risk. Investigate flagged dimensions (e.g., calibration or bias) before proceeding."
        else:
            return "Model exhibits critical issues and should not be deployed until fundamental problems are resolved."

    def _generate_insights(self) -> list[str]:
        """Generate plain-text insights based on results."""
        insight_list = []

        def add_insight(msg: str, priority: int):
            insight_list.append((priority, msg))

        # Surfaced Patterns
        if self.patterns:
            pattern_lines = [f"  - {p}" for p in self.patterns]
            pattern_msg = "Patterns Detected:\n" + "\n".join(pattern_lines)
            add_insight(pattern_msg, 2)

        # Core signals

        failure_score = self.trust_score.sub_scores.get("failure", 100.0)
        conf_gap = self.results.get("failure", {}).get("confidence_gap", {}).get("gap", 0.0)
        silhouette = (
            self.results.get("representation", {})
            .get("separability", {})
            .get("silhouette_score", 0.0)
        )

        # Legacy pattern checks removed. Patterns are now sourced from self.patterns.
        is_confidently_wrong = "Confidently Wrong" in self.patterns

        # Pattern: Generalization Risk
        if silhouette < 0 and failure_score < 50:
            add_insight(
                "⚠ Warning: Poor latent representation correlates with high failure risk.\n    → The network struggles to differentiate classes; investigate feature quality.",
                2,
            )

        cal_score = self.trust_score.sub_scores.get("calibration", 100.0)

        # Check Calibration
        if "calibration" in self.results:
            if not is_confidently_wrong:
                if cal_score < 75:
                    add_insight(
                        "Critical: Calibration is poor (score < 75).\n    → Consider temperature scaling or isotonic regression.",
                        1,
                    )
                elif cal_score < 90:
                    add_insight(
                        "Warning: Calibration is acceptable (score 75-89), but could be improved.",
                        2,
                    )
                else:
                    add_insight("ℹ Info: Calibration quality is excellent (score 90+).", 3)

        # Check Failure
        failure_module = self.results.get("failure", {})
        error_rate = (
            failure_module.get("misclassification_summary", {})
            .get("__overall__", {})
            .get("overall_error_rate", 0.0)
        )
        error_pct = int(error_rate * 100) if error_rate is not None else 0

        avg_err_conf = failure_module.get("confidence_gap", {}).get(
            "incorrect_confidence_mean", 0.0
        )
        conf_str = (
            f"~{avg_err_conf:.2f} confidence" if avg_err_conf > 0 else "confidence-weighted error"
        )

        if not is_confidently_wrong:
            if failure_score < 40:
                add_insight(
                    f"Critical: High failure risk detected ({error_pct}% error rate).\n    → Heavily penalized because errors are dangerously concentrated in the {conf_str} range.",
                    1,
                )
            elif failure_score < 60:
                add_insight(
                    f"Warning: Moderate failure risk ({error_pct}% error rate).\n    → Model exhibits concerning confidence (~{avg_err_conf:.2f}) on incorrect predictions.",
                    2,
                )

        if "failure" in self.results and not is_confidently_wrong:
            if conf_gap < 0.05:
                add_insight(
                    "Warning: Model is overconfident on incorrect predictions (low confidence gap).",
                    2,
                )

        # Check Bias
        if "bias" in self.results:
            bias_module = self.results["bias"]
            ratio = bias_module.get("class_imbalance", {}).get("imbalance_ratio", 1.0)
            if ratio > 5.0:
                add_insight(
                    "Warning: Severe class imbalance may affect performance.\n    → Consider rebalancing or fairness constraints.",
                    2,
                )

            subgroups = bias_module.get("subgroup_performance", {})
            for feat_name, feat_data in subgroups.items():
                gap = feat_data.get("__summary__", {}).get("performance_gap", 0.0)
                if gap > 0.1:
                    add_insight(
                        f"Warning: Significant performance gap detected across {feat_name}.\n    → Investigate subgroup disparities.",
                        2,
                    )

            eq_odds = bias_module.get("equalized_odds", {})
            has_eq_odds_skipped = eq_odds.get("status") == "skipped"

            has_severe = False
            if has_eq_odds_skipped:
                add_insight(
                    "ℹ Info: Fairness analysis skipped due to insufficient subgroup diversity (or non-binary target).",
                    3,
                )
            elif eq_odds:
                for k, val in eq_odds.items():
                    if isinstance(val, dict) and k not in ("status", "reason", "details"):
                        summary = val.get("__summary__", {})
                        if (
                            summary.get("tpr_violation") == "severe"
                            or summary.get("fpr_violation") == "severe"
                        ):
                            has_severe = True
                if has_severe:
                    add_insight(
                        "Critical: Severe fairness disparity detected between subgroups.\n    → Investigate subgroup disparities and consider rebalancing.",
                        1,
                    )

            if not has_severe and not has_eq_odds_skipped:
                # Check if gap from subgroup is low and no class imbalance severity.
                if ratio <= 5.0:
                    max_gap = 0.0
                    for feat_data in subgroups.values():
                        gap = feat_data.get("__summary__", {}).get("performance_gap", 0.0)
                        if gap > max_gap:
                            max_gap = gap
                    if max_gap <= 0.1:
                        margin = 0.1 - max_gap
                        has_penalty = "Fairness" in getattr(
                            self.trust_score, "penalties_applied", {}
                        )
                        if has_penalty:
                            if margin < 0.01:
                                msg = "ℹ Info: At threshold boundary (0.00 margin from 0.10 limit). Minimal penalty applied."
                            else:
                                msg = f"ℹ Info: Minor fairness variations detected (margin: {margin:.2f} from 0.10 limit). Small penalty applied."
                            add_insight(msg, 3)
                        else:
                            add_insight(
                                f"ℹ Info: No bias detected (margin: {margin:.2f} from 0.10 limit).",
                                3,
                            )

        # Sort by priority, then deduplicate while preserving order
        insight_list.sort(key=lambda x: x[0])
        seen = set()
        final_insights = []
        for _, msg in insight_list:
            if msg not in seen:
                seen.add(msg)
                final_insights.append(msg)

        return final_insights

    def _print_module(self, data: Any, indent: int = 0, verbose: bool = False) -> None:
        """Recursively pretty-print a module's result dictionary."""
        prefix = " " * indent
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and key.startswith("__") and key.endswith("__"):
                    continue
                display_key = str(key).replace("_", " ").title()

                if isinstance(value, dict):
                    if verbose:
                        print(f"{prefix}- {display_key}:")
                        self._print_module(value, indent + 2, verbose)
                elif isinstance(value, (list, np.ndarray, tuple)):
                    if verbose:
                        print(f"{prefix}- {display_key}: [data structure of size {len(value)}]")
                elif isinstance(value, float):
                    print(f"{prefix}- {display_key}: {value:.4f}")
                else:
                    print(f"{prefix}- {display_key}: {value}")
        else:
            if verbose:
                print(f"{prefix}- {data}")

    # ------------------------------------------------------------------
    # summary_plot() ← THE WOW FEATURE
    # ------------------------------------------------------------------

    def summary_plot(
        self,
        save_path: str | None = None,
        show: bool = True,
    ):
        """
        Render the TrustLens Summary Dashboard — a single-figure overview
        of the model's trustworthiness.

        Layout (2×3 grid):

          Trust Score Gauge Reliability Diag  Confidence Gap

          Error Rate Dist.  Class Dist.    Sub-score Bars


        Parameters
        ----------
        save_path : str, optional
          If provided, saves the figure to this path (PNG or PDF).
        show : bool
          If True, calls ``plt.show()`` for interactive display.
          Default True. Set to False in non-interactive environments.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from trustlens.visualization.summary_plot import plot_summary_dashboard

        fig = plot_summary_dashboard(
            trust_score=self.trust_score,
            results=self.results,
            y_true=self.y_true,
            y_pred=self.y_pred,
            y_prob=self.y_prob,
            model_name=self.metadata["model_class"],
            save_path=save_path,
        )
        if show:
            try:
                import matplotlib.pyplot as plt

                if "agg" not in plt.get_backend().lower():
                    plt.show()
            except Exception:
                pass

        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass

        return fig

    # ------------------------------------------------------------------
    # show_failures()
    # ------------------------------------------------------------------

    def show_failures(
        self,
        top_k: int = 10,
        images: np.ndarray | None = None,
        feature_names: list | None = None,
        save_path: str | None = None,
    ) -> None:
        """
        Display the most alarming model failures — high-confidence wrong
        predictions that deserve immediate attention.

        For each failure, reports:
        * Predicted class and confidence level
        * True class
        * A "danger rating" based on confidence level
        * Feature values (if ``feature_names`` provided)

        Parameters
        ----------
        top_k : int
          Number of top failures to show. Default 10.
        images : np.ndarray, optional
          Image array shape (n_samples, H, W, C) or (n_samples, H, W).
          If provided, renders a grid of the most-confident wrong predictions.
        feature_names : list[str], optional
          Column names for tabular features in ``self.X``.
        save_path : str, optional
          If provided and ``images`` is given, saves the failure grid as PNG.

        Examples
        --------
        >>> report.show_failures(top_k=10)
        >>> report.show_failures(top_k=5, images=X_images)
        """
        max_conf = self._max_confidence()
        y_true = np.asarray(self.y_true)
        y_pred = np.asarray(self.y_pred)

        # Identify wrong predictions
        wrong_mask = y_true != y_pred
        if not wrong_mask.any():
            print(" No misclassifications found — perfect predictions!")
            return

        wrong_indices = np.where(wrong_mask)[0]
        wrong_confidence = max_conf[wrong_mask]

        # Sort by confidence descending (worst offenders first)
        sorted_order = np.argsort(wrong_confidence)[::-1]
        top_indices = wrong_indices[sorted_order[:top_k]]

        print("\nCRITICAL FAILURES")
        print(
            f"{self.metadata['model_class']} | "
            f"{wrong_mask.sum()} total errors / "
            f"{len(y_true)} samples ({100 * wrong_mask.mean():.1f}%)"
        )
        print(f"\n{'#':<4} {'Sample':<8} {'True':>6} {'Pred':>6} {'Confidence':>12} {'Danger':>8}")

        for rank, idx in enumerate(top_indices, start=1):
            conf = float(max_conf[idx])
            true_cls = int(y_true[idx])
            pred_cls = int(y_pred[idx])
            danger = _danger_rating(conf)
            print(f"{rank:<4} {idx:<8} {true_cls:>6} {pred_cls:>6} {conf:>11.1%} {danger:>8}")

            # Show top features if names provided
            if feature_names is not None:
                feats = np.asarray(self.X)[idx]
                top_feat_idx = np.argsort(np.abs(feats))[::-1][:3]
                feat_strs = [
                    f"{feature_names[i]}={feats[i]:.3g}"
                    for i in top_feat_idx
                    if i < len(feature_names)
                ]
                if feat_strs:
                    print(f"    Top features: {', '.join(feat_strs)}")

        # Summary insight
        top_conf = max_conf[top_indices]
        print("\n Insights:")
        print(f"   Mean confidence on top failures: {top_conf.mean():.1%}")
        print("   These are high-confidence mistakes - the model is")
        print("   certain it is right, but it is wrong.")
        if top_conf.mean() > 0.85:
            print("   Overconfidence detected - consider calibration.")
        print()

        # Optional: image grid
        if images is not None:
            fig = _plot_failure_grid(
                images=images,
                indices=top_indices,
                y_true=y_true,
                y_pred=y_pred,
                confidences=max_conf,
                save_path=save_path,
            )
            _ = fig

    # ------------------------------------------------------------------
    # plot()
    # ------------------------------------------------------------------

    def plot(
        self,
        module: str | None = None,
        save_dir: str | None = None,
    ) -> None:
        """
        Render per-module visualisations.

        Parameters
        ----------
        module : str, optional
          Which module to plot (e.g., ``"calibration"``).
          If None, all available modules are plotted.
        save_dir : str, optional
          Directory path where figures are saved as PNG files.
        """
        from trustlens.visualization import plot_module

        modules_to_plot = [module] if module else list(self.results.keys())
        for m in modules_to_plot:
            if m in self.results:
                plot_module(
                    module_name=m,
                    data=self.results[m],
                    save_dir=save_dir,
                    embeddings=self.embeddings if m == "representation" else None,
                    labels=self.y_true if m == "representation" else None,
                )
            else:
                logger.warning("Module '%s' not found in results.", m)

    # ------------------------------------------------------------------
    # plot_embedding_2d()
    # ------------------------------------------------------------------

    def plot_embedding_2d(
        self,
        method: str = "umap",
        n_max: int = 5000,
        save_path: str | None = None,
        show: bool = True,
    ):
        """
        Project stored embeddings to 2D and render a class-colored scatter plot.

        Delegates to the underlying ``plot_embedding_2d`` in the visualization
        sub-package, forwarding silhouette score (when available) so the plot
        is annotated automatically.

        Guarantees:
        - Returns matplotlib.figure.Figure
        - Raises ValueError when embeddings were not supplied to ``analyze()``
        - Projection falls back gracefully (UMAP -> t-SNE -> PCA) when optional libraries are missing

        Parameters
        ----------
        method : str
          Projection algorithm: ``"umap"`` (default), ``"tsne"``, or ``"pca"``.
        n_max : int
          Max samples plotted.  Subsampled randomly when the embedding matrix
          exceeds this limit.
        save_path : str, optional
          File path to save the figure (e.g. ``"clusters.png"``).
        show : bool
          Whether to display the figure interactively.  Default True.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If embeddings are not available (i.e. not passed to ``analyze()``).
        """
        from trustlens.visualization.representation_plots import (
            plot_embedding_2d as _plot,
        )

        if self.embeddings is None:
            raise ValueError(
                "No embeddings available. "
                "Pass 'embeddings' to analyze() to enable 2D embedding visualization."
            )

        sil = self.results.get("representation", {}).get("separability", {}).get("silhouette_score")

        return _plot(
            embeddings=self.embeddings,
            labels=self.y_true,
            silhouette_score=sil,
            method=method,
            n_max=n_max,
            save_path=save_path,
            show=show,
        )

    # ------------------------------------------------------------------
    # plot_bias()
    # ------------------------------------------------------------------

    def plot_bias(
        self,
        mode: str = "summary",
        show: bool = True,
        save_path: str | None = None,
        multi_feature: bool = False,
    ):
        """
        Generate fairness/bias visualizations from report results.

        ``mode`` and ``multi_feature`` are independent dimensions; the return
        shape is fully determined by their combination, with no flattening.

        **Guarantees**

        - Return shape is fixed by the ``(mode, multi_feature)`` combination
          (see the table below). The structure never collapses across calls.
        - Multi-feature outputs never contain ``None`` values; missing
          components are represented by empty dicts ``{}``.
        - Raises ``ValueError`` for invalid ``mode`` or unusable data, regardless
          of ``multi_feature``.

        Return shape by ``(mode, multi_feature)``:

        +-----------------+----------------+--------------------------------------+
        | mode            | multi_feature  | Return type                          |
        +=================+================+======================================+
        | single mode     | False          | ``Figure``                           |
        | (summary,       |                |                                      |
        | subgroup,       |                |                                      |
        | equalized_odds, |                |                                      |
        | gap)            |                |                                      |
        +-----------------+----------------+--------------------------------------+
        | summary         | True           | ``Figure`` (same as ``False``)       |
        +-----------------+----------------+--------------------------------------+
        | subgroup,       | True           | ``dict[str, Figure]`` keyed by       |
        | equalized_odds, |                | feature name (sorted)                |
        | gap             |                |                                      |
        +-----------------+----------------+--------------------------------------+
        | all             | False          | ``dict[str, Figure | None]`` keyed   |
        |                 |                | by mode (existing behavior)          |
        +-----------------+----------------+--------------------------------------+
        | all             | True           | ``dict[str, dict[str, Figure]]``     |
        |                 |                | keyed by mode, then feature.         |
        |                 |                | Always contains exactly the keys     |
        |                 |                | 'subgroup', 'equalized_odds',        |
        |                 |                | 'gap' (in that order). Components    |
        |                 |                | with no data return ``{}``.          |
        +-----------------+----------------+--------------------------------------+

        Returned dict for ``mode="all"`` (regardless of ``multi_feature``)
        ALWAYS contains exactly three keys in order: ``'subgroup'`` ->
        ``'equalized_odds'`` -> ``'gap'``. Within each multi-feature dict,
        feature order follows ``sorted(feature_names)`` for determinism.

        Note: ``mode="all"`` returns a structured dict and does NOT display
        figures unless ``show=True``.

        Parameters
        ----------
        mode : str, optional
            Visualization mode. One of {"summary", "all", "subgroup",
            "equalized_odds", "gap"}. Default "summary".
        show : bool
            Whether to display the figure interactively. Default True.
        save_path : str, optional
            If provided, saves the figure(s) to this path. Only honored when
            ``multi_feature=False`` (single-feature behavior).
            - Single modes: Treated as full file path. Defaults to ``.png`` if extension missing.
            - ``mode="all"``: Treated as base name. Appends suffixes and ``.png``.
            When ``multi_feature=True``, ``save_path`` is ignored (per-feature
            saving is intentionally not exposed here; use the lower-level
            ``plot_*_multi`` helpers if you need on-disk output).
        multi_feature : bool, optional
            If True, return per-feature figures rather than only the first
            sensitive feature. Default False (backward compatible).

        Returns
        -------
        matplotlib.figure.Figure | dict
            The return type depends on the ``(mode, multi_feature)`` combination.
            Possible shapes:
            - ``Figure``
            - ``dict[str, Figure | None]``
            - ``dict[str, Figure]``
            - ``dict[str, dict[str, Figure]]``
            See the return-shape table above.

        Notes
        -----
        New modes can be added by extending ``ALLOWED_MODES`` and dispatch
        logic without breaking existing behavior.
        Function behavior MUST be deterministic given identical inputs (no
        randomness, no state mutation). This includes consistent plot ordering,
        consistent key ordering in dicts, and no randomness in visualization.
        Each returned Figure must be independent (no shared axes, state, or
        references between plots). Each plot must be independently renderable
        and savable.

        Caution: Figures are returned open. If calling this method repeatedly
        in a loop, ensure you call ``plt.close(fig)`` on the returned figures
        to avoid memory accumulation.

        Raises
        ------
        ValueError
            If ``mode`` is invalid, ``"bias"`` is not present in
            ``self.results``, or the data is unusable.
        """
        import matplotlib.pyplot as plt

        from trustlens.visualization import _plot_bias
        from trustlens.visualization.fairness import (
            plot_equalized_odds,
            plot_equalized_odds_multi,
            plot_fairness_gap,
            plot_fairness_gap_multi,
            plot_subgroup_performance,
            plot_subgroup_performance_multi,
        )

        ALLOWED_MODES = {"summary", "all", "subgroup", "equalized_odds", "gap"}
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Allowed: {ALLOWED_MODES}")

        if "bias" not in self.results:
            raise ValueError(
                "Bias results not available in report. "
                "Ensure 'bias' module was included in analyze()."
            )

        bias_data = self.results["bias"]

        # Validation: check for usable structure (valid, non-empty)
        has_subgroup = bool(bias_data.get("subgroup_performance"))
        has_eo = bool(bias_data.get("equalized_odds"))
        has_imbalance = bool(bias_data.get("class_imbalance"))

        if not (has_subgroup or has_eo or has_imbalance):
            raise ValueError("Bias data is present but not usable for visualization.")

        # Reserved meta keys that should never be treated as feature names.
        _META_KEYS = ("status", "reason", "details")

        def _get_first(key):
            d = bias_data.get(key, {})
            for k, v in d.items():
                if k not in _META_KEYS:
                    return k, v
            return None, None

        def _sorted_feature_dict(key):
            """Return ``{feature: data}`` ordered by ``sorted(feature_names)``.

            Drops reserved meta keys so wrapper functions iterate only over
            real features.
            """
            d = bias_data.get(key, {})
            return {fname: d[fname] for fname in sorted(k for k in d if k not in _META_KEYS)}

        def _get_save_path(base_path, suffix=None):
            if base_path is None:
                return None
            import os

            name, ext = os.path.splitext(base_path)
            if suffix:
                # mode="all": Strip extension if present, then append suffix and .png
                return f"{name}_{suffix}.png"

            # Single modes: If no extension -> append .png
            if not ext:
                ext = ".png"
            return f"{name}{ext}"

        # ------------------------------------------------------------------
        # multi_feature=True dispatch
        #
        # Implemented as a thin transformation layer over the same plotting
        # primitives the single-feature path uses. The "summary" mode simply
        # falls through to the existing single-figure path because a summary
        # plot is feature-agnostic by design.
        # ------------------------------------------------------------------
        if multi_feature and mode != "summary":
            if save_path is not None:
                logger.warning(
                    "save_path is ignored when multi_feature=True; "
                    "use plot_*_multi helpers directly for per-feature files."
                )

            if mode == "subgroup":
                feat_dict = _sorted_feature_dict("subgroup_performance")
                if not feat_dict:
                    raise ValueError("Missing 'subgroup_performance' data for 'subgroup' mode.")
                return plot_subgroup_performance_multi(feat_dict, show=show)

            if mode == "equalized_odds":
                feat_dict = _sorted_feature_dict("equalized_odds")
                if not feat_dict:
                    raise ValueError("Missing 'equalized_odds' data for 'equalized_odds' mode.")
                return plot_equalized_odds_multi(feat_dict, show=show)

            if mode == "gap":
                # Priority: subgroup_performance -> equalized_odds, mirroring
                # the single-feature path. The gap plot internally uses the
                # equalized-odds-shaped data, so we use whichever is present.
                feat_dict = _sorted_feature_dict("subgroup_performance")
                if not feat_dict:
                    feat_dict = _sorted_feature_dict("equalized_odds")
                if not feat_dict:
                    raise ValueError(
                        "Missing sufficient data for 'gap' mode "
                        "(either subgroup_performance or equalized_odds)."
                    )
                return plot_fairness_gap_multi(feat_dict, show=show)

            if mode == "all":
                # Return shape is fixed: three keys, in this order, every
                # call. Components with no data map to {} (never None).
                multi_results: dict[str, dict[str, plt.Figure]] = {
                    "subgroup": {},
                    "equalized_odds": {},
                    "gap": {},
                }

                sub_dict = _sorted_feature_dict("subgroup_performance")
                eo_dict = _sorted_feature_dict("equalized_odds")

                if sub_dict:
                    multi_results["subgroup"] = plot_subgroup_performance_multi(
                        sub_dict, show=False
                    )

                if eo_dict:
                    multi_results["equalized_odds"] = plot_equalized_odds_multi(eo_dict, show=False)

                # 'gap' uses the same priority rule as the single-feature
                # path: subgroup_performance first, falling back to
                # equalized_odds.
                gap_dict = sub_dict if sub_dict else eo_dict
                if gap_dict:
                    multi_results["gap"] = plot_fairness_gap_multi(gap_dict, show=False)

                if show:
                    backend = plt.get_backend().lower()
                    if "agg" not in backend:
                        try:
                            plt.show()
                        except Exception:
                            pass

                return multi_results

        if mode == "summary":
            # "summary" mode requires data compatible with _plot_bias()
            fig = _plot_bias(bias_data)
            if fig is None:
                raise ValueError("Failed to generate summary bias plot.")
            if save_path:
                fig.savefig(_get_save_path(save_path), dpi=150, bbox_inches="tight")
            if show:
                backend = plt.get_backend().lower()
                if "agg" not in backend:
                    try:
                        plt.show()
                    except Exception:
                        pass
            return fig

        if mode == "subgroup":
            feat_name, feat_data = _get_first("subgroup_performance")
            if feat_data is None:
                raise ValueError("Missing 'subgroup_performance' data for 'subgroup' mode.")
            return plot_subgroup_performance(
                feat_data, feat_name, show=show, save_path=_get_save_path(save_path)
            )

        if mode == "equalized_odds":
            feat_name, feat_data = _get_first("equalized_odds")
            if feat_data is None:
                raise ValueError("Missing 'equalized_odds' data for 'equalized_odds' mode.")
            return plot_equalized_odds(
                feat_data, feat_name, show=show, save_path=_get_save_path(save_path)
            )

        if mode == "gap":
            # Priority: subgroup_performance > equalized_odds
            feat_name, feat_data = _get_first("subgroup_performance")
            if feat_data is None:
                feat_name, feat_data = _get_first("equalized_odds")

            if feat_data is None:
                raise ValueError(
                    "Missing sufficient data for 'gap' mode (either subgroup_performance or equalized_odds)."
                )
            return plot_fairness_gap(
                feat_data, feat_name, show=show, save_path=_get_save_path(save_path)
            )

        if mode == "all":
            results = {}

            # 1. Subgroup
            f_name_s, f_data_s = _get_first("subgroup_performance")
            fig_s = None
            if f_data_s:
                try:
                    fig_s = plot_subgroup_performance(
                        f_data_s,
                        f_name_s,
                        show=False,
                        save_path=_get_save_path(save_path, "subgroup"),
                    )
                except Exception:
                    fig_s = None
            results["subgroup"] = fig_s

            # 2. Equalized Odds
            f_name_eo, f_data_eo = _get_first("equalized_odds")
            fig_eo = None
            if f_data_eo:
                try:
                    fig_eo = plot_equalized_odds(
                        f_data_eo,
                        f_name_eo,
                        show=False,
                        save_path=_get_save_path(save_path, "equalized_odds"),
                    )
                except Exception:
                    fig_eo = None
            results["equalized_odds"] = fig_eo

            # 3. Gap
            fig_g = None
            # Re-use best available data for gap priority
            g_name, g_data = (f_name_s, f_data_s) if f_data_s else (f_name_eo, f_data_eo)
            if g_data:
                try:
                    fig_g = plot_fairness_gap(
                        g_data, g_name, show=False, save_path=_get_save_path(save_path, "gap")
                    )
                except Exception:
                    fig_g = None
            results["gap"] = fig_g

            if all(v is None for v in results.values()):
                raise ValueError("Failed to generate any bias plots in 'all' mode.")

            if show:
                backend = plt.get_backend().lower()
                if "agg" not in backend:
                    try:
                        plt.show()
                    except Exception:
                        pass

            return results

    # ------------------------------------------------------------------
    # save()
    # ------------------------------------------------------------------

    def save(self, path: str = "trust_report", **kwargs) -> Path:
        """
        Save the analysis report.

        If ``path`` ends with '.json' or '.txt', saves a single file.
        Otherwise, treats ``path`` as a directory and saves a full report
        bundle (JSON, metadata, plots).

        Parameters
        ----------
        path : str
          Target file path (e.g., "report.json") or directory path.
        **kwargs : Any
          Backward compatibility for ``directory`` argument.

        Returns
        -------
        Path
          Resolved path to the saved file or directory.
        """
        if "directory" in kwargs:
            path = kwargs.pop("directory")

        p = Path(path).resolve()

        # 1. Single-file JSON export
        if path.lower().endswith(".json"):
            p.parent.mkdir(parents=True, exist_ok=True)
            # Unified structure for single-file artifact
            data = {
                "results": self._to_serializable(self.results),
                "metadata": self.metadata,
                "trust_score": self.trust_score.score,
                "grade": self.trust_score.grade,
                "sub_scores": self.trust_score.sub_scores,
                "deployment_explanation": self.deployment_explanation,
            }
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info("Unified Report JSON saved to: %s", p)
            return p

        # 2. Single-file TXT export
        if path.lower().endswith(".txt"):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(self._generate_text_report(), encoding="utf-8")
            logger.info("Report TXT saved to: %s", p)
            return p

        # 3. Directory bundle export (Original behavior)
        out_dir = p
        out_dir.mkdir(parents=True, exist_ok=True)

        # Serialize metrics
        (out_dir / "report.json").write_text(
            json.dumps(self._to_serializable(self.results), indent=2),
            encoding="utf-8",
        )

        # Serialize metadata
        (out_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2),
            encoding="utf-8",
        )

        # Serialize trust score
        ts = self.trust_score
        (out_dir / "trust_score.json").write_text(
            json.dumps(
                {
                    "score": ts.score,
                    "grade": ts.grade,
                    "verdict": ts.verdict,
                    "sub_scores": ts.sub_scores,
                    "weights_used": ts.weights_used,
                    "breakdown": ts.breakdown,
                    "deployment_explanation": self.deployment_explanation,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Save summary plot
        try:
            self.summary_plot(
                save_path=str(out_dir / "summary_plot.png"),
                show=False,
            )
        except Exception as exc:
            logger.warning("Summary plot skipped: %s", exc)

        # Save per-module plots
        try:
            self.plot(save_dir=str(out_dir))
        except Exception as exc:
            logger.warning("Plot generation skipped: %s", exc)

        logger.info("Report bundle saved to: %s", out_dir)
        return out_dir

    # ------------------------------------------------------------------
    # to_dict()
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return all results as a flat, JSON-serializable dictionary.

        Useful for logging to MLflow, W&B, or any experiment tracker.

        Returns
        -------
        dict
          Flat dict with keys like ``"calibration.brier_score"``.
        """
        from trustlens.utils import flatten_dict

        flat = flatten_dict(self._to_serializable(self.results))
        flat["trust_score"] = self.trust_score.score
        flat["trust_grade"] = self.trust_score.grade
        flat["framework"] = self.metadata.get("framework", "unknown")
        flat["trustlens_version"] = self.metadata["trustlens_version"]

        # Flatten deployment explanation for MLflow/W&B tracking
        exp = self.deployment_explanation
        flat["deployment_verdict"] = exp["verdict"]
        flat["deployment_primary_risk_metric"] = (
            exp["primary_risk"].get("metric") if exp["primary_risk"] else None
        )
        flat["deployment_primary_risk_value"] = (
            exp["primary_risk"].get("value") if exp["primary_risk"] else None
        )

        for dim, score in self.trust_score.sub_scores.items():
            flat[f"trust_{dim}_score"] = score
        return flat

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _to_serializable(self, obj: Any) -> Any:
        """Recursively convert numpy / non-JSON-native types."""
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    def __repr__(self) -> str:
        modules_str = ", ".join(self.results.keys())
        return (
            f"TrustReport(model={self.metadata['model_class']!r}, "
            f"score={self.trust_score.score}/100 [{self.trust_score.grade}], "
            f"samples={self.metadata['n_samples']}, "
            f"modules=[{modules_str}])"
        )

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        import base64
        import io

        import matplotlib.pyplot as plt

        from trustlens.visualization.summary_plot import _C, _color_for_grade

        # 1. Generate the summary plot into a buffer
        fig = self.summary_plot(show=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")

        # 2. Build the HTML wrapper
        ts = self.trust_score
        gc = _color_for_grade(ts.grade)

        html = f"""
        <div style="font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    max-width: 900px; padding: 25px; border-radius: 16px;
                    border: 1px solid #e0e0e0; background-color: #ffffff;
                    box-shadow: 0 8px 24px rgba(0,0,0,0.06); margin: 15px 0;">

            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 25px;">
                <div>
                    <h2 style="margin: 0; color: {_C["dark"]}; font-size: 24px; font-weight: 800;">TrustLens Analysis Report</h2>
                    <div style="font-size: 14px; color: {_C["gray"]}; margin-top: 4px;">
                        {self.metadata["model_class"]} &bull; {self.metadata["n_samples"]:,} samples &bull; {self.metadata["timestamp"][:19].replace("T", " ")}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 42px; font-weight: 800; color: {gc}; line-height: 1;">{ts.score}<span style="font-size: 18px; color: {_C["gray"]}; font-weight: 600;">/100</span></div>
                    <div style="font-size: 14px; font-weight: 700; color: {gc}; text-transform: uppercase;">Grade {ts.grade}</div>
                </div>
            </div>

            <div style="background-color: #f8f9fa; border-radius: 12px; padding: 15px; margin-bottom: 25px; border-left: 5px solid {gc};">
                <div style="font-size: 15px; font-weight: 600; color: {_C["dark"]}; margin-bottom: 5px;">Overall Assessment</div>
                <div style="font-size: 14px; color: #444;">{ts.verdict}</div>
            </div>
        """

        exp = self.deployment_explanation
        fail_icon = "<span style='color: #d32f2f; font-weight: bold;'>✗</span>"
        pass_icon = "<span style='color: #2e7d32; font-weight: bold;'>✓</span>"

        reasons_html = "".join(
            [
                f'<li style="margin-bottom: 4px;">{fail_icon if r["status"] == "fail" else pass_icon} {r["message"]}</li>'
                for r in exp["reasons"]
            ]
        )
        pr_html = (
            f'<div style="font-size: 13px; font-weight: 700; color: {_C["gray"]}; margin-top: 15px; margin-bottom: 4px; text-transform: uppercase;">Primary Risk</div><div style="font-size: 14px; color: #d32f2f; font-weight: 600;">{exp["primary_risk"]["metric"]}</div>'
            if exp["primary_risk"]
            else ""
        )
        recs_html = "".join(
            [f'<li style="margin-bottom: 4px;">{r}</li>' for r in exp["recommendations"]]
        )

        html += f"""
            <div style="background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 12px; padding: 15px; margin-bottom: 25px;">
                <div style="font-size: 15px; font-weight: 600; color: {_C["dark"]}; margin-bottom: 8px;">Deployment Verdict: <span style="font-weight: 700;">{exp["verdict"]}</span></div>
                {pr_html}
                <div style="display: flex; gap: 30px; margin-top: 15px; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 200px;">
                        <div style="font-size: 13px; font-weight: 700; color: {_C["gray"]}; margin-bottom: 8px; text-transform: uppercase;">Reasons</div>
                        <ul style="margin: 0; padding-left: 0; list-style-type: none; font-size: 13.5px; color: #333; line-height: 1.5;">
                            {reasons_html}
                        </ul>
                    </div>
                    <div style="flex: 1; min-width: 250px;">
                        <div style="font-size: 13px; font-weight: 700; color: {_C["gray"]}; margin-bottom: 8px; text-transform: uppercase;">Recommendations</div>
                        <ul style="margin: 0; padding-left: 20px; font-size: 13.5px; color: #333; line-height: 1.5;">
                            {recs_html}
                        </ul>
                    </div>
                </div>
            </div>

            <div style="margin-bottom: 20px;">
                <img src="data:image/png;base64,{data}" style="width: 100%; border-radius: 8px; border: 1px solid #f0f0f0;" />
            </div>

            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 250px;">
                    <div style="font-size: 13px; font-weight: 700; color: {_C["gray"]}; margin-bottom: 12px; text-transform: uppercase;">Key Observations</div>
                    <ul style="margin: 0; padding-left: 20px; font-size: 13.5px; color: #333; line-height: 1.6;">
        """

        insights = self._generate_insights()
        if not insights:
            html += "<li>No critical issues found.</li>"
        else:
            for insight in insights:
                html += f"<li>{insight}</li>"

        html += f"""
                    </ul>
                </div>
            </div>

            <div style="margin-top: 25px; pt: 15px; border-top: 1px solid #f0f0f0; text-align: right;">
                <span style="font-size: 12px; color: #aaa;">Generated by TrustLens v{__version__}</span>
            </div>
        </div>
        """
        return html


# ---------------------------------------------------------------------------
# Failure display helpers
# ---------------------------------------------------------------------------


def _danger_rating(confidence: float) -> str:
    """Map confidence level to a danger string."""
    if confidence >= 0.95:
        return "CRITICAL"
    if confidence >= 0.85:
        return "HIGH"
    if confidence >= 0.70:
        return "MEDIUM"
    return "LOW"


def _plot_failure_grid(
    images: np.ndarray,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    save_path: str | None = None,
):
    """Render a grid of failure images with prediction annotations."""
    import matplotlib.pyplot as plt

    n = len(indices)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, idx in zip(axes, indices):
        img = images[idx]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)

        conf = confidences[idx]
        color = "#FF3B30" if conf >= 0.85 else "#FF9F0A"
        ax.set_title(
            f"True: {y_true[idx]} Pred: {y_pred[idx]}\nConf: {conf:.1%}",
            fontsize=9,
            color=color,
            fontweight="bold",
        )
        ax.axis("off")

    for ax in axes[len(indices) :]:
        ax.set_visible(False)

    fig.suptitle(
        "High-Confidence Failures",
        fontsize=13,
        fontweight="bold",
        color="#FF3B30",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig
