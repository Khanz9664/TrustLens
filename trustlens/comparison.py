"""
trustlens.comparison
====================
Utility for comparative analysis across multiple TrustReports.
"""

from trustlens.report import TrustReport


def compare(reports: list[TrustReport]) -> None:
    """
    Compare multiple models and recommend the safest candidate.

    Parameters
    ----------
    reports : list[TrustReport]
        A list of generated TrustReport objects.
    """
    if not reports:
        print("========== Model Comparison & Recommendation ==========")
        print("No reports provided for comparison.\n")
        return

    print("========== Model Comparison & Recommendation ==========")

    # 1. Print summaries for all models
    for rep in reports:
        model_name = rep.metadata.get("model_class", "UnknownModel")
        print(
            f"{model_name:<16} Score: {rep.trust_score.score:3d}/100 | Verdict: {rep.trust_score.verdict}"
        )
    print()

    # 2. Extract viable candidates (unblocked)
    viable_candidates = []
    blocked_candidates = []

    for rep in reports:
        if rep.trust_score.is_blocked:
            blocked_candidates.append(rep)
        else:
            viable_candidates.append(rep)

    # 3. Determine Recommendation
    if not viable_candidates:
        print(" Recommendation: DO NOT DEPLOY any model.")
        print("  * All models triggered critical diagnostic blocks.")
        print("\nPrimary Causes:")
        for rep in blocked_candidates:
            model_name = rep.metadata.get("model_class", "UnknownModel")
            # Identify dominant reason(s) for penalties
            pen_items = sorted(
                rep.trust_score.penalties_applied.items(), key=lambda x: x[1], reverse=True
            )
            reasons = [f"{name.lower()}" for name, val in pen_items[:2] if val > 0]
            reason_str = " + ".join(reasons) if reasons else "general risk"
            print(f"  * {model_name:<16} : {reason_str}")
        return

    # Sort viable candidates by highest score
    viable_candidates.sort(key=lambda x: x.trust_score.score, reverse=True)
    best_candidate = viable_candidates[0]
    best_name = best_candidate.metadata.get("model_class", "UnknownModel")
    best_penalties = sum(best_candidate.trust_score.penalties_applied.values())

    print(f"Recommendation: Deploy {best_name}.")
    print("  * Maintains highest overall Trust Score without critical deployment blocks.")

    if len(reports) > 1:
        # Compare against the second best or the best blocked model
        runner_up = viable_candidates[1] if len(viable_candidates) > 1 else blocked_candidates[0]
        runner_name = runner_up.metadata.get("model_class", "UnknownModel")
        runner_penalties = sum(runner_up.trust_score.penalties_applied.values())

        print(f"  * Comparative Advantage: {best_name} over {runner_name}")
        if best_penalties < runner_penalties:
            # Identify dominant reason(s) for runner_up's higher penalty
            penalties_sorted = sorted(
                runner_up.trust_score.penalties_applied.items(), key=lambda x: x[1], reverse=True
            )
            top_reasons = [f"{name.lower()} risks" for name, val in penalties_sorted[:2] if val > 0]
            reason_str = " + ".join(top_reasons) if top_reasons else "general instability"

            print(f"    - Lower penalty burden (-{best_penalties:.1f} vs -{runner_penalties:.1f})")
            print(f"    - Primarily due to {runner_name}'s {reason_str}.")

        if not runner_up.trust_score.is_blocked:
            score_diff = best_candidate.trust_score.score - runner_up.trust_score.score
            if score_diff > 0:
                print(f"    - Reliability gain: +{score_diff} points higher Trust Score.")
        else:
            print(
                f"    - Note: {runner_name} was filtered out due to active diagnostic blocks ({runner_up.trust_score.verdict})."
            )

    print()
