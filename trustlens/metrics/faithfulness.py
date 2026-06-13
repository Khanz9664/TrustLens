# NOTE:
# This module is under active development and is not part of the public API.
# Do not import into production pipelines until stabilized.

"""
trustlens.metrics.faithfulness.
==============================
Metric-level wrappers for faithfulness evaluation results.

This module provides aggregation functions that operate on the
pixel deletion/insertion test outputs from
``trustlens.explainability.faithfulness``.
"""

from __future__ import annotations


def faithfulness_summary(
    deletion_result: dict,
    insertion_result: dict,
) -> dict:
    """
    Summarize faithfulness evaluation across deletion and insertion tests.

    Parameters
    ----------
    deletion_result : dict
      Output from ``pixel_deletion_test()``.
    insertion_result : dict
      Output from ``pixel_insertion_test()``.

    Returns
    -------
    dict
      Combined summary with:
      * ``deletion_aupc`` — lower is better
      * ``insertion_aupc`` — higher is better
      * ``faithfulness_score`` — insertion_aupc - deletion_aupc (higher = more faithful)
    """
    d_aupc = deletion_result["aupc"]
    i_aupc = insertion_result["aupc"]

    return {
        "deletion_aupc": d_aupc,
        "insertion_aupc": i_aupc,
        "faithfulness_score": round(i_aupc - d_aupc, 4),
        "verdict": (
            "faithful"
            if (i_aupc - d_aupc) > 0.1
            else "marginally_faithful"
            if (i_aupc - d_aupc) > 0
            else "unfaithful"
        ),
    }
