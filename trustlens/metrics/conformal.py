"""
trustlens.metrics.conformal.
============================
Conditional-coverage diagnostics for split-conformal prediction sets.

Split conformal prediction turns any point classifier into a set-valued
predictor ``C(x) ⊆ {0, .., K-1}`` that satisfies a *marginal* coverage
guarantee: ``P(y ∈ C(x)) ≥ 1 - α`` averaged over the whole distribution.
Marginal coverage is necessary but weak — a predictor can hit its overall
target while systematically under-covering a specific class or every
hard (large) prediction set. These diagnostics surface that hidden
conditional failure from the prediction sets alone.

Canonical representation
------------------------
Every function works on an ``(n, K)`` boolean membership matrix ``S`` where
``S[i, k]`` is ``True`` iff class ``k`` is in the prediction set for sample
``i``. :func:`to_membership_matrix` builds ``S`` from either an already-formed
0/1 matrix or a ragged list of per-sample label lists.

Metrics implemented
-------------------
* ``to_membership_matrix``      — normalise inputs to the boolean set matrix.
* ``marginal_coverage``         — overall ``P(y ∈ C(x))``.
* ``class_conditional_coverage`` — coverage within each true-label group.
* ``size_stratified_coverage``  — coverage within each set-size stratum (SSC).
* ``set_size_summary``          — set-size efficiency summary.
* ``conformal_diagnostics``     — orchestrator assembling a single report.

Interpretation
--------------
These are *approximate* conditional diagnostics. Exact conditional coverage
(coverage that holds for every fixed x, class, or set size) is impossible to
guarantee distribution-free with finitely many samples (Vovk, 2012; Barber et
al., 2021); the class- and size-conditional numbers here are finite-sample
estimates and should be read together with their per-group sample counts.

References
----------
* Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a
  Random World. Springer. (Split / inductive conformal prediction.)
* Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal
  Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.
  (Size-stratified coverage.)
* Vovk, V. (2012). Conditional validity of inductive conformal predictors.
  ACML. (Impossibility of exact conditional coverage.)
* Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). The
  limits of distribution-free conditional predictive inference. Information and
  Inference.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "to_membership_matrix",
    "marginal_coverage",
    "class_conditional_coverage",
    "size_stratified_coverage",
    "set_size_summary",
    "conformal_diagnostics",
]


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def _is_sequence(x: object) -> bool:
    """Return True for list/tuple/ndarray row containers (but not str/bytes)."""
    return isinstance(x, (list, tuple, np.ndarray)) and not isinstance(x, (str, bytes))


def _is_binary_value(v: object) -> bool:
    """Return True iff ``v`` is a scalar equal to 0 or 1 (bool/int/float)."""
    try:
        fv = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return fv in (0.0, 1.0)


def _matrix_from_array(arr: np.ndarray, n_classes: int | None) -> np.ndarray:
    """Validate a candidate 2D 0/1/bool array and return it as a bool matrix."""
    if arr.ndim != 2:
        raise ValueError(
            f"A membership matrix must be 2D with shape (n_samples, n_classes); got ndim={arr.ndim}."
        )
    matrix: np.ndarray
    if arr.dtype == bool:
        matrix = arr.astype(bool)
    elif np.issubdtype(arr.dtype, np.number):
        unique = np.unique(arr)
        if not set(unique.tolist()).issubset({0, 1}):
            raise ValueError(
                "A membership matrix must contain only 0/1 (or bool) entries; "
                f"got values {unique.tolist()}."
            )
        matrix = arr.astype(bool)
    else:
        raise ValueError(
            f"A membership matrix must be numeric 0/1 or boolean; got dtype {arr.dtype}."
        )
    if n_classes is not None and matrix.shape[1] != n_classes:
        raise ValueError(f"Matrix width {matrix.shape[1]} does not match n_classes={n_classes}.")
    return matrix


def _matrix_from_label_lists(sets: list, n_classes: int | None) -> np.ndarray:
    """Build an ``(n, K)`` bool matrix from a sequence of per-sample label lists."""
    n = len(sets)
    parsed: list[list[int]] = []
    max_label = -1
    for i, row in enumerate(sets):
        if not _is_sequence(row):
            raise ValueError(
                f"Prediction set for sample {i} must be a sequence of class indices; "
                f"got {type(row).__name__}."
            )
        labels: list[int] = []
        for v in row:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid label {v!r} in sample {i}: labels must be integer class indices."
                ) from None
            iv = int(fv)
            if iv != fv:
                raise ValueError(
                    f"Invalid non-integer label {v!r} in sample {i}: labels must be integers."
                )
            if iv < 0:
                raise ValueError(f"Invalid negative label {iv} in sample {i}: labels must be >= 0.")
            labels.append(iv)
        parsed.append(labels)
        if labels:
            max_label = max(max_label, max(labels))

    if n_classes is None:
        if max_label < 0:
            raise ValueError(
                "Cannot infer n_classes from all-empty label sets; pass n_classes explicitly."
            )
        k = max_label + 1
    else:
        k = n_classes
        if max_label >= k:
            raise ValueError(
                f"Label {max_label} is out of range for n_classes={k} (valid labels are 0..{k - 1})."
            )

    matrix: np.ndarray = np.zeros((n, k), dtype=bool)
    for i, labels in enumerate(parsed):
        for iv in labels:
            matrix[i, iv] = True
    return matrix


def to_membership_matrix(y_pred_sets, n_classes: int | None = None) -> np.ndarray:
    """
    Normalise prediction sets to an ``(n, K)`` boolean membership matrix.

    What it accepts
    ---------------
    Either representation of the per-sample prediction sets ``C(x_i)``:

    * an ``(n, K)`` 0/1 or boolean **matrix** — returned as ``bool`` with its
      shape unchanged (a rectangular sequence whose entries are all 0/1 is
      treated as a matrix), or
    * a ragged **list of per-sample label lists**, e.g. ``[[0, 2], [1], []]``,
      where each inner list holds the class indices in that sample's set (``[]``
      is the empty set).

    Disambiguation
    --------------
    A rectangular input whose every entry is 0 or 1 is read as a membership
    matrix. Any ragged input, or a rectangular input containing a value outside
    ``{0, 1}``, is read as label lists. (Equal-length label lists drawn only
    from ``{0, 1}`` are therefore treated as a matrix — pass them with an
    explicit ragged shape if that is not intended.)

    Parameters
    ----------
    y_pred_sets : np.ndarray or sequence
      The prediction sets in either representation described above.
    n_classes : int or None, default=None
      Number of classes ``K``. If ``None`` it is inferred from the matrix width
      (matrix input) or ``max(label) + 1`` across all sets (label-list input).
      If given, it is checked against a matrix's width and used as an upper
      bound (``0 <= label < n_classes``) for label lists.

    Returns
    -------
    np.ndarray
      Boolean array of shape ``(n_samples, n_classes)``.

    Raises
    ------
    ValueError
      If the input is ragged in an unrecognised way, contains invalid
      (non-integer, negative, or out-of-range) labels, a matrix carries values
      other than 0/1, or ``n_classes`` cannot be inferred from all-empty sets.

    Examples
    --------
    >>> to_membership_matrix([[0, 2], [1], []]).astype(int)
    array([[1, 0, 1],
           [0, 1, 0],
           [0, 0, 0]])
    """
    if isinstance(y_pred_sets, np.ndarray):
        arr = y_pred_sets
        if arr.ndim == 2 and arr.dtype != object:
            return _matrix_from_array(arr, n_classes)
        if arr.ndim == 1 and arr.dtype == object:
            return _matrix_from_label_lists(list(arr), n_classes)
        raise ValueError(
            "ndarray input must be a 2D 0/1 membership matrix or a 1D object array of "
            f"label lists; got shape {arr.shape} with dtype {arr.dtype}."
        )

    seq = list(y_pred_sets)
    if len(seq) == 0:
        if n_classes is None:
            raise ValueError("Cannot build a membership matrix from empty input without n_classes.")
        empty: np.ndarray = np.zeros((0, n_classes), dtype=bool)
        return empty

    if not all(_is_sequence(row) for row in seq):
        raise ValueError(
            "Expected a 2D 0/1 matrix or a list of per-sample label lists; got a sequence "
            "whose elements are not themselves sequences."
        )

    lengths = {len(row) for row in seq}
    if len(lengths) == 1:
        width = next(iter(lengths))
        # Rectangular and all-0/1: a membership matrix, *unless* an explicit
        # n_classes is given that the width cannot satisfy — a matrix's width is
        # by definition n_classes, so a mismatch means these are label lists.
        if width > 0 and all(_is_binary_value(v) for row in seq for v in row):
            if n_classes is None or width == n_classes:
                return _matrix_from_array(np.asarray(seq), n_classes)
    return _matrix_from_label_lists(seq, n_classes)


# ---------------------------------------------------------------------------
# Shared internals
# ---------------------------------------------------------------------------


def _check_y_true(y_true, n: int) -> np.ndarray:
    """Validate ``y_true`` as ``n`` non-negative integer labels; return int array."""
    yt = np.asarray(y_true)
    if yt.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {yt.shape}.")
    if yt.size == 0:
        raise ValueError("y_true must be non-empty.")
    if yt.size != n:
        raise ValueError(
            f"Length mismatch: y_true has {yt.size} samples but the prediction sets have {n}."
        )
    if not np.issubdtype(yt.dtype, np.integer):
        try:
            yt_float = yt.astype(float)
        except (TypeError, ValueError):
            raise ValueError("y_true must contain integer class labels.") from None
        if not np.all(yt_float == np.round(yt_float)):
            raise ValueError("y_true must contain integer class labels.")
        yt = yt_float
    yti: np.ndarray = yt.astype(int)
    if np.any(yti < 0):
        raise ValueError("y_true labels must be non-negative integers.")
    return yti


def _coverage_vector(S: np.ndarray, y_true_int: np.ndarray) -> np.ndarray:
    """Per-sample coverage ``S[i, y_true[i]]``.

    A true label ``>= K`` (the class span of the prediction sets) is counted as a
    miss, not an error: if the predictor never places class ``k`` in any set, then
    ``k`` is genuinely never covered, and surfacing that as 0.0 coverage — rather
    than raising — is precisely what the conditional diagnostics exist to reveal.
    ``_check_y_true`` already rejects negative labels.
    """
    n, k = S.shape
    covered: np.ndarray = np.zeros(n, dtype=bool)
    in_range = y_true_int < k
    idx = np.nonzero(in_range)[0]
    if idx.size:
        covered[idx] = S[idx, y_true_int[idx]]
    return covered


def _class_conditional(y_true_int: np.ndarray, covered: np.ndarray) -> dict:
    """Class-conditional coverage report from label + coverage vectors."""
    per_class: dict[int, float] = {}
    for k in np.unique(y_true_int):
        mask = y_true_int == k
        per_class[int(k)] = float(covered[mask].mean())
    worst_class = min(per_class, key=lambda kk: (per_class[kk], kk))
    return {
        "per_class": per_class,
        "worst_class": int(worst_class),
        "worst_coverage": float(per_class[worst_class]),
    }


def _size_stratified(S: np.ndarray, covered: np.ndarray, min_stratum: int) -> dict:
    """Size-stratified coverage report from a membership matrix + coverage vector."""
    sizes = S.sum(axis=1)
    per_size: dict[int, dict] = {}
    eligible: dict[int, float] = {}
    for s in np.unique(sizes):
        mask = sizes == s
        count = int(mask.sum())
        cov = float(covered[mask].mean())
        low_confidence = count < min_stratum
        per_size[int(s)] = {
            "coverage": cov,
            "count": count,
            "low_confidence": low_confidence,
        }
        if not low_confidence:
            eligible[int(s)] = cov

    if eligible:
        worst_size = min(eligible, key=lambda ss: (eligible[ss], ss))
        worst_cov: float | None = eligible[worst_size]
        worst_stratum_size: int | None = int(worst_size)
    else:
        worst_cov = None
        worst_stratum_size = None

    return {
        "per_size": per_size,
        "worst_stratum_coverage": worst_cov,
        "worst_stratum_size": worst_stratum_size,
    }


def _size_summary(S: np.ndarray) -> dict:
    """Set-size efficiency summary from a membership matrix."""
    n, k = S.shape
    sizes = S.sum(axis=1)
    avg_size = float(sizes.mean())
    singleton_rate = float(np.mean(sizes == 1))
    empty_rate = float(np.mean(sizes == 0))
    if k <= 1:
        size_efficiency = 1.0
    else:
        eff = 1.0 - (avg_size - 1.0) / (k - 1.0)
        size_efficiency = float(min(1.0, max(0.0, eff)))
    return {
        "avg_size": avg_size,
        "singleton_rate": singleton_rate,
        "empty_rate": empty_rate,
        "size_efficiency": size_efficiency,
    }


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------


def marginal_coverage(y_true, pred_sets) -> float:
    """
    Overall (marginal) coverage of the prediction sets.

    What it measures
    ----------------
    The fraction of samples whose true label lies inside its prediction set,
    ``mean_i 1[y_i ∈ C(x_i)]``. An empty set counts as a miss.

    Why it matters
    --------------
    This is the quantity a split-conformal predictor is built to control, so it
    is the first sanity check: does realised coverage match the nominal target?

    Limitations
    -----------
    Marginal coverage is an *average* — it can sit exactly on target while a
    specific class or hard-example region is badly under-covered. Read it
    alongside :func:`class_conditional_coverage` and
    :func:`size_stratified_coverage`.

    Parameters
    ----------
    y_true : array-like of int
      True class labels, shape ``(n_samples,)``.
    pred_sets : np.ndarray or sequence
      Prediction sets in any form accepted by :func:`to_membership_matrix`.

    Returns
    -------
    float
      Marginal coverage in ``[0, 1]``.

    Raises
    ------
    ValueError
      If the inputs are empty or ``len(y_true)`` does not match the number of
      prediction sets.

    Examples
    --------
    >>> marginal_coverage([0, 1, 2], [[0], [1], []])
    0.6666666666666666
    """
    S = to_membership_matrix(pred_sets)
    n = S.shape[0]
    if n == 0:
        raise ValueError("marginal_coverage requires non-empty inputs.")
    y_true_int = _check_y_true(y_true, n)
    covered = _coverage_vector(S, y_true_int)
    return float(covered.mean())


def class_conditional_coverage(y_true, pred_sets, n_classes: int | None = None) -> dict:
    """
    Coverage computed separately within each true-label group.

    What it measures
    ----------------
    For every class ``k`` that appears in ``y_true``, the coverage among the
    samples whose *true* label is ``k`` — i.e. how often the predictor keeps
    class ``k`` when ``k`` is the right answer.

    Why it matters
    --------------
    A predictor can meet its marginal target while sacrificing a minority class.
    The worst class-conditional coverage is the number a fairness or safety
    review cares about: the group the predictor protects least.

    Limitations
    -----------
    Estimated from however many samples carry each label; rare classes give
    noisy estimates. Classes with no samples in ``y_true`` are omitted. A class
    that appears in ``y_true`` but in no prediction set surfaces as 0.0 coverage
    (its true samples cannot be covered) — intentional, and often the headline
    failure signal rather than an error.

    Parameters
    ----------
    y_true : array-like of int
      True class labels, shape ``(n_samples,)``.
    pred_sets : np.ndarray or sequence
      Prediction sets in any form accepted by :func:`to_membership_matrix`.
    n_classes : int or None, default=None
      Passed through to :func:`to_membership_matrix`.

    Returns
    -------
    dict with keys:
      * ``per_class`` — ``{class: coverage}`` for each class present in ``y_true``.
      * ``worst_class`` — the class with the lowest coverage (ties: lowest index).
      * ``worst_coverage`` — that class's coverage.

    Raises
    ------
    ValueError
      On empty inputs or a length mismatch.

    Examples
    --------
    >>> class_conditional_coverage([0, 0, 1, 1], [[0], [0], [1], []])["worst_class"]
    1
    """
    S = to_membership_matrix(pred_sets, n_classes)
    n = S.shape[0]
    if n == 0:
        raise ValueError("class_conditional_coverage requires non-empty inputs.")
    y_true_int = _check_y_true(y_true, n)
    covered = _coverage_vector(S, y_true_int)
    return _class_conditional(y_true_int, covered)


def size_stratified_coverage(y_true, pred_sets, min_stratum: int = 20) -> dict:
    """
    Coverage computed separately within each prediction-set-size stratum.

    What it measures
    ----------------
    Size-stratified coverage (SSC; Angelopoulos & Bates, 2021): group samples by
    the size ``s = |C(x_i)|`` of their prediction set and measure coverage within
    each size group. Well-behaved conformal predictors keep coverage roughly
    constant across sizes; a slump on the large-set (hard) strata reveals that
    marginal coverage is being propped up by easy, singleton examples.

    Why it matters
    --------------
    The worst-size stratum is the "conditional failure marginal coverage hides":
    overall coverage can read 0.90 while every large-set (uncertain) example sits
    well below target.

    Limitations
    -----------
    Small strata give noisy estimates, so any stratum with fewer than
    ``min_stratum`` samples is reported but flagged ``low_confidence`` and
    excluded from the worst-stratum computation.

    Parameters
    ----------
    y_true : array-like of int
      True class labels, shape ``(n_samples,)``.
    pred_sets : np.ndarray or sequence
      Prediction sets in any form accepted by :func:`to_membership_matrix`.
    min_stratum : int, default=20
      Minimum stratum size to be treated as reliable (eligible for the
      worst-stratum computation).

    Returns
    -------
    dict with keys:
      * ``per_size`` — ``{size: {"coverage", "count", "low_confidence"}}``.
      * ``worst_stratum_coverage`` — the minimum coverage over eligible strata,
        or ``None`` if no stratum reaches ``min_stratum`` samples.
      * ``worst_stratum_size`` — the set size achieving that minimum (or ``None``).

    Raises
    ------
    ValueError
      On empty inputs or a length mismatch.

    Examples
    --------
    >>> out = size_stratified_coverage(y_true, pred_sets, min_stratum=20)
    >>> out["worst_stratum_coverage"]
    """
    S = to_membership_matrix(pred_sets)
    n = S.shape[0]
    if n == 0:
        raise ValueError("size_stratified_coverage requires non-empty inputs.")
    y_true_int = _check_y_true(y_true, n)
    covered = _coverage_vector(S, y_true_int)
    return _size_stratified(S, covered, min_stratum)


def set_size_summary(pred_sets, n_classes: int | None = None) -> dict:
    """
    Summarise prediction-set sizes and their efficiency.

    What it measures
    ----------------
    The distribution of set sizes ``|C(x_i)|`` — average size, the rate of
    singleton (size-1, maximally informative) sets, the rate of empty sets, and a
    normalised ``size_efficiency`` score.

    Why it matters
    --------------
    Coverage is only half the story: a predictor that returns the full label set
    every time is trivially 100% covered but useless. Efficiency captures how
    *informative* the sets are — small sets at fixed coverage are the goal.

    Efficiency definition
    ---------------------
    ``size_efficiency = 1 - (avg_size - 1) / (K - 1)`` for ``K > 1``, clipped to
    ``[0, 1]``; defined as ``1.0`` when ``K == 1``. All-singleton sets score
    ``1.0``; all-full sets score ``0.0``. Because empty sets pull ``avg_size``
    below 1, they raise the raw score above 1 and are clipped to ``1.0`` — so
    ``size_efficiency`` should always be read next to ``empty_rate`` and coverage
    (empty sets are size-efficient but never cover).

    Parameters
    ----------
    pred_sets : np.ndarray or sequence
      Prediction sets in any form accepted by :func:`to_membership_matrix`.
    n_classes : int or None, default=None
      Passed through to :func:`to_membership_matrix`; determines ``K``.

    Returns
    -------
    dict with keys ``avg_size``, ``singleton_rate``, ``empty_rate`` and
    ``size_efficiency``.

    Raises
    ------
    ValueError
      On empty input.

    Examples
    --------
    >>> set_size_summary([[0], [1], [2]])["size_efficiency"]
    1.0
    """
    S = to_membership_matrix(pred_sets, n_classes)
    n = S.shape[0]
    if n == 0:
        raise ValueError("set_size_summary requires non-empty inputs.")
    return _size_summary(S)


def conformal_diagnostics(
    y_true,
    y_pred_sets,
    nominal_coverage: float | None = None,
    n_classes: int | None = None,
) -> dict:
    """
    Assemble the full conditional-coverage diagnostic report in one pass.

    Converts the inputs once and computes marginal, class-conditional and
    size-stratified coverage plus the set-size summary, then folds in the gaps
    against the nominal target. When ``nominal_coverage`` is ``None`` the
    coverage and size numbers are still reported but every gap/violation field is
    ``None`` (never silently ``0``) so a missing target is unmistakable.

    Parameters
    ----------
    y_true : array-like of int
      True class labels, shape ``(n_samples,)``.
    y_pred_sets : np.ndarray or sequence
      Prediction sets in any form accepted by :func:`to_membership_matrix`.
    nominal_coverage : float or None, default=None
      The target coverage ``1 - α`` the sets claim to provide, in ``(0, 1]``.
    n_classes : int or None, default=None
      Passed through to :func:`to_membership_matrix`.

    Returns
    -------
    dict with keys:
      * ``nominal`` — the supplied ``nominal_coverage`` (echoed).
      * ``marginal_coverage`` — overall coverage.
      * ``coverage_gap`` — ``marginal - nominal`` (``None`` if no target).
      * ``worst_class_coverage`` — lowest class-conditional coverage.
      * ``worst_class_gap`` — ``nominal - worst_class_coverage`` (``None`` if no target).
      * ``ssc_violation`` — ``max(0, nominal - worst_stratum_coverage)`` (``None``
        if no target or no eligible stratum).
      * ``avg_set_size``, ``singleton_rate``, ``size_efficiency``, ``empty_rate``.
      * ``n_classes``, ``n_samples``.

    Raises
    ------
    ValueError
      On empty inputs, a length mismatch, or ``nominal_coverage`` outside ``(0, 1]``.

    Examples
    --------
    >>> report = conformal_diagnostics(y_true, pred_sets, nominal_coverage=0.9)
    >>> report["ssc_violation"]
    """
    if nominal_coverage is not None and not 0.0 < nominal_coverage <= 1.0:
        raise ValueError(f"nominal_coverage must be in (0, 1]; got {nominal_coverage}.")

    S = to_membership_matrix(y_pred_sets, n_classes)
    n, k = S.shape
    if n == 0:
        raise ValueError("conformal_diagnostics requires non-empty inputs.")
    y_true_int = _check_y_true(y_true, n)
    covered = _coverage_vector(S, y_true_int)

    marginal = float(covered.mean())
    ccc = _class_conditional(y_true_int, covered)
    ssc = _size_stratified(S, covered, min_stratum=20)
    sizes = _size_summary(S)

    worst_class_cov = ccc["worst_coverage"]
    worst_stratum_cov = ssc["worst_stratum_coverage"]

    if nominal_coverage is None:
        coverage_gap: float | None = None
        worst_class_gap: float | None = None
        ssc_violation: float | None = None
    else:
        coverage_gap = marginal - nominal_coverage
        worst_class_gap = nominal_coverage - worst_class_cov
        if worst_stratum_cov is None:
            ssc_violation = None
        else:
            ssc_violation = max(0.0, nominal_coverage - worst_stratum_cov)

    return {
        "nominal": nominal_coverage,
        "marginal_coverage": marginal,
        "coverage_gap": coverage_gap,
        "worst_class_coverage": worst_class_cov,
        "worst_class_gap": worst_class_gap,
        "ssc_violation": ssc_violation,
        "avg_set_size": sizes["avg_size"],
        "singleton_rate": sizes["singleton_rate"],
        "size_efficiency": sizes["size_efficiency"],
        "empty_rate": sizes["empty_rate"],
        "n_classes": int(k),
        "n_samples": int(n),
    }
