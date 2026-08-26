"""Seed-balanced monotone calibration, e50 extraction, and operator residuals.

00 §29.1: parameter-recovery error e is the INDEPENDENT variable and R_priv the dependent
one. The planted relationship in DRY-A is R_priv = 1 - c*e, so the fitted curve is
non-increasing in e; a fit that is increasing in e has the axes reversed.

00 §29.2: the operator residual is u_j = r_j - f_hat_{s(j)}(e_j), i.e. the observed recovery
minus what the seed's own rank-matched calibration curve predicts at the same error.

Nothing is extrapolated. A point outside the calibrated support is labelled, and an e50 that
does not exist inside the estimable domain is reported as a state, never as a number
[AUTH: 00 §34B.1 DRY-F; §28A.7 "no extrapolation"].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

CrossingState = Literal["ESTIMABLE", "NO_CROSSING", "INSUFFICIENT_SUPPORT", "INELIGIBLE"]
BEYOND_CALIBRATION: Final = "UNDEFINED_BEYOND_CALIBRATION"


class CalibrationError(ValueError):
    """A calibration curve cannot be fitted or read as requested."""


@dataclass(frozen=True)
class MonotoneCurve:
    """A non-increasing fit of recovery against error, valid only on its own support."""

    error: FloatArray
    recovery: FloatArray
    seed: int | None = None

    @property
    def support(self) -> tuple[float, float]:
        return (float(self.error[0]), float(self.error[-1]))

    def in_support(self, value: float) -> bool:
        low, high = self.support
        return low <= value <= high

    def predict(self, value: float) -> float:
        """Interpolate inside the support. Outside it, refuse [AUTH: 00 §28A.7]."""
        if not self.in_support(value):
            raise CalibrationError(
                f"{BEYOND_CALIBRATION}: e={value:.6g} lies outside the calibrated support"
                f" {self.support}"
            )
        return float(np.interp(value, self.error, self.recovery))

    def is_non_increasing(self) -> bool:
        return bool(np.all(np.diff(self.recovery) <= 1e-12))


def pool_adjacent_violators(values: FloatArray, weights: FloatArray | None = None) -> FloatArray:
    """Non-increasing isotonic fit by pool-adjacent-violators, written out explicitly.

    The frozen pre-data calibration depends on these numbers, so the solver is here rather
    than behind a library whose implementation may change between versions. PAVA is exact:
    blocks are merged while an earlier block mean is below a later one, and each block takes
    its own mean. Ties merge deterministically because the sweep is left to right.
    """
    if weights is None:
        weights = np.ones(values.size, dtype=np.float64)
    if weights.shape != values.shape:
        raise CalibrationError("weights and values disagree in length")
    sums: list[float] = []
    counts: list[float] = []
    for value, weight in zip(values.tolist(), weights.tolist(), strict=True):
        sums.append(float(value) * float(weight))
        counts.append(float(weight))
        while len(sums) >= 2 and sums[-2] / counts[-2] < sums[-1] / counts[-1]:
            merged_sum = sums.pop() + sums.pop()
            merged_count = counts.pop() + counts.pop()
            sums.append(merged_sum)
            counts.append(merged_count)
    fitted: list[float] = []
    for total, count in zip(sums, counts, strict=True):
        fitted.append(total / count)
    expanded: list[float] = []
    for value, run in zip(fitted, _run_lengths(weights, counts), strict=True):
        expanded.extend([value] * run)
    return np.asarray(expanded, dtype=np.float64)


def _run_lengths(weights: FloatArray, block_weights: Sequence[float]) -> list[int]:
    """How many input positions each merged block covers."""
    runs: list[int] = []
    position = 0
    for block in block_weights:
        accumulated = 0.0
        count = 0
        while position < weights.size and accumulated < block - 1e-12:
            accumulated += float(weights[position])
            position += 1
            count += 1
        runs.append(count)
    return runs


def fit_monotone_curve(
    error: Sequence[float] | FloatArray,
    recovery: Sequence[float] | FloatArray,
    *,
    seed: int | None = None,
) -> MonotoneCurve:
    """Isotonic regression of recovery on error, constrained non-increasing.

    Repeated x values are collapsed to their mean before the sweep, weighted by how many
    observations share the rung. Leaving duplicates in place makes the fitted value at that x
    depend on the order they happen to arrive in, which is how `e=[0, 0, 1]` with
    `r=[1, 0.4, 0]` used to read as a crossing at e = 0 [AUTH: 00 §28A.7, §29.1].
    """
    e = np.asarray(error, dtype=np.float64)
    r = np.asarray(recovery, dtype=np.float64)
    if e.shape != r.shape:
        raise CalibrationError("error and recovery arrays disagree in length")
    if e.size < 2:
        raise CalibrationError("at least two points are needed to fit a curve")

    unique, inverse, counts = np.unique(e, return_inverse=True, return_counts=True)
    if unique.size < 2:
        raise CalibrationError("a curve needs at least two distinct error coordinates")
    totals = np.zeros(unique.size, dtype=np.float64)
    np.add.at(totals, inverse, r)
    means = totals / counts
    return MonotoneCurve(
        error=unique.astype(np.float64),
        recovery=pool_adjacent_violators(means, weights=counts.astype(np.float64)),
        seed=seed,
    )


def seed_balanced_points(
    rows: Sequence[tuple[int, float, float]], required_seeds: Sequence[int] | None = None
) -> list[tuple[float, float]]:
    """Aggregate by EXACT registered rung, giving every required training seed equal weight.

    The ladder is a registry of fixed error rungs [AUTH: 00 §28A.1, §28A.2]. Re-binning those
    rungs onto synthesised intervals invents error coordinates the design never registered and
    can collapse distinct rungs together, so grouping is by the exact x value:

    1. group observations by their exact registered rung;
    2. within a rung, average each training seed's repeated draws — synthetic perturbation
       draws are not independent training replicates [AUTH: 00 §29.1];
    3. average across the seeds present, so a seed with more draws cannot dominate
       [AUTH: 00 §32.2];
    4. emit one (rung, recovery) point per rung, in rung order.

    Every rung that appears in the input survives; nothing is merged and nothing is created.
    `required_seeds`, when given, is checked so a missing seed is a visible error rather than
    a silently lighter rung.
    """
    if not rows:
        raise CalibrationError("no rows to balance")
    per_rung: dict[float, dict[int, list[float]]] = {}
    for seed, error, recovery in rows:
        per_rung.setdefault(float(error), {}).setdefault(int(seed), []).append(float(recovery))

    if required_seeds is not None:
        wanted = set(int(seed) for seed in required_seeds)
        for rung, per_seed in sorted(per_rung.items()):
            missing = sorted(wanted - set(per_seed))
            if missing:
                raise CalibrationError(
                    f"rung {rung:.6g} is missing required training seed(s) {missing}"
                )

    balanced: list[tuple[float, float]] = []
    for rung in sorted(per_rung):
        per_seed = per_rung[rung]
        seed_means = [float(np.mean(values)) for _, values in sorted(per_seed.items())]
        balanced.append((rung, float(np.mean(seed_means))))
    return balanced


def rung_weights(rows: Sequence[tuple[int, float, float]]) -> dict[float, int]:
    """How many training seeds contribute to each registered rung."""
    per_rung: dict[float, set[int]] = {}
    for seed, error, _ in rows:
        per_rung.setdefault(float(error), set()).add(int(seed))
    return {rung: len(seeds) for rung, seeds in sorted(per_rung.items())}


@dataclass(frozen=True)
class CrossingResult:
    """Where the curve crosses a level, or why it does not."""

    state: CrossingState
    value: float | None
    detail: str = ""


def crossing_point(curve: MonotoneCurve, level: float, *, min_support: int) -> CrossingResult:
    """The error at which a non-increasing curve first reaches `level`.

    Reported only when the crossing is bracketed by observed points. No extrapolation beyond
    the calibrated support is performed, and no value is invented when the curve never
    reaches the level [AUTH: 00 §28A.7, §34B.1 DRY-F].
    """
    if curve.error.size < min_support:
        return CrossingResult(
            state="INSUFFICIENT_SUPPORT",
            value=None,
            detail=f"{curve.error.size} points, {min_support} required",
        )
    recovery = curve.recovery
    if float(recovery[0]) < level:
        return CrossingResult(
            state="NO_CROSSING", value=None, detail="the curve starts below the level"
        )
    if float(recovery[-1]) > level:
        return CrossingResult(
            state="NO_CROSSING", value=None, detail="the curve never falls to the level"
        )
    below = np.flatnonzero(recovery <= level)
    upper = int(below[0])
    if upper == 0:
        return CrossingResult(state="ESTIMABLE", value=float(curve.error[0]))
    lower = upper - 1
    r_low, r_high = float(recovery[lower]), float(recovery[upper])
    e_low, e_high = float(curve.error[lower]), float(curve.error[upper])
    if r_low == r_high:
        return CrossingResult(state="ESTIMABLE", value=e_high)
    weight = (r_low - level) / (r_low - r_high)
    return CrossingResult(state="ESTIMABLE", value=e_low + weight * (e_high - e_low))


def delta_e50(privacy: CrossingResult, functional: CrossingResult) -> CrossingResult:
    """Δe50 = e50_priv - e50_func, only when BOTH are estimable [AUTH: 00 §34C.2, §29.5]."""
    for name, result in (("privacy", privacy), ("functional", functional)):
        if result.state != "ESTIMABLE" or result.value is None:
            return CrossingResult(
                state=result.state, value=None, detail=f"{name} crossing is {result.state}"
            )
    assert privacy.value is not None and functional.value is not None
    return CrossingResult(state="ESTIMABLE", value=privacy.value - functional.value)


@dataclass(frozen=True)
class Residual:
    """One operator observation's deviation from its own seed's calibration curve."""

    label: str
    operator: str
    seed: int
    error: float
    observed: float
    predicted: float | None
    value: float | None
    state: str


def operator_residuals(
    observations: Sequence[tuple[str, str, int, float, float]],
    curves: Mapping[int, MonotoneCurve],
) -> list[Residual]:
    """u_j = r_j - f_hat_{s(j)}(e_j) [AUTH: 00 §29.2].

    An observation beyond its seed's calibrated support yields no residual at all; it is
    labelled `UNDEFINED_BEYOND_CALIBRATION` so nothing downstream can average it in.
    """
    out: list[Residual] = []
    for label, operator, seed, error, observed in observations:
        curve = curves.get(seed)
        if curve is None:
            out.append(
                Residual(label, operator, seed, error, observed, None, None, "NO_CALIBRATION")
            )
            continue
        try:
            predicted = curve.predict(error)
        except CalibrationError:
            out.append(
                Residual(label, operator, seed, error, observed, None, None, BEYOND_CALIBRATION)
            )
            continue
        out.append(
            Residual(
                label, operator, seed, error, observed, predicted, observed - predicted, "ESTIMABLE"
            )
        )
    return out


def residual_medians(residuals: Sequence[Residual]) -> dict[str, float]:
    """Median residual per operator, over estimable observations only."""
    grouped: dict[str, list[float]] = {}
    for residual in residuals:
        if residual.state == "ESTIMABLE" and residual.value is not None:
            grouped.setdefault(residual.operator, []).append(residual.value)
    return {
        operator: float(np.median(np.asarray(values, dtype=np.float64)))
        for operator, values in sorted(grouped.items())
    }


def spearman(x: Sequence[float] | FloatArray, y: Sequence[float] | FloatArray) -> float:
    """Spearman rank correlation, computed from ranks so no extra dependency is needed."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        raise CalibrationError("spearman needs two equal-length arrays of at least two points")
    ranked_a, ranked_b = _rank(a), _rank(b)
    a_centred = ranked_a - ranked_a.mean()
    b_centred = ranked_b - ranked_b.mean()
    denominator = np.sqrt((a_centred**2).sum() * (b_centred**2).sum())
    if denominator == 0.0:
        return 0.0
    return float((a_centred * b_centred).sum() / denominator)


def _rank(values: FloatArray) -> FloatArray:
    """Average ranks, so ties do not depend on input order."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index in np.flatnonzero(counts > 1):
        mask = inverse == index
        ranks[mask] = ranks[mask].mean()
    return ranks
