"""Planted-truth generators and the null/positive calibrations they feed.

Everything here is synthetic by construction: no real record, no real model, no real
outcome. The dry run is an implementation test, not a scientific result [AUTH: 00 §34B].

DRY-G is gated by binding correction 02 §C4:

    N_FWER_CALIBRATION      trials on fixed master seeds from resolved config
    statistic vector        Spearman rho, operator residual, mean D_L, mean D_R, Delta_CV
    standardisation         centre and scale estimated ONLY from the calibration bank
    two-sided statistics    absolute standardised deviation
    one-sided Delta_CV      positive standardised deviation
    T_max = max_j Z_j       c_FWER = Q_0.95(T_max)
    a trial passes iff      T_max <= c_FWER
    validation              3/3 PASS, 2/3 PASS_WITH_INVESTIGATION, <=1/3 FAIL

Cell-cluster bootstrap CIs are computed and reported as diagnostics; §C4 states plainly that
they are not a second pass/fail criterion, so nothing here reads them to decide a verdict.

The planted-null parameterisation is a plain structural choice recorded in
configs/p0/dry_run.json — one cell-level and one within-seed noise scale for recovery, and
one of each for the gaps because D_L and D_R are differences between correlated views. No
search or optimisation fits those parameters to any acceptance value; the 00 §34B.1A
200-trial marginal numbers are computed only as NON_GATING_DIAGNOSTIC and decide nothing.

Gaussian draws are Box-Muller over `Generator.random()` rather than `Generator.normal()`:
NumPy guarantees the bit stream but permits distribution algorithms to change between major
versions, which would silently move a frozen pre-data tolerance [AUTH: 01 §30; 00 §34B.1A].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.analysis.isotonic import (
    MonotoneCurve,
    fit_monotone_curve,
    operator_residuals,
    residual_medians,
    seed_balanced_points,
    spearman,
)
from src.analysis.resampling import ClusterUnit, bootstrap, cluster_rows

FloatArray = NDArray[np.float64]

#: The five DRY-G null statistics, in the frozen order used by the family-wise vector.
NULL_STATISTIC_NAMES: Final[tuple[str, ...]] = (
    "spearman_rho",
    "max_abs_median_operator_residual",
    "mean_d_l",
    "mean_d_r",
    "delta_cv",
)

#: The 00 §28A.1 Stage-1 coarse ladder. Observations sit on these registered rungs; they are
#: never re-binned [AUTH: 00 §28A.1, §28A.7].
COARSE_RUNGS: Final[tuple[float, ...]] = (
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.60,
    0.80,
    1.00,
    1.50,
    2.00,
)

#: 00 §28A.1: three perturbation draws per seed, level and ladder type. Calibration
#: replicates, not independent training seeds.
N_PERTURBATION_DRAWS: Final = 3

#: Delta_CV is one-sided: only a positive structural improvement is evidence against the
#: null, so its tolerance is an upper bound on the signed value [AUTH: 00 §34B.1].
ONE_SIDED_STATISTICS: Final[frozenset[str]] = frozenset({"delta_cv"})


class DryRunError(ValueError):
    """A planted scenario cannot be generated or evaluated as specified."""


def normal(rng: np.random.Generator, size: int) -> FloatArray:
    """Box-Muller over the guaranteed uniform stream. Deterministic across NumPy versions."""
    if size <= 0:
        return np.zeros(0, dtype=np.float64)
    # One (u1, u2) pair yields one value here: only the cosine branch is used, so the pair
    # count equals the requested size and the stream position is a simple function of it.
    uniforms = rng.random(2 * size)
    u1 = np.clip(uniforms[0::2], np.finfo(np.float64).tiny, 1.0)
    u2 = uniforms[1::2]
    radius = np.sqrt(-2.0 * np.log(u1))
    return np.asarray(radius * np.cos(2.0 * np.pi * u2), dtype=np.float64)


@dataclass(frozen=True)
class TrialRow:
    """One registry-cell/seed observation of the downstream (e, r) analysis table."""

    cell: str
    seed: int
    operator: str
    error: float
    recovery: float
    d_l: float
    d_r: float


@dataclass(frozen=True)
class TrialDesign:
    """The clustered design shared by calibration and validation trials [AUTH: 00 §34B.1]."""

    n_cells: int
    seeds: tuple[int, ...]
    operators: tuple[str, ...]
    error_low: float
    error_high: float
    cell_sd: float
    row_sd: float
    recovery_base: float
    #: D_L and D_R are differences between two views' TPRs on the same records, so their null
    #: spread is smaller than the recovery spread. Given their own scale rather than reusing
    #: the recovery one [AUTH: 00 §34B.1 "cell-level shared noise plus within-seed residual"].
    gap_cell_sd: float
    gap_row_sd: float


def generate_null_trial(design: TrialDesign, master_seed: int) -> list[TrialRow]:
    """A complete planted null: recovery independent of e, no operator or lineage effect.

    Cell-level shared noise plus within-seed residual noise makes the effective independent
    unit the registry cell rather than the row, which is the dependence the cell-cluster
    bootstrap must respect [AUTH: 00 §34B.1].
    """
    rng = np.random.Generator(np.random.PCG64(master_seed))
    errors = design.error_low + rng.random(design.n_cells) * (design.error_high - design.error_low)
    cell_noise = normal(rng, design.n_cells) * design.cell_sd
    cell_l = normal(rng, design.n_cells) * design.gap_cell_sd
    cell_r = normal(rng, design.n_cells) * design.gap_cell_sd

    rows: list[TrialRow] = []
    for index in range(design.n_cells):
        operator = design.operators[index % len(design.operators)]
        row_noise = normal(rng, len(design.seeds)) * design.row_sd
        row_l = normal(rng, len(design.seeds)) * design.gap_row_sd
        row_r = normal(rng, len(design.seeds)) * design.gap_row_sd
        for position, seed in enumerate(design.seeds):
            rows.append(
                TrialRow(
                    cell=f"cell-{index:02d}",
                    seed=seed,
                    operator=operator,
                    error=float(errors[index]),
                    recovery=float(design.recovery_base + cell_noise[index] + row_noise[position]),
                    d_l=float(cell_l[index] + row_l[position]),
                    d_r=float(cell_r[index] + row_r[position]),
                )
            )
    return rows


def generate_monotone_trial(
    design: TrialDesign, master_seed: int, *, slope: float, operator_effect: Mapping[str, float]
) -> list[TrialRow]:
    """DRY-A / DRY-B: r = 1 - slope*e + planted operator residual + noise."""
    rng = np.random.Generator(np.random.PCG64(master_seed))
    errors = design.error_low + rng.random(design.n_cells) * (design.error_high - design.error_low)
    cell_noise = normal(rng, design.n_cells) * design.cell_sd

    rows: list[TrialRow] = []
    for index in range(design.n_cells):
        operator = design.operators[index % len(design.operators)]
        row_noise = normal(rng, len(design.seeds)) * design.row_sd
        for position, seed in enumerate(design.seeds):
            recovery = (
                1.0
                - slope * float(errors[index])
                + operator_effect.get(operator, 0.0)
                + cell_noise[index]
                + row_noise[position]
            )
            rows.append(
                TrialRow(
                    cell=f"cell-{index:02d}",
                    seed=seed,
                    operator=operator,
                    error=float(errors[index]),
                    recovery=float(recovery),
                    d_l=0.0,
                    d_r=0.0,
                )
            )
    return rows


def generate_calibration_ladder(
    design: TrialDesign, master_seed: int, *, slope: float
) -> list[TrialRow]:
    """Rank-matched synthetic calibration observations on the registered rungs.

    These carry NO operator effect: they are the magnitude baseline that operator points are
    later compared against. Every training seed contributes `N_PERTURBATION_DRAWS` draws at
    every rung [AUTH: 00 §28A.1, §28A.2, §29.1].
    """
    rng = np.random.Generator(np.random.PCG64(master_seed))
    rows: list[TrialRow] = []
    for rung_index, rung in enumerate(COARSE_RUNGS):
        for seed in design.seeds:
            noise = normal(rng, N_PERTURBATION_DRAWS) * design.row_sd
            for draw in range(N_PERTURBATION_DRAWS):
                rows.append(
                    TrialRow(
                        cell=f"ladder-{rung_index:02d}-{draw}",
                        seed=seed,
                        operator="SYNTHETIC",
                        error=rung,
                        recovery=float(1.0 - slope * rung + noise[draw]),
                        d_l=0.0,
                        d_r=0.0,
                    )
                )
    return rows


def generate_operator_observations(
    design: TrialDesign, master_seed: int, *, slope: float, operator_effect: Mapping[str, float]
) -> list[TrialRow]:
    """Real-operator points, generated independently of the calibration ladder.

    They land on the registered rungs so the baseline can be read at their exact error, but
    they are a disjoint sample: nothing here contributes to the curve they are judged by
    [AUTH: 00 §29.2].
    """
    rng = np.random.Generator(np.random.PCG64(master_seed))
    rows: list[TrialRow] = []
    for index, rung in enumerate(COARSE_RUNGS):
        operator = design.operators[index % len(design.operators)]
        noise = normal(rng, len(design.seeds)) * design.row_sd
        for position, seed in enumerate(design.seeds):
            rows.append(
                TrialRow(
                    cell=f"op-{index:02d}",
                    seed=seed,
                    operator=operator,
                    error=rung,
                    recovery=float(
                        1.0 - slope * rung + operator_effect.get(operator, 0.0) + noise[position]
                    ),
                    d_l=0.0,
                    d_r=0.0,
                )
            )
    return rows


def baseline_curves(calibration_rows: Sequence[TrialRow]) -> dict[int, MonotoneCurve]:
    """Per-seed magnitude curves fitted ONLY from calibration observations [AUTH: 00 §29.2]."""
    return seed_curves(calibration_rows)


def residuals_against_baseline(
    observations: Sequence[TrialRow], curves: Mapping[int, MonotoneCurve]
) -> dict[str, float]:
    """u_j = r_j - f_hat_{s(j)}(e_j) against a baseline the observations did not shape."""
    return residual_medians(
        operator_residuals(
            [
                (f"{row.cell}:{row.seed}", row.operator, row.seed, row.error, row.recovery)
                for row in observations
            ],
            curves,
        )
    )


def seed_curves(rows: Sequence[TrialRow]) -> dict[int, MonotoneCurve]:
    """One rank-matched calibration curve per training seed [AUTH: 00 §29.1, §29.2]."""
    grouped: dict[int, list[TrialRow]] = {}
    for row in rows:
        grouped.setdefault(row.seed, []).append(row)
    curves: dict[int, MonotoneCurve] = {}
    for seed, seed_rows in grouped.items():
        if len({row.error for row in seed_rows}) < 2:
            continue
        points = seed_balanced_points([(seed, r.error, r.recovery) for r in seed_rows])
        curves[seed] = fit_monotone_curve([p[0] for p in points], [p[1] for p in points], seed=seed)
    return curves


def _mean_absolute_error(observed: Sequence[float], predicted: Sequence[float]) -> float:
    return float(np.mean(np.abs(np.asarray(observed) - np.asarray(predicted))))


def grouped_cv_improvement(rows: Sequence[TrialRow]) -> float:
    """Delta_CV = MAE_magnitude - MAE_structural, leave-one-training-seed-out [AUTH: 00 §34B.1].

    The structural model is the magnitude curve plus a per-operator offset learned on the
    training seeds. Under the null the offset is noise, so the improvement is about zero.
    """
    seeds = sorted({row.seed for row in rows})
    if len(seeds) < 2:
        raise DryRunError("grouped CV needs at least two training seeds")
    magnitude_errors: list[float] = []
    structural_errors: list[float] = []
    for held_out in seeds:
        train = [row for row in rows if row.seed != held_out]
        test = [row for row in rows if row.seed == held_out]
        if len(train) < 2 or not test:
            continue
        curve = fit_monotone_curve([r.error for r in train], [r.recovery for r in train])
        offsets: dict[str, float] = {}
        for operator in {r.operator for r in train}:
            deviations = [
                r.recovery - curve.predict(min(max(r.error, curve.support[0]), curve.support[1]))
                for r in train
                if r.operator == operator
            ]
            offsets[operator] = float(np.mean(deviations)) if deviations else 0.0
        magnitude = [
            curve.predict(min(max(r.error, curve.support[0]), curve.support[1])) for r in test
        ]
        structural = [
            value + offsets.get(row.operator, 0.0)
            for value, row in zip(magnitude, test, strict=True)
        ]
        observed = [row.recovery for row in test]
        magnitude_errors.append(_mean_absolute_error(observed, magnitude))
        structural_errors.append(_mean_absolute_error(observed, structural))
    if not magnitude_errors:
        raise DryRunError("grouped CV produced no folds")
    return float(np.mean(magnitude_errors) - np.mean(structural_errors))


def null_statistics(rows: Sequence[TrialRow]) -> dict[str, float]:
    """The five DRY-G statistics, in the frozen order [AUTH: 00 §34B.1; 02 §C4]."""
    curves = seed_curves(rows)
    residuals = operator_residuals(
        [(f"{r.cell}:{r.seed}", r.operator, r.seed, r.error, r.recovery) for r in rows], curves
    )
    medians = residual_medians(residuals)
    return {
        "spearman_rho": spearman([r.error for r in rows], [r.recovery for r in rows]),
        "max_abs_median_operator_residual": (
            float(max(abs(v) for v in medians.values())) if medians else 0.0
        ),
        "mean_d_l": float(np.mean([r.d_l for r in rows])),
        "mean_d_r": float(np.mean([r.d_r for r in rows])),
        "delta_cv": grouped_cv_improvement(rows),
    }


@dataclass(frozen=True)
class FamilywiseCalibration:
    """The 02 §C4 family-wise bank. This is the DRY-G gate."""

    n_trials: int
    seeds: tuple[int, ...]
    quantile: float
    centre: Mapping[str, float]
    scale: Mapping[str, float]
    c_fwer: float

    def standardise(self, statistics: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in NULL_STATISTIC_NAMES:
            scale = self.scale[name]
            if scale <= 0.0:
                raise DryRunError(f"calibration scale for {name!r} is not positive")
            z = (statistics[name] - self.centre[name]) / scale
            out[name] = float(z if name in ONE_SIDED_STATISTICS else abs(z))
        return out

    def t_max(self, statistics: Mapping[str, float]) -> float:
        return float(max(self.standardise(statistics).values()))

    def passes(self, statistics: Mapping[str, float]) -> bool:
        return self.t_max(statistics) <= self.c_fwer


def calibrate_familywise(
    design: TrialDesign, master_seeds: Sequence[int], *, quantile: float
) -> FamilywiseCalibration:
    """Build the null bank, standardise from it alone, freeze Q_quantile(T_max) [02 §C4]."""
    if len(master_seeds) < 2:
        raise DryRunError("a calibration bank needs at least two trials")
    bank = [null_statistics(generate_null_trial(design, seed)) for seed in master_seeds]
    columns = {
        name: np.array([trial[name] for trial in bank], dtype=np.float64)
        for name in NULL_STATISTIC_NAMES
    }
    centre = {name: float(np.mean(values)) for name, values in columns.items()}
    scale = {name: float(np.std(values, ddof=1)) for name, values in columns.items()}
    partial = FamilywiseCalibration(
        n_trials=len(bank),
        seeds=tuple(int(seed) for seed in master_seeds),
        quantile=quantile,
        centre=centre,
        scale=scale,
        c_fwer=float("inf"),
    )
    t_values = np.array([partial.t_max(trial) for trial in bank], dtype=np.float64)
    return FamilywiseCalibration(
        n_trials=len(bank),
        seeds=tuple(int(seed) for seed in master_seeds),
        quantile=quantile,
        centre=centre,
        scale=scale,
        c_fwer=float(np.percentile(t_values, 100.0 * quantile)),
    )


def marginal_diagnostic_tolerances(
    design: TrialDesign, master_seeds: Sequence[int], *, quantile: float
) -> dict[str, float]:
    """NON_GATING_DIAGNOSTIC.

    The 00 §34B.1 marginal percentile bank, retained for comparison against the §34B.1A
    literals. It decides nothing: 02 §C4 replaced marginal per-statistic gates with the
    family-wise max-statistic, and no readiness decision reads this function.
    """
    bank = [null_statistics(generate_null_trial(design, seed)) for seed in master_seeds]
    out: dict[str, float] = {}
    for name in NULL_STATISTIC_NAMES:
        column = np.array([trial[name] for trial in bank], dtype=np.float64)
        magnitudes = column if name in ONE_SIDED_STATISTICS else np.abs(column)
        out[name] = float(np.percentile(magnitudes, 100.0 * quantile))
    return out


# ----------------------------------------------------------------------------------------
# Cell-cluster confidence intervals [AUTH: 00 §34B.1 "Null confidence intervals"]
# ----------------------------------------------------------------------------------------


def _cluster_units(
    rows: Sequence[TrialRow], residual_by_row: Sequence[float | None]
) -> list[ClusterUnit]:
    """One unit per registry cell, carrying every seed observation of that cell."""
    grouped: dict[str, list[tuple[float, float, float, float, float]]] = {}
    for row, residual in zip(rows, residual_by_row, strict=True):
        grouped.setdefault(row.cell, []).append(
            (
                row.error,
                row.recovery,
                row.d_l,
                row.d_r,
                float("nan") if residual is None else residual,
            )
        )
    return [ClusterUnit(cell=cell, rows=tuple(values)) for cell, values in sorted(grouped.items())]


def cell_cluster_intervals(
    rows: Sequence[TrialRow], *, n_replicates: int, seed: int, alpha: float
) -> dict[str, tuple[float, float]]:
    """95% cell-cluster intervals for Spearman, each operator's mean u, D_L and D_R.

    Cells are sampled with replacement and every seed row of a sampled cell is retained, which
    preserves the dependence the planted design deliberately creates [AUTH: 00 §34B.1].
    """
    curves = seed_curves(rows)
    residuals = operator_residuals(
        [(f"{r.cell}:{r.seed}", r.operator, r.seed, r.error, r.recovery) for r in rows], curves
    )
    residual_by_row = [r.value for r in residuals]
    operators = sorted({row.operator for row in rows})
    operator_of_cell = {row.cell: row.operator for row in rows}
    units = _cluster_units(rows, residual_by_row)

    intervals: dict[str, tuple[float, float]] = {}

    def rho(sampled: Sequence[ClusterUnit]) -> float:
        values = cluster_rows(sampled)
        return spearman([v[0] for v in values], [v[1] for v in values])

    intervals["spearman_rho"] = bootstrap(
        rho, units, n_replicates=n_replicates, seed=seed, alpha=alpha
    ).interval

    for index, name in ((2, "mean_d_l"), (3, "mean_d_r")):

        def gap(sampled: Sequence[ClusterUnit], position: int = index) -> float:
            return float(np.mean([v[position] for v in cluster_rows(sampled)]))

        intervals[name] = bootstrap(
            gap, units, n_replicates=n_replicates, seed=seed + index, alpha=alpha
        ).interval

    for offset, operator in enumerate(operators):
        chosen = [unit for unit in units if operator_of_cell[unit.cell] == operator]
        if not chosen:
            continue

        def mean_u(sampled: Sequence[ClusterUnit]) -> float:
            values = [v[4] for v in cluster_rows(sampled)]
            finite = [v for v in values if np.isfinite(v)]
            return float(np.mean(finite)) if finite else 0.0

        intervals[f"mean_u[{operator}]"] = bootstrap(
            mean_u, chosen, n_replicates=n_replicates, seed=seed + 100 + offset, alpha=alpha
        ).interval
    return intervals


# ----------------------------------------------------------------------------------------
# DRY-H4 — finite-sample fixed-FPR estimator calibration [AUTH: 02 §C3]
# ----------------------------------------------------------------------------------------


def gaussian_membership_scores(
    *, n_member: int, n_non_member: int, separation: float, seed: int, round_to: float | None = None
) -> tuple[FloatArray, NDArray[np.int_]]:
    """Non-members ~ N(0,1), members ~ N(separation,1). Higher score = more member-like.

    `round_to` discretises the scores onto a grid, which is how the tied-score stress in
    02 §C3 H4B manufactures exact ties across both classes.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    non_member = normal(rng, n_non_member)
    member = normal(rng, n_member) + separation
    scores = np.concatenate([member, non_member])
    labels = np.concatenate(
        [np.ones(n_member, dtype=np.int_), np.zeros(n_non_member, dtype=np.int_)]
    )
    if round_to is not None:
        if round_to <= 0.0:
            raise DryRunError("round_to must be positive")
        scores = np.round(scores / round_to) * round_to
    return scores.astype(np.float64), labels


def analytic_true_positive_rate(separation: float, target: float) -> float:
    """Closed-form TPR at the exact target operating point, independent of the estimator.

    With non-members ~ N(0,1) and members ~ N(mu,1), the threshold at false-positive rate p
    is ndtri(1-p) and the power is ndtr(mu - threshold). Derived here rather than measured,
    so the estimator is checked against mathematics and not against itself.
    """
    from scipy.special import ndtr, ndtri  # type: ignore[import-untyped]

    threshold = float(ndtri(1.0 - target))
    return float(ndtr(separation - threshold))


@dataclass(frozen=True)
class EstimatorCalibration:
    """Finite-sample behaviour of the fixed-FPR estimator at one sample size."""

    n_non_member: int
    n_simulations: int
    analytic: float
    signed_bias: float
    absolute_error: float
    rmse: float
    percentiles: tuple[float, float, float]


def calibrate_fixed_point_estimator(
    *,
    sample_sizes: Sequence[int],
    n_simulations: int,
    separation: float,
    target: float,
    seed_start: int,
    round_to: float | None = None,
) -> list[EstimatorCalibration]:
    """02 §C3: sweep sample size, record bias, absolute error, RMSE and percentiles."""
    from src.scoring.roc import tpr_at_fixed_fpr

    analytic = analytic_true_positive_rate(separation, target)
    out: list[EstimatorCalibration] = []
    for size in sample_sizes:
        estimates = np.empty(n_simulations, dtype=np.float64)
        for index in range(n_simulations):
            scores, labels = gaussian_membership_scores(
                n_member=size,
                n_non_member=size,
                separation=separation,
                seed=seed_start + index,
                round_to=round_to,
            )
            estimates[index] = tpr_at_fixed_fpr(scores, labels, target).true_positive_rate
        errors = estimates - analytic
        out.append(
            EstimatorCalibration(
                n_non_member=size,
                n_simulations=n_simulations,
                analytic=analytic,
                signed_bias=float(np.mean(errors)),
                absolute_error=float(np.mean(np.abs(errors))),
                rmse=float(np.sqrt(np.mean(errors**2))),
                percentiles=(
                    float(np.percentile(estimates, 2.5)),
                    float(np.percentile(estimates, 50.0)),
                    float(np.percentile(estimates, 97.5)),
                ),
            )
        )
    return out


# ----------------------------------------------------------------------------------------
# Positive-control power calibration [AUTH: 02 §C5; 00 §34B.1A]
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PositiveControl:
    """A planted-effect bank and the percentile threshold frozen from it [AUTH: 02 §C5]."""

    name: str
    n_trials: int
    quantile: float
    threshold: float
    values: tuple[float, ...]

    @property
    def in_bank_fraction(self) -> float:
        """Fraction of the CALIBRATION bank above its own percentile.

        This is arithmetic, not evidence: the 2.5th percentile of a bank always leaves about
        97.5% of that same bank above it. Independent power is `detection_fraction` measured
        on trials outside the bank.
        """
        return float(np.mean(np.asarray(self.values) > self.threshold))


def calibrate_operator_positive_control(
    design: TrialDesign,
    master_seeds: Sequence[int],
    *,
    slope: float,
    planted_effect: float,
    quantile: float,
) -> PositiveControl:
    """DRY-B power against an INDEPENDENT magnitude baseline [AUTH: 02 §C5; 00 §29.2].

    Each trial fits the curve from its own calibration ladder and then measures a disjoint
    set of operator observations against it, so the planted residual cannot be absorbed into
    the baseline it is judged by.
    """
    target = design.operators[0]
    values: list[float] = []
    for seed in master_seeds:
        curves = baseline_curves(generate_calibration_ladder(design, seed, slope=slope))
        observations = generate_operator_observations(
            design, seed + 500_000, slope=slope, operator_effect={target: planted_effect}
        )
        values.append(residuals_against_baseline(observations, curves).get(target, 0.0))
    array = np.asarray(values, dtype=np.float64)
    return PositiveControl(
        name=f"operator_residual[{target}]",
        n_trials=len(values),
        quantile=quantile,
        threshold=float(np.percentile(array, quantile)),
        values=tuple(float(v) for v in array),
    )


def operator_detection_fraction(
    design: TrialDesign,
    master_seeds: Sequence[int],
    *,
    slope: float,
    planted_effect: float,
    threshold: float,
) -> float:
    """Out-of-bank power: detection on trials that did not build the threshold."""
    target = design.operators[0]
    detected = 0
    for seed in master_seeds:
        curves = baseline_curves(generate_calibration_ladder(design, seed, slope=slope))
        observations = generate_operator_observations(
            design, seed + 500_000, slope=slope, operator_effect={target: planted_effect}
        )
        if residuals_against_baseline(observations, curves).get(target, 0.0) > threshold:
            detected += 1
    return detected / len(master_seeds)


def calibrate_h1_positive_control(
    *,
    master_seeds: Sequence[int],
    n_member: int,
    n_non_member: int,
    separation: float,
    target: float,
    quantile: float,
) -> PositiveControl:
    """DRY-H1 power: the distribution of |observed - analytic| under the planted separation.

    Regenerated under the final estimator, as 02 §C5 requires. The frozen assertion is an
    upper bound, so the threshold is the `quantile`-th percentile of the absolute error.
    """
    from src.scoring.roc import tpr_at_fixed_fpr

    analytic = analytic_true_positive_rate(separation, target)
    values: list[float] = []
    for seed in master_seeds:
        scores, labels = gaussian_membership_scores(
            n_member=n_member, n_non_member=n_non_member, separation=separation, seed=seed
        )
        observed = tpr_at_fixed_fpr(scores, labels, target).true_positive_rate
        values.append(abs(observed - analytic))
    array = np.asarray(values, dtype=np.float64)
    return PositiveControl(
        name="h1_absolute_error",
        n_trials=len(values),
        quantile=quantile,
        threshold=float(np.percentile(array, quantile)),
        values=tuple(float(v) for v in array),
    )


def h1_detection_fraction(
    *,
    master_seeds: Sequence[int],
    n_member: int,
    n_non_member: int,
    separation: float,
    target: float,
    threshold: float,
) -> float:
    """Out-of-bank fraction of trials whose absolute error stays inside the frozen bound."""
    from src.scoring.roc import tpr_at_fixed_fpr

    analytic = analytic_true_positive_rate(separation, target)
    inside = 0
    for seed in master_seeds:
        scores, labels = gaussian_membership_scores(
            n_member=n_member, n_non_member=n_non_member, separation=separation, seed=seed
        )
        observed = tpr_at_fixed_fpr(scores, labels, target).true_positive_rate
        if abs(observed - analytic) <= threshold:
            inside += 1
    return inside / len(master_seeds)


# ----------------------------------------------------------------------------------------
# DRY-G verdict — T_max <= c_FWER, CIs diagnostic only [AUTH: 02 §C4; 00 §34B.1]
# ----------------------------------------------------------------------------------------

DRY_G_PASS: Final = "PASS"
DRY_G_INVESTIGATE: Final = "PASS_WITH_INVESTIGATION"
DRY_G_FAIL: Final = "FAIL"


@dataclass(frozen=True)
class TrialVerdict:
    seed: int
    passed: bool
    t_max: float
    statistics: Mapping[str, float]
    standardised: Mapping[str, float]
    #: Reported, never gating [AUTH: 02 §C4 "not a second pass/fail criterion"].
    diagnostic_intervals: Mapping[str, tuple[float, float]]


@dataclass(frozen=True)
class DryGVerdict:
    """3/3 -> PASS, 2/3 -> PASS_WITH_INVESTIGATION, <=1/3 -> FAIL [AUTH: 00 §34B.1]."""

    status: str
    trials: tuple[TrialVerdict, ...]
    calibration: FamilywiseCalibration

    @property
    def n_passed(self) -> int:
        return sum(1 for trial in self.trials if trial.passed)

    def investigation_note(self) -> str:
        """Why a failed trial is a null fluctuation and not an implementation bug."""
        lines: list[str] = []
        for trial in self.trials:
            if trial.passed:
                continue
            worst = max(trial.standardised.items(), key=lambda item: item[1])
            lines.append(
                f"seed {trial.seed}: T_max={trial.t_max:.6f} > c_FWER="
                f"{self.calibration.c_fwer:.6f}, driven by {worst[0]} (Z={worst[1]:.4f})"
            )
        return "\n".join(lines)


def evaluate_dry_g(
    design: TrialDesign,
    calibration: FamilywiseCalibration,
    validation_seeds: Sequence[int],
    *,
    n_replicates: int,
    alpha: float,
) -> DryGVerdict:
    """Run the frozen validation trials against the frozen c_FWER [AUTH: 02 §C4].

    Validation seeds must not appear in the calibration bank: a trial calibrated against
    itself is not a test.
    """
    overlap = sorted(set(validation_seeds) & set(calibration.seeds))
    if overlap:
        raise DryRunError(
            f"validation seed(s) {overlap} are inside the calibration bank; the bank must be"
            " independent of the trials it judges [AUTH: 02 §C4]"
        )
    trials: list[TrialVerdict] = []
    for seed in validation_seeds:
        rows = generate_null_trial(design, seed)
        statistics = null_statistics(rows)
        standardised = calibration.standardise(statistics)
        t_max = float(max(standardised.values()))
        trials.append(
            TrialVerdict(
                seed=seed,
                passed=t_max <= calibration.c_fwer,
                t_max=t_max,
                statistics=statistics,
                standardised=standardised,
                diagnostic_intervals=cell_cluster_intervals(
                    rows, n_replicates=n_replicates, seed=seed, alpha=alpha
                ),
            )
        )
    passed = sum(1 for trial in trials if trial.passed)
    total = len(trials)
    if passed == total:
        status = DRY_G_PASS
    elif passed * 3 >= total * 2:
        status = DRY_G_INVESTIGATE
    else:
        status = DRY_G_FAIL
    return DryGVerdict(status=status, trials=tuple(trials), calibration=calibration)


# ----------------------------------------------------------------------------------------
# Dry-run coverage status — machine readable, authority consistent
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# DRY-D — spectral-tail effect [AUTH: 00 §34B.1 DRY-D, §28B.4]
# ----------------------------------------------------------------------------------------

#: DRY-D plants truncation points below the rank-matched privacy curve and reads
#: Delta_tail_priv for the reference and Min-K% scores [AUTH: 00 §34B.1 DRY-D, §28B]. Those
#: quantities are defined over O3 SVD-truncation merges and their recoveries, which are S07
#: and S08 objects. Block B could not build them, so DRY-D carried an unmet dependency.
#:
#: S07 (merge operators) and S08 (recovery) now exist, so the dependency checkpoint is
#: satisfied and the already-frozen scenario is activated. Nothing about the statistic, the
#: rank-matched curve semantics, the reference/Min-K interpretation or the thresholds changed.
#:
#: The wording is deliberately narrow. 00 §28B.4 defines
#:
#:     Delta_tail_priv(s) = R_priv^trunc(s) - f_priv^rank(e_floor(s))
#:
#: over the TRUNCATED O3 artifact's privacy response against the rank-matched calibration
#: curve evaluated at e_floor(s). It does not consume a recovered A or a solver error. So S07
#: supplies the truncation geometry the statistic is defined on, and S08 satisfies the planned
#: recovery-stage dependency checkpoint and supplies the wider O3 diagnostic and recovery
#: implementation the scenario sits inside — not the statistic's own inputs.
DRY_D_STATUS: Final = "COVERED"
DRY_D_REASON: Final = (
    "DRY-D is active: S07 O3 SVD-truncation merges supply the real truncation geometry"
    " e_floor(s) that Delta_tail_priv is evaluated at, and S08 completes the planned"
    " recovery-stage dependency checkpoint and the wider O3 diagnostic implementation. The"
    " Delta_tail_priv statistic itself compares the truncated artifact's privacy response"
    " against the rank-matched calibration curve; it does not consume a recovered A or a"
    " solver error [AUTH: 00 §34B.1 DRY-D, §28B.4; 01 §39 S07, S08]"
)

#: The 00 §34B.1 DRY-D cutoffs now live in `configs/p0/dry_run.json`, because DRY-D1's pair
#: is CALIBRATED — the 97.5th percentile of a 200-trial planted-effect bank — and 02 §C5 binds
#: that bank to the implementation it was generated from. Keeping a calibrated number as a
#: source literal is what let the 10-rank implementation's cutoffs survive the correction to
#: the authoritative 12-rank schedule. DRY-D2's pair is a structural spec constant that 02 §C5
#: does not cover; it is moved for consistency and its values are unchanged.
#:
#: The superseded pair is recorded here as history, never consumed:
#:
#:     DRY-D1 reference  -0.136213   SUPERSEDED_PRE_DATA_BY_02_C5_RECALIBRATION
#:     DRY-D1 Min-K%     -0.135921   SUPERSEDED_PRE_DATA_BY_02_C5_RECALIBRATION
#:
#: 00 §34B.1A still states them; the implementation evidence records that binding clarification
#: 02 §C5 superseded them pre-data, after the retained-rank schedule defect was measured.
SUPERSEDED_DRY_D1_REFERENCE_MAX: Final = -0.136213
SUPERSEDED_DRY_D1_MINK_MAX: Final = -0.135921
SUPERSEDED_THRESHOLD_STATUS: Final = "SUPERSEDED_PRE_DATA_BY_02_C5_RECALIBRATION"


@dataclass(frozen=True)
class TailThresholds:
    """The four resolved DRY-D cutoffs [AUTH: 00 §34B.1; 02 §C5]."""

    d1_reference: float
    d1_mink: float
    d2_reference: float
    d2_mink_abs: float


def tail_thresholds(root: Path | None = None) -> TailThresholds:
    """Resolve the frozen DRY-D cutoffs. One source of truth, version-controlled [01 §17]."""
    from src.analysis.settings import dry_run_settings

    settings = dry_run_settings(root or Path(__file__).resolve().parents[2])
    return TailThresholds(
        d1_reference=settings.number("dry_d1_reference_threshold"),
        d1_mink=settings.number("dry_d1_mink_threshold"),
        d2_reference=settings.number("dry_d2_reference_threshold"),
        d2_mink_abs=settings.number("dry_d2_mink_abs_threshold"),
    )


DRY_D_PASS: Final = "TAIL_EFFECT_CORROBORATED"
DRY_D_BASE_GEOMETRY: Final = "BASE_GEOMETRY_SENSITIVE"
DRY_D_FAIL: Final = "TAIL_EFFECT_NOT_DETECTED"

#: The complete 00 §28B.1 retained-rank ladder. s = 32 is the oracle anchor (T_32(A) = A for a
#: rank-32 adapter) and s = 0 is the zero-delta base anchor; neither is a truncation point that
#: needs new scoring artifacts, so DRY-D runs over the TWELVE interior ranks between them.
DRY_D_ANCHOR_RANKS: Final[tuple[int, ...]] = (32, 0)
DRY_D_INTERIOR_RANKS: Final[tuple[int, ...]] = (31, 28, 24, 20, 16, 12, 8, 6, 4, 3, 2, 1)
DRY_D_RETAINED_RANKS: Final[tuple[int, ...]] = DRY_D_INTERIOR_RANKS

#: IMPLEMENTATION_ADJUDICATION_PRE_DATA. 00 §28B.4 defines Delta_tail_priv(s) per retained
#: rank; 00 §34B.1 states one scalar cutoff per score. The trial statistic is the mean over the
#: schedule, matching the house convention DRY-B already uses ("mean planted operator
#: residual"). The per-rank values are reported beside it and nothing is discarded.
DRY_D_TRIAL_STATISTIC: Final = "MEAN_DELTA_TAIL_PRIV_OVER_RETAINED_RANK_SCHEDULE"

#: Sweeps for the planted dry-run solve. Not a scientific value: it drives a fixture context.
DRY_D_FIXTURE_ITERATIONS: Final = 60


@dataclass(frozen=True)
class TailTrial:
    """One planted spectral-tail trial [AUTH: 00 §28B.4, §34B.1 DRY-D].

    `e_floor` and `solver_error` come from real S07 truncations and real S08 recoveries. The
    privacy responses are planted, exactly as DRY-A and DRY-B plant theirs: a dry run has no
    model to score.
    """

    retained_ranks: tuple[int, ...]
    e_floor: tuple[float, ...]
    solver_error: tuple[float, ...]
    reference_delta: tuple[float, ...]
    mink_delta: tuple[float, ...]

    @property
    def mean_reference_delta(self) -> float:
        return float(np.mean(self.reference_delta))

    @property
    def mean_mink_delta(self) -> float:
        return float(np.mean(self.mink_delta))

    def as_dict(self) -> dict[str, object]:
        return {
            "statistic": DRY_D_TRIAL_STATISTIC,
            "retained_ranks": list(self.retained_ranks),
            "e_floor": list(self.e_floor),
            "solver_error": list(self.solver_error),
            "reference_delta_tail_priv": list(self.reference_delta),
            "mink_delta_tail_priv": list(self.mink_delta),
            "mean_reference_delta_tail_priv": self.mean_reference_delta,
            "mean_mink_delta_tail_priv": self.mean_mink_delta,
        }


def _tail_geometry(
    master_seed: int, retained_ranks: Sequence[int]
) -> tuple[list[float], list[float]]:
    """Real e_floor(s) and real S08 solver error per retained rank.

    This is the S07/S08 dependency checkpoint DRY-D was waiting on: the O3 truncations
    genuinely exist, so the spectral-tail axis the statistic is evaluated at is measured rather
    than assumed. The solver error is reported alongside as a diagnostic; 00 §28B.4's
    Delta_tail_priv does not consume it. Imported inside the function so Block B carries no
    module-level dependency on later stages.

    The surface is 40x36 with a rank-32 protected constituent, matching the 01 §8B LoRA rank
    the 00 §28B.1 ladder is written for, so retaining 31 or 28 directions is a real truncation
    rather than a no-op clamp against a smaller matrix dimension.
    """
    from pathlib import Path

    from src.merge.context import resolve_merge_context
    from src.merge.family import MergeSpec, build_release_family
    from src.merge.updates import fixture_induced_update
    from src.recovery.context import fixture_recovery_context
    from src.recovery.evaluation import o3_error_report
    from src.recovery.observations import observe_incomplete_fixed
    from src.recovery.settings import recovery_settings
    from src.recovery.solvers import recover_c2_fixed_alpha
    from src.recovery.truth import bind_truth

    root = Path(__file__).resolve().parents[2]
    merge_context = resolve_merge_context(root)
    # A dry run exercises the frozen algorithm on planted fixtures; the production sweep count
    # is for real artifacts. The fixture context is labelled FIXTURE_ONLY_NOT_SCIENTIFIC, so
    # nothing solved here can be read as the accepted scientific method [AUTH: 01 §17].
    recovery_context = fixture_recovery_context(
        recovery_settings(root).document, c2_n_iters=DRY_D_FIXTURE_ITERATIONS
    )

    rng = np.random.Generator(np.random.PCG64(master_seed))
    names = ("w0", "w1")
    rows, columns = 40, 36

    def low_rank(rank: int) -> dict[str, FloatArray]:
        """An exactly rank-`rank` induced update, as ΔW = BA is by construction."""
        return {
            name: rng.normal(size=(rows, rank)) @ rng.normal(size=(rank, columns)) for name in names
        }

    protected = fixture_induced_update(low_rank(32), context=merge_context, origin="A")
    partners = {
        f"B{index}": fixture_induced_update(low_rank(8), context=merge_context, origin=f"B{index}")
        for index in range(4)
    }

    floors: list[float] = []
    errors: list[float] = []
    for retained in retained_ranks:
        family = build_release_family(
            protected=protected,
            partners=partners,
            specs=[
                MergeSpec(
                    descendant_id=f"D{index}",
                    operator="O3_SVD_TRUNC_MERGE",
                    alpha=0.5,
                    partner_id=f"B{index}",
                    retained_rank=retained,
                )
                for index in range(4)
            ],
            permitted_k=(4,),
            context=merge_context,
        )
        observation = observe_incomplete_fixed(family=family, context=recovery_context)
        recovered = recover_c2_fixed_alpha(observation)
        binding = bind_truth(observation=observation, source=family, protected_truth=protected)
        report = o3_error_report(recovered, binding, o3_source=family)
        floors.append(float(str(report["e_floor"])))
        errors.append(float(str(report["e_solver_retained"])))
    return floors, errors


def generate_tail_trial(
    master_seed: int,
    *,
    reference_shift: float,
    mink_shift: float,
    retained_ranks: Sequence[int] = DRY_D_RETAINED_RANKS,
) -> TailTrial:
    """Plant truncation points below the rank-matched privacy curve [AUTH: 00 §34B.1 DRY-D].

    Delta_tail_priv(s) = R_priv^trunc(s) - f_priv^rank(e_floor(s)) [AUTH: 00 §28B.4]. The
    planted truncation response sits `shift` below the rank-matched curve evaluated at the real
    e_floor(s), so the planted deviation IS the statistic — which is what "plant truncation
    points below the rank-matched curve" means.

    `mink_shift` is separate so the geometry-confounded scenario can move the reference score
    while leaving the Min-K% score near zero [AUTH: 00 §28B.4 corroboration rule].
    """
    floors, errors = _tail_geometry(master_seed, retained_ranks)
    rng = np.random.Generator(np.random.PCG64(master_seed + 1))

    # The rank-matched additive ladder: a planted non-increasing recovery-against-error curve,
    # fitted with the frozen S04 isotonic implementation.
    ladder_error = np.linspace(0.0, 1.0, 24)
    ladder_recovery = 1.0 - 0.8 * ladder_error
    curve = fit_monotone_curve(ladder_error, ladder_recovery)

    reference: list[float] = []
    mink: list[float] = []
    for index, floor in enumerate(floors):
        matched = curve.predict(min(max(floor, 0.0), 1.0))
        noise = float(normal(rng, 2)[0]) * 0.01
        reference.append((matched + reference_shift + noise) - matched)
        mink.append((matched + mink_shift + noise * 0.5) - matched)
        del index
    return TailTrial(
        retained_ranks=tuple(retained_ranks),
        e_floor=tuple(floors),
        solver_error=tuple(errors),
        reference_delta=tuple(reference),
        mink_delta=tuple(mink),
    )


@dataclass(frozen=True)
class TailVerdict:
    """The frozen DRY-D1 / DRY-D2 decision [AUTH: 00 §34B.1]."""

    scenario: str
    status: str
    trial: TailTrial

    def as_dict(self) -> dict[str, object]:
        return {"scenario": self.scenario, "status": self.status, **self.trial.as_dict()}


def evaluate_dry_d1(trial: TailTrial, *, thresholds: TailThresholds | None = None) -> TailVerdict:
    """Both directions must corroborate against the frozen cutoffs [AUTH: 00 §34B.1].

    The cutoffs are resolved from config, not regenerated: calibration happens once, pre-data,
    and validation consumes the frozen result [AUTH: 02 §C5].
    """
    limits = thresholds or tail_thresholds()
    corroborated = (
        trial.mean_reference_delta < limits.d1_reference and trial.mean_mink_delta < limits.d1_mink
    )
    return TailVerdict(
        scenario="DRY-D1",
        status=DRY_D_PASS if corroborated else DRY_D_FAIL,
        trial=trial,
    )


def evaluate_dry_d2(trial: TailTrial, *, thresholds: TailThresholds | None = None) -> TailVerdict:
    """Reference moves, Min-K% does not: BASE_GEOMETRY_SENSITIVE [AUTH: 00 §34B.1, §28B.4].

    Not recalibrated: 02 §C5 covers positive-control power calibration, and this pair is a
    structural spec constant rather than a percentile of a planted bank.
    """
    limits = thresholds or tail_thresholds()
    confounded = (
        trial.mean_reference_delta < limits.d2_reference
        and abs(trial.mean_mink_delta) < limits.d2_mink_abs
    )
    return TailVerdict(
        scenario="DRY-D2",
        status=DRY_D_BASE_GEOMETRY if confounded else DRY_D_FAIL,
        trial=trial,
    )


@dataclass(frozen=True)
class TailPositiveControl:
    """One arm of the DRY-D1 planted-effect bank and the cutoff frozen from it [02 §C5]."""

    name: str
    n_trials: int
    quantile: float
    threshold: float
    values: tuple[float, ...]

    @property
    def empirical_power(self) -> float:
        """Fraction of planted-effect trials that would be detected at this cutoff.

        The planted DRY-D1 effect is NEGATIVE, so detection is `statistic < threshold`.
        """
        return float(np.mean(np.asarray(self.values) < self.threshold))

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_trials": self.n_trials,
            "quantile": self.quantile,
            "threshold": self.threshold,
            "empirical_power": self.empirical_power,
            "values": list(self.values),
        }


#: numpy's default. Recorded explicitly so the cutoff is reproducible from the bank alone.
CALIBRATION_QUANTILE_METHOD: Final = "numpy.percentile(method='linear')"


def calibrate_dry_d1_positive_control(
    master_seeds: Sequence[int], *, planted_shift: float, quantile: float
) -> dict[str, TailPositiveControl]:
    """DRY-D1 power against the planted tail effect [AUTH: 02 §C5; 00 §34B.1A].

    This calls `generate_tail_trial` — the SAME canonical statistic validation runs, over the
    same twelve interior ranks and the same planted-effect model. Only the master seed differs,
    so there is no separate, easier calibration statistic.

    Both readouts are calibrated independently: the reference and Min-K% arms carry different
    noise (the generator scales the Min-K perturbation by half), so forcing one cutoff on both
    would mis-state the power of at least one of them.
    """
    reference: list[float] = []
    mink: list[float] = []
    for seed in master_seeds:
        trial = generate_tail_trial(seed, reference_shift=planted_shift, mink_shift=planted_shift)
        reference.append(trial.mean_reference_delta)
        mink.append(trial.mean_mink_delta)
    return {
        "reference": TailPositiveControl(
            name="dry_d1_reference_delta_tail_priv",
            n_trials=len(reference),
            quantile=quantile,
            threshold=float(np.percentile(np.asarray(reference, dtype=np.float64), quantile)),
            values=tuple(reference),
        ),
        "mink": TailPositiveControl(
            name="dry_d1_mink_delta_tail_priv",
            n_trials=len(mink),
            quantile=quantile,
            threshold=float(np.percentile(np.asarray(mink, dtype=np.float64), quantile)),
            values=tuple(mink),
        ),
    }


#: Every scenario 00 §34B.2 requires before the P0-PRE gate may be considered.
REQUIRED_DRY_SCENARIOS: Final[tuple[str, ...]] = (
    "DRY-A",
    "DRY-B",
    "DRY-C",
    "DRY-D",
    "DRY-E",
    "DRY-F",
    "DRY-G",
    "DRY-H",
    "DRY-H4",
    "DRY-CACHE",
)

#: Scenarios the synthetic lane implements and runs. DRY-D joined once S07 and S08 supplied
#: the O3 truncations and recoveries it is defined over [AUTH: 00 §34B.1 DRY-D; 01 §39].
COVERED_DRY_SCENARIOS: Final[tuple[str, ...]] = REQUIRED_DRY_SCENARIOS
