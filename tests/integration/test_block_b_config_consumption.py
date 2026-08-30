"""C2 — every active config field is proven to control real execution [AUTH: 01 §17].

The matrix below is not a name check. For each field it:

1. copies `configs/` to a temporary root;
2. observes the OWNING EXECUTION PATH on the pristine copy;
3. mutates the field to a value derived programmatically from the repository value, so it
   differs from both the frozen value and every literal used elsewhere in the suite;
4. observes the same execution path again;
5. requires the observation to move, or the mutated configuration to be rejected outright.

A field whose mutation changes nothing observable is INERT and fails the run. The final
assertion is `INERT_ACTIVE_CONFIG_FIELDS == []`.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.analysis.dryrun import (
    baseline_curves,
    calibrate_familywise,
    calibrate_fixed_point_estimator,
    calibrate_h1_positive_control,
    calibrate_operator_positive_control,
    cell_cluster_intervals,
    evaluate_dry_g,
    gaussian_membership_scores,
    generate_calibration_ladder,
    generate_monotone_trial,
    generate_null_trial,
    generate_operator_observations,
    marginal_diagnostic_tolerances,
    operator_detection_fraction,
    residuals_against_baseline,
)
from src.analysis.isotonic import crossing_point, fit_monotone_curve
from src.analysis.metrics import (
    denominator_stability,
    functional_eligibility,
    privacy_eligibility,
    r_priv_for_condition,
)
from src.analysis.resampling import PairedUnit, bootstrap, paired_difference
from src.analysis.settings import (
    ANALYSIS_CONFIG,
    DRY_RUN_CONFIG,
    SCORING_CONFIG,
    analysis_constants,
    dry_run_settings,
    scoring_constants,
    trial_design,
)
from src.provenance.hashing import JSONValue, sha256_canonical
from src.scoring.backends import ConstantBackend, SyntheticBackend
from src.scoring.crossfit import Channel, Partition, plan_folds, run_outer_crossfit
from src.scoring.engine import Arm, ScoreFamily, ScoringContext, View, score_records
from src.scoring.pooled import fit_ridge_pool, select_pooled_method

#: Fields that describe the document rather than the mathematics.
METADATA_KEYS = frozenset({"authority", "description", "legacy_marginal_status"})

#: Removed because the mechanism they configured no longer exists.
OBSOLETE_REMOVED = ("score_families", "seed_balance_bins", "beyond_calibration_error")

RECORDS = [f"mem-{i:03d}" for i in range(10)] + [f"non-{i:03d}" for i in range(10)]
MEMBERSHIP = {r: (1 if r.startswith("mem-") else 0) for r in RECORDS}


def _population(n: int, separation: float, seed: int) -> tuple[dict[str, float], dict[str, int]]:
    scores, labels = gaussian_membership_scores(
        n_member=n, n_non_member=n, separation=separation, seed=seed
    )
    ids = [f"mem-{i:05d}" for i in range(n)] + [f"non-{i:05d}" for i in range(n)]
    return (
        {r: float(v) for r, v in zip(ids, scores, strict=True)},
        {r: int(v) for r, v in zip(ids, labels, strict=True)},
    )


POPULATION, LABELS = _population(240, 1.6, 424242)

#: `src/` always lives at the repository root; only `configs/` is copied per case.
SOURCE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Field:
    """One config field, its owning execution path, and how to move it."""

    name: str
    config: str
    consumer: str
    mutate: Callable[[Any], Any]
    observe: Callable[[Path], Any]


def _scale(factor: float) -> Callable[[Any], Any]:
    return lambda value: type(value)(value * factor) if isinstance(value, int) else value * factor


def _offset(delta: int) -> Callable[[Any], Any]:
    return lambda value: value + delta


# ------------------------------------------------------------------ observation helpers
def _obs_endpoint(root: Path) -> Any:
    constants = scoring_constants(root)
    plan = plan_folds(list(POPULATION), constants.n_outer_folds, constants.fold_assignment_seed)
    outcome = run_outer_crossfit(
        channels=[Channel("ONLY", POPULATION)],
        labels=LABELS,
        plan=plan,
        target=constants.target_fpr,
    )
    return (
        round(outcome.true_positive_rate, 12),
        len(outcome.per_fold),
        round(outcome.aggregate_realised_false_positive_rate, 12),
        sorted(plan.assignment.items())[:5],
    )


def _obs_min_k(root: Path) -> Any:
    settings = scoring_settings_document(root)
    context = ScoringContext(
        root=SOURCE_ROOT,
        backend=SyntheticBackend(label="c2", tokens_per_record=16, member_shift=0.7),
        reference_backend=ConstantBackend(label="ref", tokens_per_record=16),
        resolved_config=settings,
        model_revision="a" * 40,
        tokenizer_sha256="1" * 64,
        environment_lock_sha256="6" * 64,
    )
    table = score_records(
        arm=Arm.CANARY,
        score=ScoreFamily.MIN_K,
        view=View.SYNTHETIC,
        artifact="a",
        seed=101,
        fold=0,
        record_ids=RECORDS,
        membership=MEMBERSHIP,
        context=context,
        cache=None,
    )
    return (
        table.cache_key,
        context.precision,
        context.max_sequence_length,
        tuple(round(row.scalar_score, 12) for row in table.rows),
    )


def scoring_settings_document(root: Path) -> Mapping[str, JSONValue]:
    from src.analysis.settings import scoring_settings

    return scoring_settings(root).document


def _obs_ridge(root: Path) -> Any:
    constants = scoring_constants(root)
    members = [r for r in POPULATION if LABELS[r] == 1]
    non_members = [r for r in POPULATION if LABELS[r] == 0]
    partition = Partition(
        calibration_ids=tuple(members[:160] + non_members[:160]),
        evaluation_ids=tuple(members[160:] + non_members[160:]),
        fold=0,
    )
    ids = list(partition.calibration_ids) + list(partition.evaluation_ids)
    other = {r: v * 0.4 + 0.1 for r, v in POPULATION.items()}
    selection = select_pooled_method(
        descendant_scores=[POPULATION, other],
        partition=partition,
        labels=LABELS,
        target=constants.target_fpr,
        regularisation=constants.ridge_regularisation,
    )
    # The pooled selection AND the ridge coefficients the setting actually reaches
    # [AUTH: 00 §15.3].
    features = np.column_stack([[POPULATION[r] for r in ids], [other[r] for r in ids]])
    marks = np.array([LABELS[r] for r in ids], dtype=np.int_)
    coefficients = fit_ridge_pool(features, marks, constants.ridge_regularisation)
    return (
        selection.method,
        round(selection.calibration_true_positive_rate, 12),
        round(sum(selection.scores.values()), 9),
        tuple(round(float(value), 9) for value in coefficients),
    )


def _obs_r_priv(root: Path) -> Any:
    constants = analysis_constants(root)
    verdict = privacy_eligibility(
        oracle_per_seed=[0.11, 0.12, 0.13],
        signal_ci=(0.05, 0.15),
        null_level=constants.null_level,
        median_floor=constants.eligibility_median_floor,
    )
    return (
        verdict.status,
        round(r_priv_for_condition(0.07, 0.12, constants.null_level, verdict), 12),
    )


def _obs_eligibility(root: Path) -> Any:
    constants = analysis_constants(root)
    return tuple(
        privacy_eligibility(
            oracle_per_seed=[value, value, value],
            signal_ci=(0.005, 0.2),
            null_level=constants.null_level,
            median_floor=constants.eligibility_median_floor,
        ).status
        for value in (0.03, 0.06, 0.09, 0.15)
    )


def _obs_bootstrap(root: Path) -> Any:
    constants = analysis_constants(root)
    units = [PairedUnit(key=f"u{i}", values=(float(i % 7), 1.0)) for i in range(40)]
    result = bootstrap(
        paired_difference(0, 1),
        units,
        n_replicates=constants.bootstrap_replicates,
        seed=99,
        alpha=constants.bootstrap_alpha,
    )
    return (result.n_replicates, round(result.low, 12), round(result.high, 12))


def _obs_denominator(root: Path) -> Any:
    constants = analysis_constants(root)
    return tuple(
        denominator_stability(
            signal=0.1,
            standard_error=error,
            unstable_at=constants.denominator_unstable_above,
            unusable_above=constants.denominator_unusable_above,
        )
        for error in (0.005, 0.018, 0.023, 0.028, 0.033, 0.045, 0.07)
    )


def _obs_functional(root: Path) -> Any:
    constants = analysis_constants(root)
    return tuple(
        functional_eligibility(
            loss_base=2.0,
            loss_oracle=oracle,
            signal_ci=(0.001, 0.5),
            relative_floor=constants.functional_relative_floor,
            max_relative_se=constants.functional_max_relative_se,
            standard_error=error,
        ).status
        for oracle, error in (
            (1.99, 0.0005),
            (1.975, 0.0012),
            (1.945, 0.004),
            (1.93, 0.008),
            (1.9, 0.011),
            (1.9, 0.018),
            (1.8, 0.03),
        )
    )


def _obs_e50(root: Path) -> Any:
    constants = analysis_constants(root)
    errors = np.linspace(0.1, 2.0, 9)
    curve = fit_monotone_curve(errors, 1.0 - 0.35 * errors)
    short = fit_monotone_curve(errors[:3], (1.0 - 0.35 * errors)[:3])
    full = crossing_point(curve, constants.e50_level, min_support=constants.e50_min_support)
    limited = crossing_point(short, constants.e50_level, min_support=constants.e50_min_support)
    return (
        full.state,
        None if full.value is None else round(full.value, 12),
        limited.state,
    )


def _obs_null_trial(root: Path) -> Any:
    design = trial_design(root)
    rows = generate_null_trial(design, 820000)
    return tuple(
        (
            r.cell,
            r.seed,
            r.operator,
            round(r.error, 12),
            round(r.recovery, 12),
            round(r.d_l, 12),
            round(r.d_r, 12),
        )
        for r in rows
    )


def _obs_familywise(root: Path) -> Any:
    settings = dry_run_settings(root)
    design = trial_design(root)
    start = settings.integer("familywise_calibration_seed_start")
    count = settings.integer("familywise_calibration_trials")
    calibration = calibrate_familywise(
        design, list(range(start, start + count)), quantile=settings.number("familywise_quantile")
    )
    return (
        calibration.n_trials,
        calibration.seeds[0],
        calibration.seeds[-1],
        round(calibration.c_fwer, 12),
    )


def _obs_validation(root: Path) -> Any:
    settings = dry_run_settings(root)
    design = trial_design(root)
    start = settings.integer("familywise_calibration_seed_start")
    calibration = calibrate_familywise(
        design, list(range(start, start + 40)), quantile=settings.number("familywise_quantile")
    )
    verdict = evaluate_dry_g(
        design,
        calibration,
        settings.integers("validation_seeds"),
        n_replicates=20,
        alpha=settings.number("cell_cluster_alpha"),
    )
    return tuple((t.seed, t.passed, round(t.t_max, 9)) for t in verdict.trials)


def _obs_clusters(root: Path) -> Any:
    settings = dry_run_settings(root)
    rows = generate_null_trial(trial_design(root), 820000)
    intervals = cell_cluster_intervals(
        rows,
        n_replicates=settings.integer("cell_cluster_bootstrap_replicates"),
        seed=7,
        alpha=settings.number("cell_cluster_alpha"),
    )
    return tuple((k, round(v[0], 12), round(v[1], 12)) for k, v in sorted(intervals.items()))


def _obs_legacy(root: Path) -> Any:
    settings = dry_run_settings(root)
    start = settings.integer("legacy_marginal_seed_start")
    count = settings.integer("legacy_marginal_trials")
    values = marginal_diagnostic_tolerances(
        trial_design(root),
        list(range(start, start + count)),
        quantile=settings.number("legacy_marginal_quantile"),
    )
    return tuple(sorted((k, round(v, 12)) for k, v in values.items()))


def _obs_dry_d_thresholds(root: Path) -> Any:
    """The four DRY-D cutoffs, as `evaluate_dry_d1` / `evaluate_dry_d2` actually read them."""
    from src.analysis.dryrun import TailTrial, evaluate_dry_d1, evaluate_dry_d2, tail_thresholds

    limits = tail_thresholds(root)
    #: a hand-built trial, so the verdicts move with the cutoffs and nothing else
    trial = TailTrial(
        retained_ranks=(31, 1),
        e_floor=(0.01, 0.9),
        solver_error=(0.2, 0.0),
        reference_delta=(-0.20, -0.20),
        mink_delta=(-0.20, -0.20),
    )
    return (
        round(limits.d1_reference, 12),
        round(limits.d1_mink, 12),
        round(limits.d2_reference, 12),
        round(limits.d2_mink_abs, 12),
        evaluate_dry_d1(trial, thresholds=limits).status,
        evaluate_dry_d2(trial, thresholds=limits).status,
    )


def _obs_dry_d1_calibration(root: Path) -> Any:
    """The DRY-D1 positive-control bank, over a short window of the frozen seed family.

    The full 200-trial bank is generated once, pre-data, by
    `scripts/calibrate_dryrun_tolerances.py`. A consumption probe only has to prove the
    parameters reach the generator, so it runs the same canonical statistic over two seeds.
    """
    from src.analysis.dryrun import calibrate_dry_d1_positive_control

    settings = dry_run_settings(root)
    start = settings.integer("dry_d1_calibration_seed_start")
    count = settings.integer("dry_d1_calibration_trials")
    seeds = list(range(start, start + min(count, 2)))
    bank = calibrate_dry_d1_positive_control(
        seeds,
        planted_shift=settings.number("dry_d1_planted_shift"),
        quantile=settings.number("dry_d1_quantile"),
    )
    return (
        count,
        seeds[0],
        seeds[-1],
        round(bank["reference"].threshold, 12),
        round(bank["mink"].threshold, 12),
    )


def _obs_slope(root: Path) -> Any:
    settings = dry_run_settings(root)
    design = trial_design(root)
    slope = settings.number("dry_a_slope")
    rows = generate_monotone_trial(design, 1, slope=slope, operator_effect={})
    ladder = generate_calibration_ladder(design, 2, slope=slope)
    return (
        tuple(round(r.recovery, 12) for r in rows),
        tuple(round(r.recovery, 12) for r in ladder),
    )


def _obs_planted_effect(root: Path) -> Any:
    settings = dry_run_settings(root)
    design = trial_design(root)
    slope = settings.number("dry_a_slope")
    effect = settings.number("dry_b_operator_effect")
    curves = baseline_curves(generate_calibration_ladder(design, 11, slope=slope))
    observations = generate_operator_observations(
        design, 12, slope=slope, operator_effect={design.operators[0]: effect}
    )
    medians = residuals_against_baseline(observations, curves)
    return tuple(sorted((k, round(v, 12)) for k, v in medians.items()))


def _obs_positive_control(root: Path) -> Any:
    settings = dry_run_settings(root)
    design = trial_design(root)
    start = settings.integer("positive_control_seed_start")
    count = settings.integer("positive_control_trials")
    control = calibrate_operator_positive_control(
        design,
        list(range(start, start + count)),
        slope=settings.number("dry_a_slope"),
        planted_effect=settings.number("dry_b_operator_effect"),
        quantile=settings.number("positive_control_quantile"),
    )
    return (len(control.values), round(control.threshold, 12), round(control.values[0], 12))


def _obs_holdout(root: Path) -> Any:
    settings = dry_run_settings(root)
    design = trial_design(root)
    start = settings.integer("positive_control_holdout_seed_start")
    count = settings.integer("positive_control_holdout_trials")
    return tuple(
        round(
            operator_detection_fraction(
                design,
                list(range(start, start + count)),
                slope=settings.number("dry_a_slope"),
                planted_effect=settings.number("dry_b_operator_effect"),
                threshold=threshold,
            ),
            12,
        )
        for threshold in (0.05, 0.20, 0.24, 0.245, 0.25, 0.26, 0.30)
    )


def _obs_h1(root: Path) -> Any:
    settings = dry_run_settings(root)
    start = settings.integer("h1_calibration_seed_start")
    count = settings.integer("h1_calibration_trials")
    control = calibrate_h1_positive_control(
        master_seeds=list(range(start, start + count)),
        n_member=settings.integer("h1_records_per_class"),
        n_non_member=settings.integer("h1_records_per_class"),
        separation=settings.number("h1_separation"),
        target=settings.number("h4_target_fpr"),
        quantile=settings.number("h1_quantile"),
    )
    return (len(control.values), round(control.threshold, 12), round(control.values[0], 12))


def _obs_h4(root: Path) -> Any:
    settings = dry_run_settings(root)
    rows = calibrate_fixed_point_estimator(
        sample_sizes=settings.integers("h4_sample_sizes"),
        n_simulations=settings.integer("h4_simulations"),
        separation=settings.number("h4_separation"),
        target=settings.number("h4_target_fpr"),
        seed_start=settings.integer("h4_seed_start"),
    )
    return tuple(
        (r.n_non_member, r.n_simulations, round(r.analytic, 12), round(r.signed_bias, 10))
        for r in rows
    )


# ------------------------------------------------------------------ the matrix
FIELDS: tuple[Field, ...] = (
    Field(
        "target_fpr",
        SCORING_CONFIG,
        "run_outer_crossfit -> tpr_at_fixed_fpr",
        _scale(3.7),
        _obs_endpoint,
    ),
    Field(
        "n_outer_folds",
        SCORING_CONFIG,
        "plan_folds -> run_outer_crossfit",
        lambda v: 4,
        _obs_endpoint,
    ),
    Field("fold_assignment_seed", SCORING_CONFIG, "plan_folds", _offset(7919), _obs_endpoint),
    Field(
        "min_k_fraction",
        SCORING_CONFIG,
        "score_records -> min_k_percent_score",
        lambda v: 0.65,
        _obs_min_k,
    ),
    Field(
        "precision",
        SCORING_CONFIG,
        "ScoringContext snapshot -> CacheIdentity",
        lambda v: "float16",
        _obs_min_k,
    ),
    Field(
        "max_sequence_length",
        SCORING_CONFIG,
        "ScoringContext snapshot -> CacheIdentity",
        _offset(37),
        _obs_min_k,
    ),
    Field(
        "ridge_regularisation",
        SCORING_CONFIG,
        "select_pooled_method -> fit_ridge_pool",
        _scale(0.013),
        _obs_ridge,
    ),
    Field(
        "null_level",
        ANALYSIS_CONFIG,
        "r_priv_for_condition / oracle_signal",
        _scale(2.3),
        _obs_r_priv,
    ),
    Field(
        "eligibility_median_floor",
        ANALYSIS_CONFIG,
        "privacy_eligibility",
        lambda v: 0.077,
        _obs_eligibility,
    ),
    Field("bootstrap_replicates", ANALYSIS_CONFIG, "bootstrap", lambda v: 733, _obs_bootstrap),
    Field(
        "bootstrap_alpha",
        ANALYSIS_CONFIG,
        "bootstrap percentile interval",
        lambda v: 0.23,
        _obs_bootstrap,
    ),
    Field(
        "denominator_unstable_above",
        ANALYSIS_CONFIG,
        "denominator_stability",
        lambda v: 0.17,
        _obs_denominator,
    ),
    Field(
        "denominator_unusable_above",
        ANALYSIS_CONFIG,
        "denominator_stability",
        lambda v: 0.41,
        _obs_denominator,
    ),
    Field(
        "functional_relative_floor",
        ANALYSIS_CONFIG,
        "functional_eligibility",
        lambda v: 0.043,
        _obs_functional,
    ),
    Field(
        "functional_max_relative_se",
        ANALYSIS_CONFIG,
        "functional_eligibility",
        lambda v: 0.07,
        _obs_functional,
    ),
    Field("e50_level", ANALYSIS_CONFIG, "crossing_point", lambda v: 0.63, _obs_e50),
    Field("e50_min_support", ANALYSIS_CONFIG, "crossing_point", lambda v: 3, _obs_e50),
    Field(
        "n_registry_cells",
        DRY_RUN_CONFIG,
        "trial_design -> generate_null_trial",
        lambda v: 17,
        _obs_null_trial,
    ),
    Field(
        "training_seeds",
        DRY_RUN_CONFIG,
        "trial_design -> generate_null_trial",
        lambda v: [77, 88],
        _obs_null_trial,
    ),
    Field(
        "operators",
        DRY_RUN_CONFIG,
        "trial_design -> generate_null_trial",
        lambda v: ["Q1", "Q2"],
        _obs_null_trial,
    ),
    Field("error_low", DRY_RUN_CONFIG, "generate_null_trial", lambda v: 0.113, _obs_null_trial),
    Field("error_high", DRY_RUN_CONFIG, "generate_null_trial", lambda v: 1.717, _obs_null_trial),
    Field("recovery_base", DRY_RUN_CONFIG, "generate_null_trial", lambda v: 0.317, _obs_null_trial),
    Field(
        "cell_sd", DRY_RUN_CONFIG, "generate_null_trial cell noise", _scale(3.7), _obs_null_trial
    ),
    Field("row_sd", DRY_RUN_CONFIG, "generate_null_trial row noise", _scale(4.1), _obs_null_trial),
    Field(
        "gap_cell_sd",
        DRY_RUN_CONFIG,
        "generate_null_trial D_L/D_R cell noise",
        _scale(5.3),
        _obs_null_trial,
    ),
    Field(
        "gap_row_sd",
        DRY_RUN_CONFIG,
        "generate_null_trial D_L/D_R row noise",
        _scale(6.1),
        _obs_null_trial,
    ),
    Field(
        "familywise_calibration_seed_start",
        DRY_RUN_CONFIG,
        "calibrate_familywise",
        _offset(31337),
        _obs_familywise,
    ),
    Field(
        "familywise_calibration_trials",
        DRY_RUN_CONFIG,
        "calibrate_familywise",
        lambda v: 43,
        _obs_familywise,
    ),
    Field(
        "familywise_quantile",
        DRY_RUN_CONFIG,
        "calibrate_familywise c_FWER",
        lambda v: 0.71,
        _obs_familywise,
    ),
    Field(
        "validation_seeds",
        DRY_RUN_CONFIG,
        "evaluate_dry_g",
        lambda v: [810207, 810208, 810209],
        _obs_validation,
    ),
    Field(
        "cell_cluster_bootstrap_replicates",
        DRY_RUN_CONFIG,
        "cell_cluster_intervals",
        lambda v: 61,
        _obs_clusters,
    ),
    Field(
        "cell_cluster_alpha",
        DRY_RUN_CONFIG,
        "cell_cluster_intervals",
        lambda v: 0.19,
        _obs_clusters,
    ),
    Field(
        "legacy_marginal_seed_start",
        DRY_RUN_CONFIG,
        "marginal_diagnostic_tolerances",
        _offset(4441),
        _obs_legacy,
    ),
    Field(
        "legacy_marginal_trials",
        DRY_RUN_CONFIG,
        "marginal_diagnostic_tolerances",
        lambda v: 29,
        _obs_legacy,
    ),
    Field(
        "legacy_marginal_quantile",
        DRY_RUN_CONFIG,
        "marginal_diagnostic_tolerances",
        lambda v: 0.83,
        _obs_legacy,
    ),
    Field(
        "dry_d1_reference_threshold",
        DRY_RUN_CONFIG,
        "evaluate_dry_d1 via tail_thresholds",
        lambda v: -0.9,
        _obs_dry_d_thresholds,
    ),
    Field(
        "dry_d1_mink_threshold",
        DRY_RUN_CONFIG,
        "evaluate_dry_d1 via tail_thresholds",
        lambda v: -0.9,
        _obs_dry_d_thresholds,
    ),
    Field(
        "dry_d2_reference_threshold",
        DRY_RUN_CONFIG,
        "evaluate_dry_d2 via tail_thresholds",
        lambda v: -0.5,
        _obs_dry_d_thresholds,
    ),
    Field(
        "dry_d2_mink_abs_threshold",
        DRY_RUN_CONFIG,
        "evaluate_dry_d2 via tail_thresholds",
        lambda v: 0.001,
        _obs_dry_d_thresholds,
    ),
    Field(
        "dry_d1_planted_shift",
        DRY_RUN_CONFIG,
        "calibrate_dry_d1_positive_control -> generate_tail_trial",
        lambda v: -0.55,
        _obs_dry_d1_calibration,
    ),
    Field(
        "dry_d1_calibration_seed_start",
        DRY_RUN_CONFIG,
        "calibrate_dry_d1_positive_control seed family",
        _offset(31),
        _obs_dry_d1_calibration,
    ),
    Field(
        "dry_d1_calibration_trials",
        DRY_RUN_CONFIG,
        "calibrate_dry_d1_positive_control loop",
        lambda v: 1,
        _obs_dry_d1_calibration,
    ),
    Field(
        "dry_d1_quantile",
        DRY_RUN_CONFIG,
        "calibrate_dry_d1_positive_control percentile",
        lambda v: 12.5,
        _obs_dry_d1_calibration,
    ),
    Field(
        "dry_a_slope",
        DRY_RUN_CONFIG,
        "generate_monotone_trial / calibration ladder",
        lambda v: 0.267,
        _obs_slope,
    ),
    Field(
        "dry_b_operator_effect",
        DRY_RUN_CONFIG,
        "generate_operator_observations",
        lambda v: 0.417,
        _obs_planted_effect,
    ),
    Field(
        "positive_control_seed_start",
        DRY_RUN_CONFIG,
        "calibrate_operator_positive_control",
        _offset(6151),
        _obs_positive_control,
    ),
    Field(
        "positive_control_trials",
        DRY_RUN_CONFIG,
        "calibrate_operator_positive_control loop",
        lambda v: 13,
        _obs_positive_control,
    ),
    Field(
        "positive_control_quantile",
        DRY_RUN_CONFIG,
        "calibrate_operator_positive_control",
        lambda v: 11.5,
        _obs_positive_control,
    ),
    Field(
        "positive_control_holdout_seed_start",
        DRY_RUN_CONFIG,
        "operator_detection_fraction",
        _offset(7717),
        _obs_holdout,
    ),
    Field(
        "positive_control_holdout_trials",
        DRY_RUN_CONFIG,
        "operator_detection_fraction loop",
        lambda v: 9,
        _obs_holdout,
    ),
    Field(
        "h1_separation", DRY_RUN_CONFIG, "calibrate_h1_positive_control", lambda v: 1.37, _obs_h1
    ),
    Field(
        "h1_records_per_class",
        DRY_RUN_CONFIG,
        "calibrate_h1_positive_control",
        lambda v: 311,
        _obs_h1,
    ),
    Field(
        "h1_calibration_seed_start",
        DRY_RUN_CONFIG,
        "calibrate_h1_positive_control",
        _offset(8123),
        _obs_h1,
    ),
    Field(
        "h1_calibration_trials",
        DRY_RUN_CONFIG,
        "calibrate_h1_positive_control loop",
        lambda v: 7,
        _obs_h1,
    ),
    Field("h1_quantile", DRY_RUN_CONFIG, "calibrate_h1_positive_control", lambda v: 61.5, _obs_h1),
    Field(
        "h4_target_fpr", DRY_RUN_CONFIG, "calibrate_fixed_point_estimator", lambda v: 0.037, _obs_h4
    ),
    Field(
        "h4_simulations",
        DRY_RUN_CONFIG,
        "calibrate_fixed_point_estimator loop",
        lambda v: 11,
        _obs_h4,
    ),
    Field(
        "h4_sample_sizes",
        DRY_RUN_CONFIG,
        "calibrate_fixed_point_estimator sweep",
        lambda v: [131, 257],
        _obs_h4,
    ),
    Field(
        "h4_seed_start", DRY_RUN_CONFIG, "calibrate_fixed_point_estimator", _offset(9133), _obs_h4
    ),
    Field(
        "h4_separation",
        DRY_RUN_CONFIG,
        "calibrate_fixed_point_estimator analytic truth",
        lambda v: 1.23,
        _obs_h4,
    ),
)

#: Expensive fields are observed with a reduced but still config-driven workload by first
#: shrinking the loop counts that are NOT under test. The field under test always keeps its
#: real value on the pristine side.
SHRINK = {
    "familywise_calibration_trials": 60,
    "positive_control_trials": 20,
    "positive_control_holdout_trials": 20,
    "h1_calibration_trials": 20,
    "h1_records_per_class": 256,
    "h4_simulations": 25,
    "h4_sample_sizes": [200, 400],
    "cell_cluster_bootstrap_replicates": 60,
    "legacy_marginal_trials": 40,
}


@pytest.fixture(scope="module")
def repo_configs(repo_root: Path) -> Path:
    return repo_root


def _prepare(repo_root: Path, destination: Path, field: Field) -> Path:
    shutil.copytree(repo_root / "configs", destination / "configs")
    path = destination / "configs" / DRY_RUN_CONFIG
    document = json.loads(path.read_text(encoding="utf-8"))
    for key, value in SHRINK.items():
        if key != field.name and key in document:
            document[key] = value
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return destination


def _mutate(root: Path, field: Field) -> Any:
    path = root / "configs" / field.config
    document = json.loads(path.read_text(encoding="utf-8"))
    original = document[field.name]
    replacement = field.mutate(original)
    assert replacement != original, f"{field.name}: the mutation did not change the value"
    document[field.name] = replacement
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return replacement


@pytest.mark.parametrize("field", FIELDS, ids=lambda f: f"{f.config}:{f.name}")
def test_the_field_controls_its_execution_path(
    repo_root: Path, tmp_path: Path, field: Field
) -> None:
    """Mutate the field, run the owning path, and require a moved observation or a refusal."""
    root = _prepare(repo_root, tmp_path / "case", field)
    before = field.observe(root)
    replacement = _mutate(root, field)
    try:
        after = field.observe(root)
    except Exception as exc:  # noqa: BLE001 — an explicit refusal is an accepted outcome
        assert str(exc), f"{field.name}: rejected with an empty message"
        return
    assert after != before, (
        f"INERT: {field.name} in {field.config} was mutated to {replacement!r} and"
        f" {field.consumer} produced an identical observation"
    )


def test_no_active_config_field_is_inert(repo_root: Path, tmp_path: Path) -> None:
    """The whole matrix at once, reported as the required table and asserted empty."""
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    inert: list[str] = []
    for index, field in enumerate(FIELDS):
        root = _prepare(repo_root, tmp_path / f"sweep{index}", field)
        before = field.observe(root)
        replacement = _mutate(root, field)
        try:
            after = field.observe(root)
        except Exception as exc:  # noqa: BLE001
            effect, status = f"REJECTED: {type(exc).__name__}", "CONSUMED"
        else:
            moved = after != before
            effect = "OBSERVATION_CHANGED" if moved else "NO_EFFECT"
            status = "CONSUMED" if moved else "INERT"
            if not moved:
                inert.append(field.name)
        rows.append(
            (
                field.name,
                field.config,
                "ACTIVE_MATERIAL",
                field.consumer,
                repr(replacement)[:28],
                effect,
                status,
            )
        )

    header = (
        "FIELD",
        "CONFIG",
        "CLASS",
        "EXECUTION_CONSUMER",
        "MUTATION",
        "OBSERVED_EFFECT",
        "STATUS",
    )
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    print("\n" + "  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)))
    print(f"\nACTIVE_CONFIG_FIELD_COUNT = {len(rows)}")
    print(f"CONSUMED_ACTIVE_CONFIG_FIELD_COUNT = {sum(1 for r in rows if r[6] == 'CONSUMED')}")
    print(f"INERT_ACTIVE_CONFIG_FIELDS = {inert}")

    assert inert == [], f"INERT_ACTIVE_CONFIG_FIELDS must be empty, found {inert}"
    assert all(row[6] == "CONSUMED" for row in rows)


def test_the_matrix_covers_every_non_metadata_field(repo_root: Path) -> None:
    """No field may exist in a config without a row in the matrix above."""
    from src.analysis.settings import load_settings

    covered = {(field.config, field.name) for field in FIELDS}
    for relative in (SCORING_CONFIG, ANALYSIS_CONFIG, DRY_RUN_CONFIG):
        for key in load_settings(repo_root, relative).document:
            if key in METADATA_KEYS:
                continue
            assert (relative, key) in covered, f"{relative}:{key} has no consumption row"
    assert len(covered) == len(FIELDS)


def test_obsolete_keys_are_absent(repo_root: Path) -> None:
    from src.analysis.settings import load_settings

    for relative in (SCORING_CONFIG, ANALYSIS_CONFIG, DRY_RUN_CONFIG):
        document = load_settings(repo_root, relative).document
        for key in OBSOLETE_REMOVED:
            assert key not in document, f"{relative} still carries {key!r}"


def test_the_config_identity_moves_with_the_body(repo_root: Path, tmp_path: Path) -> None:
    from src.analysis.settings import load_settings

    root = _prepare(repo_root, tmp_path / "identity", FIELDS[0])
    before = load_settings(root, SCORING_CONFIG)
    assert before.sha256 == sha256_canonical(before.document)
    _mutate(root, FIELDS[0])
    after = load_settings(root, SCORING_CONFIG)
    assert after.sha256 != before.sha256


def test_the_cache_invalidates_after_a_material_scoring_change(
    repo_root: Path, tmp_path: Path
) -> None:
    root = _prepare(repo_root, tmp_path / "cachecase", FIELDS[3])
    key_before, _, _, scores_before = _obs_min_k(root)
    _mutate(root, FIELDS[3])
    key_after, _, _, scores_after = _obs_min_k(root)
    assert key_after != key_before, "a material scoring change did not move the cache key"
    assert scores_after != scores_before, "a material scoring change did not move the scores"
