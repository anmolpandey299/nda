"""The truth side of the firewall [AUTH: 00 §22, §24A.1, §25; 01 §10].

Everything here needs the ground-truth protected constituent, and nothing here is importable
from a solver's execution path: `src.recovery.solvers` does not import this module, so no
recovery hyperparameter, stopping rule or output can depend on an evaluation error.

Metrics are computed on induced updates ΔW = BA. Raw LoRA factors are never compared
[AUTH: 00 §22]. Scalars accumulate in the diagnostic dtype (float64) because a derived
statistic is not artifact bytes [AUTH: 01 §10], while the artifacts themselves stay float32.

A zero target is an explicit undefined state, never a silent Inf, and e_F is never clamped.

**Truth is bound, not passed.** Every metric here is a statement about a *specific* protected
constituent, so the evaluator takes an `EvaluationTruthBinding` rather than a loose task
vector. The binding factory has already proved that the truth it holds is what the evaluated
recovery's observation was actually built from, so scoring an A_1 recovery against an
unrelated A_2 has no API through which to happen [AUTH: 00 §22, §24A].

**Thresholds are frozen, not parameters.** The C1 criterion and the partner norm-match limit
come from the resolved recovery context, never from a caller argument, so no evaluation call
can move a scientific pass/fail decision [AUTH: 00 §25, §34B.1A; 01 §17].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from src.merge.family import ReleaseFamily
from src.merge.updates import Matrix, TaskVector
from src.provenance.hashing import JSONValue
from src.recovery.context import RecoveryExecutionContext
from src.recovery.solvers import (
    TRUNCATED_PROTECTED_SOURCE,
    RecoveryResult,
    truncate_to_rank,
)
from src.recovery.truth import EvaluationTruthBinding, TruthBindingError, o3_source_for

EVALUATION_VERSION: Final = "s08.evaluation.v1"

#: The state a ratio reports when its denominator is zero.
UNDEFINED: Final = "UNDEFINED_ZERO_DENOMINATOR"
PARTNER_NORM_MATCH_FAILED: Final = "PARTNER_NORM_MATCH_FAILED"


class EvaluationError(ValueError):
    """A recovery metric is not defined for the inputs given."""


def _bound_truth(result: RecoveryResult, binding: EvaluationTruthBinding) -> TaskVector:
    """The one truth this result may be scored against [AUTH: 00 §22].

    The binding already proved which observation its truth belongs to; this is the other half
    — that the result being evaluated ran on that same observation.
    """
    if not isinstance(binding, EvaluationTruthBinding):
        raise EvaluationError(
            "scientific evaluation takes an EvaluationTruthBinding, not"
            f" {type(binding).__name__}; a loose task vector cannot be attributed to this"
            " recovery [AUTH: 00 §22, §24A]"
        )
    if result.observation_identity != binding.observation_identity:
        raise TruthBindingError(
            "this truth binding was issued for a different observation; the recovery being"
            " evaluated did not run on it [AUTH: 00 §22]"
        )
    return binding.protected_truth


def _aligned(result: RecoveryResult, truth: TaskVector) -> tuple[list[str], np.dtype[Any]]:
    if result.surface_identity() != _surface_of(truth, result):
        raise EvaluationError(
            "the recovered update and the ground truth cover different parameter surfaces"
        )
    return list(result.names), result.context.diagnostic


def _surface_of(truth: TaskVector, result: RecoveryResult) -> str:
    """The truth's surface identity in the RESULT's own encoding, so the two are comparable."""
    import hashlib

    from src.recovery.solvers import RESULT_SCHEMA

    digest = hashlib.sha256(f"{RESULT_SCHEMA}|surface".encode())
    for name in truth.names:
        rows, columns = truth.shapes[name]
        digest.update(f"|{name}:{rows}x{columns}:{result.dtype}".encode())
    return digest.hexdigest()


def _frobenius(values: Sequence[Matrix], wide: np.dtype[Any]) -> float:
    return float(np.sqrt(sum(float(np.sum(np.asarray(v, dtype=wide) ** 2)) for v in values)))


def relative_frobenius(
    candidate: Mapping[str, Matrix], target: Mapping[str, Matrix], *, wide: np.dtype[Any]
) -> float:
    """||candidate - target||_F / ||target||_F over one shared surface."""
    names = sorted(target)
    denominator = _frobenius([target[name] for name in names], wide)
    if denominator == 0.0:
        raise EvaluationError(
            f"{UNDEFINED}: the target update is exactly zero, so a relative Frobenius error"
            " is undefined [AUTH: 00 §22]"
        )
    numerator = _frobenius(
        [
            np.asarray(candidate[name], dtype=wide) - np.asarray(target[name], dtype=wide)
            for name in names
        ],
        wide,
    )
    return numerator / denominator


# ----------------------------------------------------------------------------------------
# 00 §22 parameter-recovery metrics
# ----------------------------------------------------------------------------------------


def parameter_recovery_metrics(
    result: RecoveryResult, binding: EvaluationTruthBinding
) -> dict[str, JSONValue]:
    """The 00 §22 primary and secondary parameter metrics.

    Primary: global relative Frobenius e_F, and the per-layer relative error.
    Secondary: flattened cosine similarity, layerwise error, spectral error, maximum layer
    error, and error concentration across layers.

    e_F is not clamped, and the result is not mutated. The truth comes from a binding proved
    against the observation this result ran on, so the numbers are attributable.
    """
    protected_truth = _bound_truth(result, binding)
    names, wide = _aligned(result, protected_truth)
    candidate = {name: result[name] for name in names}
    target = {name: protected_truth[name] for name in names}

    e_f = relative_frobenius(candidate, target, wide=wide)

    per_layer: dict[str, JSONValue] = {}
    for name in names:
        denominator = _frobenius([target[name]], wide)
        per_layer[name] = (
            UNDEFINED
            if denominator == 0.0
            else _frobenius(
                [np.asarray(candidate[name], dtype=wide) - np.asarray(target[name], dtype=wide)],
                wide,
            )
            / denominator
        )
    numeric_layers = [v for v in per_layer.values() if isinstance(v, float)]

    flat_candidate = np.concatenate(
        [np.asarray(candidate[name], dtype=wide).ravel() for name in names]
    )
    flat_target = np.concatenate([np.asarray(target[name], dtype=wide).ravel() for name in names])
    norms = float(np.linalg.norm(flat_candidate)) * float(np.linalg.norm(flat_target))
    cosine: JSONValue = (
        UNDEFINED if norms == 0.0 else float(np.dot(flat_candidate, flat_target) / norms)
    )

    spectral: dict[str, JSONValue] = {}
    for name in names:
        left = np.linalg.svd(np.asarray(candidate[name], dtype=wide), compute_uv=False)
        right = np.linalg.svd(np.asarray(target[name], dtype=wide), compute_uv=False)
        spectral[name] = float(np.linalg.norm(left - right))

    total_error = float(sum(v * v for v in numeric_layers))
    concentration: JSONValue = (
        UNDEFINED if total_error == 0.0 else float(max(v * v for v in numeric_layers) / total_error)
    )

    return {
        "evaluation_version": EVALUATION_VERSION,
        "method": result.method,
        "target_class": result.target_class,
        "recovery_result_sha256": result.identity(),
        "truth_binding_sha256": binding.identity(),
        "protected_update_sha256": binding.protected_identity,
        "e_f": e_f,
        "per_layer_relative_error": per_layer,
        "cosine_similarity": cosine,
        "spectral_error": spectral,
        "max_layer_error": max(numeric_layers) if numeric_layers else UNDEFINED,
        "error_concentration": concentration,
    }


def c1_pass(result: RecoveryResult, binding: EvaluationTruthBinding) -> bool:
    """00 §25 P0-C1: e_F <= the frozen criterion, on float32 artifacts.

    The limit is read from the recovery context the result actually ran under. There is no
    caller parameter, so no evaluation call can move a scientific pass/fail decision
    [AUTH: 00 §25, §34B.1A; 01 §17]. The solver never sees this function's output.
    """
    limit = result.context.c1_pass_criterion_ef
    metrics = parameter_recovery_metrics(result, binding)
    value = metrics["e_f"]
    if not isinstance(value, float):  # pragma: no cover - relative_frobenius raised already
        raise EvaluationError("e_F is undefined for this pair")
    return value <= limit


# ----------------------------------------------------------------------------------------
# 00 §24A O3 error decomposition
# ----------------------------------------------------------------------------------------


def o3_error_report(
    result: RecoveryResult,
    binding: EvaluationTruthBinding,
    *,
    o3_source: ReleaseFamily | None = None,
) -> dict[str, JSONValue]:
    """The four separate O3 quantities [AUTH: 00 §24A.1, §24A.5].

    1. `e_f`                   ||A_hat - A||_F / ||A||_F        absolute original-constituent
    2. `e_floor`               the analytic operator floor      via the frozen S07 implementation
    3. `e_solver_origscale`    ||A_hat - T_s(A)||_F / ||A||_F
    4. `e_solver_retained`     ||A_hat - T_s(A)||_F / ||T_s(A)||_F

    `e_F - e_floor` is deliberately NOT computed. The floor and the solver error are different
    quantities on different denominators, and their difference is not the solver's error
    [AUTH: 00 §24A.5].

    When `o3_source` is given it must be the very release family the binding was issued for,
    and the floor is taken from that family's descendant at the bound retained rank — never
    from an unrelated O3 result a caller happens to hold.
    """
    if result.target_class != TRUNCATED_PROTECTED_SOURCE:
        raise EvaluationError(
            f"the O3 error decomposition applies to a {TRUNCATED_PROTECTED_SOURCE} recovery;"
            f" this one targets {result.target_class}"
        )
    protected_truth = _bound_truth(result, binding)
    rank = result.retained_rank
    if rank is None:
        raise EvaluationError("an O3 recovery must carry its public retained rank")
    if binding.retained_rank != rank:
        raise TruthBindingError(
            f"this binding was issued at retained rank {binding.retained_rank}; the recovery"
            f" ran at {rank}. A floor is a property of one truncation [AUTH: 00 §24A.5]"
        )

    names, wide = _aligned(result, protected_truth)
    context = result.context
    candidate = {name: result[name] for name in names}
    original = {name: protected_truth[name] for name in names}
    truncated = {
        name: truncate_to_rank(protected_truth[name], rank=rank, context=context) for name in names
    }

    original_norm = _frobenius([original[name] for name in names], wide)
    truncated_norm = _frobenius([truncated[name] for name in names], wide)
    solver_gap = _frobenius(
        [
            np.asarray(candidate[name], dtype=wide) - np.asarray(truncated[name], dtype=wide)
            for name in names
        ],
        wide,
    )

    report: dict[str, JSONValue] = {
        "evaluation_version": EVALUATION_VERSION,
        "retained_rank": rank,
        "recovery_result_sha256": result.identity(),
        "truth_binding_sha256": binding.identity(),
        "e_f": UNDEFINED
        if original_norm == 0.0
        else relative_frobenius(candidate, original, wide=wide),
        "e_solver_origscale": UNDEFINED if original_norm == 0.0 else solver_gap / original_norm,
        "e_solver_retained": UNDEFINED if truncated_norm == 0.0 else solver_gap / truncated_norm,
    }
    if o3_source is not None:
        from src.merge.diagnostics import protected_information_floor_from_o3

        release = o3_source_for(binding, o3_source)
        report["e_floor"] = protected_information_floor_from_o3(release, protected_truth)
        report["e_floor_source"] = "s07.protected_information_floor_from_o3"
    else:
        report["e_floor"] = (
            UNDEFINED
            if original_norm == 0.0
            else _frobenius(
                [
                    np.asarray(truncated[name], dtype=wide) - np.asarray(original[name], dtype=wide)
                    for name in names
                ],
                wide,
            )
            / original_norm
        )
        report["e_floor_source"] = "s08.recomputed_truncation"
    return report


# ----------------------------------------------------------------------------------------
# 00 §25 P0-C2B stress diagnostics
# ----------------------------------------------------------------------------------------


def beta_spread(alphas: Sequence[float]) -> dict[str, JSONValue]:
    """rho_beta and a_alpha — parameter-only, so a solver may hold these [00 §25 P0-C2B.3]."""
    from src.recovery.solvers import beta_for

    betas = beta_for(alphas)
    magnitudes = [abs(b) for b in betas]
    if min(magnitudes) == 0.0:
        raise EvaluationError(f"{UNDEFINED}: a zero beta makes the analytic spread undefined")
    return {
        "beta": list(betas),
        "rho_beta": max(magnitudes) / min(magnitudes),
        "a_alpha": max(1.0 / abs(float(a)) for a in alphas),
    }


def residual_spread(
    alphas: Sequence[float], partner_updates: Sequence[TaskVector]
) -> dict[str, JSONValue]:
    """rho_resid — EVALUATOR ONLY [AUTH: 00 §25 P0-C2B.3].

    This needs the true partner updates, so it is never reachable from a C2B attack solver.
    It exists for experiment construction and reporting.
    """
    from src.recovery.solvers import beta_for

    betas = beta_for(alphas)
    if len(betas) != len(partner_updates):
        raise EvaluationError("one beta is needed per partner update")
    norms = [
        abs(beta) * partner.frobenius_norm()
        for beta, partner in zip(betas, partner_updates, strict=True)
    ]
    if min(norms) == 0.0:
        raise EvaluationError(f"{UNDEFINED}: a zero residual norm makes rho_resid undefined")
    return {"residual_norms": list(norms), "rho_resid": max(norms) / min(norms)}


def partner_norm_guard(
    partner_updates: Sequence[TaskVector], *, context: RecoveryExecutionContext
) -> dict[str, JSONValue]:
    """00 §25 P0-C2B.2: max||tau_B||_F / min||tau_B||_F <= the frozen limit.

    The limit is read from the resolved recovery context, not from a caller argument, so no
    call site can widen a scientific match criterion [AUTH: 00 §25, §34B.1A; 01 §17].

    EXPERIMENT-CONSTRUCTION AND EVALUATION ONLY. A recovery method may never use it to search
    or select partners; partners are chosen on parameter information before any privacy
    outcome is viewed.
    """
    limit = context.partner_norm_match_limit
    norms = [partner.frobenius_norm() for partner in partner_updates]
    if not norms:
        raise EvaluationError("the guard needs at least one partner")
    if min(norms) == 0.0:
        raise EvaluationError(f"{UNDEFINED}: a zero-norm partner makes the ratio undefined")
    ratio = max(norms) / min(norms)
    return {
        "partner_norms": list(norms),
        "norm_ratio": ratio,
        "limit": limit,
        "status": "PARTNER_NORM_MATCHED" if ratio <= limit else PARTNER_NORM_MATCH_FAILED,
    }
