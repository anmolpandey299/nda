"""Operator-side facts the later analysis needs [AUTH: 00 §24.1, §24A.1, §24A.6].

These are properties of a merge transformation, not privacy conclusions and not recovery
results. S07 exposes them; Block B owns their statistical interpretation and S08 owns
recovery. The one distortion metric here, e_DARE, belongs to the operator specification
itself (00 §24.1) rather than to any attacker.

Everything is defined on induced updates, so a zero denominator is an explicit undefined
state and never a silent Inf or NaN.

Two diagnostics are **bound**: they take built results, not loose arrays, and verify the
correspondence the quantity's definition assumes before computing anything.

* `dare_perturbation_from_results` requires that the DARE and LINEAR descendants share the
  same protected constituent, the same partner, the same α, the same surface and the same
  execution context. Comparing a DARE descendant to an unrelated linear merge produces a
  number, and that number is not e_DARE.
* `protected_information_floor` requires a `ProtectedConstituent` — an object that structurally
  carries the protected role together with its own truncation at s. e_floor is a property of
  ΔW_A [AUTH: 00 §24A.1]; handing it a partner's truncation would silently answer a different
  question.

The low-level `relative_frobenius_error` remains available, and is deliberately not named
e_DARE: it is a norm ratio with no correspondence proof.

Scalars accumulate in the context's diagnostic dtype (float64), because a derived statistic is
not artifact bytes [AUTH: 01 §10].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from src.merge.operators import (
    DARE,
    LINEAR,
    SVD_TRUNC,
    Truncation,
    truncation_set_identity,
    truncations,
)
from src.merge.updates import Matrix, TaskVector, require_compatible

DIAGNOSTIC_VERSION: Final = "s07.diagnostics.v2"

#: The constituent role e_floor is defined for [AUTH: 00 §24A.1]. Never caller-asserted:
#: `protected_information_floor_from_o3` establishes it from the issued O3 result's own
#: recorded protected identity.
PROTECTED_CONSTITUENT: Final = "PROTECTED_CONSTITUENT"

#: The state a ratio reports when its denominator is zero [AUTH: 00 §24.1].
UNDEFINED: Final = "UNDEFINED_ZERO_DENOMINATOR"


class DiagnosticError(ValueError):
    """A diagnostic is not defined for the inputs given."""


# ----------------------------------------------------------------------------------------
# distortion
# ----------------------------------------------------------------------------------------


def _relative_frobenius_error(candidate: TaskVector, reference: TaskVector) -> float:
    """||candidate - reference||_F / ||reference||_F over the shared surface.

    PRIVATE and low-level. It proves no correspondence between its two arguments, so its
    output may not be labelled e_DARE or e_floor; those come from the bound paths below.

    A zero reference raises rather than returning Inf: "infinitely wrong relative to nothing"
    is not a measurement, and silently propagating it would poison every downstream summary.
    Accumulated in the diagnostic dtype: a scalar, not artifact bytes [AUTH: 01 §10].
    """
    require_compatible([candidate, reference], what="relative error")
    wide = reference.context.diagnostic
    denominator = reference.frobenius_norm()
    if denominator == 0.0:
        raise DiagnosticError(
            f"{UNDEFINED}: the reference update is exactly zero, so a relative Frobenius"
            " error is undefined [AUTH: 00 §24.1]"
        )
    numerator = float(
        np.sqrt(
            sum(
                float(np.sum((candidate[name].astype(wide) - reference[name].astype(wide)) ** 2))
                for name in reference.names
            )
        )
    )
    return numerator / denominator


def _correspondence_problems(dare_result: Any, linear_result: Any) -> list[str]:
    """Every way two descendants fail to be the same comparison at two operators.

    The scientific pairing is checked through `comparison_identity`, which each issued result
    derives from its own protected identity, partner identity, alpha, surface and context —
    so a mismatch cannot be talked around by relabelling anything.
    """
    problems: list[str] = []
    if getattr(dare_result, "operator", None) != DARE:
        problems.append(
            f"the compared artifact is {getattr(dare_result, 'operator', None)!r}, not {DARE}"
        )
    if getattr(linear_result, "operator", None) != LINEAR:
        problems.append(
            f"the baseline is {getattr(linear_result, 'operator', None)!r}, not {LINEAR}"
        )
    left = getattr(dare_result, "comparison_identity", None)
    right = getattr(linear_result, "comparison_identity", None)
    if left is None or right is None:
        problems.append("a descendant is not an issued MergeResult")
        return problems
    if left() != right():
        for field_name, label in (
            ("protected_identity", "protected constituent"),
            ("partner_identity", "partner"),
            ("surface_identity", "parameter surface"),
            ("alpha", "merge coefficient"),
        ):
            if getattr(dare_result, field_name, None) != getattr(
                linear_result, field_name, object()
            ):
                problems.append(f"the two descendants use a different {label}")
        if getattr(dare_result, "execution_context_sha256", None) != getattr(
            linear_result, "execution_context_sha256", object()
        ):
            problems.append("the two descendants ran under different execution contexts")
        if not problems:  # pragma: no cover - identity differs for an unenumerated reason
            problems.append("the two descendants do not share a comparison identity")
    if (
        getattr(dare_result, "update", None) is not None
        and getattr(linear_result, "update", None) is not None
    ):
        if dare_result.update.dtype != linear_result.update.dtype:
            problems.append("the two descendants have different artifact dtypes")
    return problems


def dare_perturbation_from_results(dare_result: Any, baseline_linear_result: Any) -> float:
    """e_DARE,p = ||ΔW_Cp - ΔW_Cbaseline||_F / ||ΔW_Cbaseline||_F [AUTH: 00 §24.1].

    THE authoritative DARE distortion, and the only scientifically labelled one. The pair must
    be the same comparison at two operators, checked before any arithmetic runs. Comparing a
    DARE descendant to an unrelated linear merge produces a number, and that number is not
    e_DARE.

    A realised operator distortion, reported beside the theoretical reference — not the DARE
    utility gate, which is model NLL on real checkpoints and is NOT_RUN.
    """
    problems = _correspondence_problems(dare_result, baseline_linear_result)
    if problems:
        raise DiagnosticError(
            "these two descendants are not the same comparison, so their norm ratio is not"
            f" e_DARE [AUTH: 00 §24.1]: {'; '.join(problems)}"
        )
    return _relative_frobenius_error(dare_result.update, baseline_linear_result.update)


def dare_theoretical_perturbation(drop_probability: float) -> float:
    """sqrt(p / (1-p)) — the reference RMS error factor [AUTH: 00 §24.1].

    A reference for reporting, not a pass/fail requirement on any finite realisation: a
    fixture of a few thousand coordinates will scatter around it. At p = 0.25 it is
    sqrt(1/3) ≈ 0.577; at p = 0 it is 0; and it grows without bound as p approaches 1.
    """
    if not 0.0 <= drop_probability < 1.0:
        raise DiagnosticError(
            f"the theoretical DARE perturbation is defined on [0, 1); got {drop_probability!r}"
        )
    return float(np.sqrt(drop_probability / (1.0 - drop_probability)))


# ----------------------------------------------------------------------------------------
# spectra and the O3 information floor
# ----------------------------------------------------------------------------------------


def singular_values(matrix: Matrix) -> Matrix:
    if matrix.ndim != 2:
        raise DiagnosticError("a spectrum is defined on a 2-D induced update")
    return np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)


@dataclass(frozen=True)
class SpectralEnergy:
    """Discarded and total singular energy summed over every target matrix."""

    discarded: float
    total: float
    per_matrix: Mapping[str, tuple[float, float]]

    @property
    def defined(self) -> bool:
        return self.total > 0.0

    def floor(self) -> float:
        """e_floor(s) = sqrt(discarded / total) [AUTH: 00 §24A.1].

        The best achievable original-constituent error if only T_s(ΔW_A) is released and
        nothing external restores the tail. An operator-imposed information floor — S07
        computes it; it does not interpret it.
        """
        if not self.defined:
            raise DiagnosticError(
                f"{UNDEFINED}: the constituent has zero singular energy, so e_floor is"
                " undefined [AUTH: 00 §24A.1]"
            )
        return float(np.sqrt(self.discarded / self.total))


def spectral_energy(parts: Mapping[str, Truncation]) -> SpectralEnergy:
    """Sum the discarded and total energy across a surface, keeping the per-matrix split.

    A LOW-LEVEL helper: it does not know whose truncation it was given. `.floor()` on the
    result is only e_floor when the parts are the protected constituent's, which is what
    `protected_information_floor` establishes.
    """
    if not parts:
        raise DiagnosticError("no truncations to summarise")
    per_matrix = {
        name: (part.discarded_energy, part.total_energy) for name, part in sorted(parts.items())
    }
    return SpectralEnergy(
        discarded=float(sum(d for d, _ in per_matrix.values())),
        total=float(sum(t for _, t in per_matrix.values())),
        per_matrix=per_matrix,
    )


def protected_information_floor_from_o3(o3_result: Any, protected_update: TaskVector) -> float:
    """e_floor(s) for an issued O3 descendant and its own protected constituent [00 §24A.1].

    THE authoritative O3 information floor. The caller supplies no truncation and no rank:

    1. the result must be an O3 descendant;
    2. `protected_update` must hash to the result's recorded protected content identity;
    3. it must be the same parameter surface;
    4. it must be the same execution context;
    5. the retained rank is read off the issued result;
    6. T_s(A) is RECOMPUTED here under that same frozen context;
    7. the recomputed truncation identity must match the one the issued result stored;
    8. e_floor is computed from the protected spectrum alone.

    So a partner's truncation, an unrelated truncation, a mismatched rank and a wrong A are
    all unrepresentable rather than merely refused.

    It is the best achievable original-constituent error if only T_s(ΔW_A) is released and
    nothing external restores the discarded tail — an operator-imposed floor, not a solver
    error. S07 computes it and does not interpret it.
    """
    if getattr(o3_result, "operator", None) != SVD_TRUNC:
        raise DiagnosticError(
            f"e_floor is defined for an {SVD_TRUNC} descendant; got"
            f" {getattr(o3_result, 'operator', None)!r} [AUTH: 00 §24A.1]"
        )
    if protected_update.content_identity() != o3_result.protected_identity:
        raise DiagnosticError(
            "the supplied update is not the protected constituent this O3 descendant was"
            " built from; e_floor is a property of ΔW_A [AUTH: 00 §24A.1]"
        )
    if protected_update.surface_identity() != o3_result.surface_identity:
        raise DiagnosticError("the supplied update covers a different parameter surface")
    context = o3_result.context
    if protected_update.context.identity() != context.identity():
        raise DiagnosticError("the supplied update was resolved under a different context")

    rank = o3_result.retained_rank
    if rank is None:  # pragma: no cover - guaranteed by the O3 factory
        raise DiagnosticError("the O3 descendant records no retained rank")

    parts = truncations(protected_update, rank=rank, context=context)
    recomputed = truncation_set_identity(parts)
    if recomputed != o3_result.protected_truncation_identity:
        raise DiagnosticError(
            "the recomputed protected truncation does not match the one the O3 descendant"
            " was issued with; the artifact and the constituent disagree"
        )

    energy = spectral_energy(parts)
    return energy.floor()


def numerical_effective_rank(matrix: Matrix, *, tolerance: float) -> int:
    """Count of σ_j with σ_j/σ_1 >= tolerance [AUTH: 00 §24A.6].

    `tolerance` is resolved from config, never a source literal; 00 §24A.6 freezes it at
    1e-6. A zero matrix has effective rank 0, stated rather than divided by.
    """
    if not 0.0 < tolerance <= 1.0:
        raise DiagnosticError(f"effective-rank tolerance {tolerance!r} is not in (0, 1]")
    spectrum = singular_values(matrix)
    if spectrum.size == 0 or spectrum[0] == 0.0:
        return 0
    return int(np.count_nonzero(spectrum / spectrum[0] >= tolerance))


def stable_rank(matrix: Matrix) -> float:
    """||W||_F^2 / ||W||_2^2 [AUTH: 00 §24A.6]. Zero matrix -> 0.0, explicitly."""
    spectrum = singular_values(matrix)
    if spectrum.size == 0 or spectrum[0] == 0.0:
        return 0.0
    return float(np.sum(spectrum**2) / (spectrum[0] ** 2))


def theoretical_rank_bound(*, constituent_ranks: Sequence[int]) -> int:
    """The pre-overlap upper bound on a merged matrix's rank [AUTH: 00 §24A.6].

    A two-parent linear merge of rank-r constituents is bounded by 2r; a two-parent O3 merge
    at retained rank s by 2s. "Before accidental subspace overlap" is the point: the bound is
    an upper bound, and a lower realised rank is a fact about the parents, not a defect.
    """
    if not constituent_ranks:
        raise DiagnosticError("a rank bound needs at least one constituent")
    for rank in constituent_ranks:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise DiagnosticError(f"constituent rank {rank!r} must be a positive integer")
    return int(sum(constituent_ranks))


@dataclass(frozen=True)
class RankReport:
    """The three 00 §24A.6 rank quantities for one matrix, plus what produced them."""

    name: str
    theoretical_bound: int
    effective_rank: int
    stable_rank: float
    tolerance: float
    shape: tuple[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "theoretical_rank_bound": self.theoretical_bound,
            "numerical_effective_rank": self.effective_rank,
            "stable_rank": self.stable_rank,
            "effective_rank_tolerance": self.tolerance,
        }


def rank_report(
    vector: TaskVector, *, constituent_ranks: Sequence[int], tolerance: float
) -> list[RankReport]:
    """Per-target-matrix rank diagnostics for a merged descendant [AUTH: 00 §24A.6]."""
    bound = theoretical_rank_bound(constituent_ranks=constituent_ranks)
    return [
        RankReport(
            name=name,
            theoretical_bound=min(bound, min(vector.shapes[name])),
            effective_rank=numerical_effective_rank(vector[name], tolerance=tolerance),
            stable_rank=stable_rank(vector[name]),
            tolerance=tolerance,
            shape=vector.shapes[name],
        )
        for name in vector.names
    ]
