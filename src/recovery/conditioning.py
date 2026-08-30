"""Direction-matrix conditioning [AUTH: 00 §26].

For descendants whose partner is known, D_i = vec(C_i - B_i) and D = [D_1, ..., D_k]. The
conditioning number and effective numerical rank of D are explanatory variables for recovery
error — they are not a recovery input and not a novelty claim.

**Information boundary.** `C_i - B_i` needs the partner, so this is available only where the
experiment design authorises known-partner information. It is deliberately not reachable from
an incomplete-lineage attack path: `direction_matrix` takes a Z-HIGH observation, and the
incomplete factories never attach a partner, so an incomplete observation raises when asked
for one [AUTH: 00 §12, §26].

The numerical-rank tolerance is the project's already-frozen 00 §24A.6 criterion,
sigma_j / sigma_1 >= 1e-6, resolved from config rather than restated here. Reusing the
existing frozen tolerance is a pre-data implementation adjudication; inventing a second one
would create two numerical-rank criteria in one study.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np

from src.merge.updates import Matrix
from src.provenance.hashing import JSONValue
from src.recovery.context import RecoveryExecutionContext
from src.recovery.observations import Z_HIGH, NotAuthorizedError, ObservedLineage

CONDITIONING_VERSION: Final = "s08.conditioning.v1"

#: A direction family with no non-zero singular value has no conditioning number.
RANK_DEFICIENT: Final = "UNDEFINED_RANK_DEFICIENT"


class ConditioningError(ValueError):
    """A conditioning diagnostic is not defined for the inputs given."""


def direction_matrix(observations: Sequence[ObservedLineage]) -> Matrix:
    """D = [vec(C_1 - B_1), ..., vec(C_k - B_k)] from known-partner observations [00 §26].

    Each observation must be Z-HIGH. An incomplete-lineage observation has no partner, so it
    cannot reach this quantity at all.
    """
    if not observations:
        raise ConditioningError("a direction matrix needs at least one descendant")
    columns: list[Matrix] = []
    surfaces = set()
    for observation in observations:
        if not isinstance(observation, ObservedLineage):
            raise ConditioningError(
                "conditioning reads a factory-issued ObservedLineage, not"
                f" {type(observation).__name__} [AUTH: 00 §12, §26]"
            )
        if observation.regime != Z_HIGH:
            raise NotAuthorizedError(
                f"conditioning uses C_i - B_i, which needs the partner; this observation is"
                f" {observation.regime} and holds none [AUTH: 00 §12, §26]"
            )
        descendant, partner = observation.updates[0], observation.partner
        surfaces.add(descendant.surface_identity())
        wide = observation.context.diagnostic
        columns.append(
            np.concatenate(
                [
                    (
                        np.asarray(descendant[name], dtype=wide)
                        - np.asarray(partner[name], dtype=wide)
                    ).ravel()
                    for name in descendant.names
                ]
            )
        )
    if len(surfaces) != 1:
        raise ConditioningError("the descendants do not share one parameter surface")
    return np.column_stack(columns)


def conditioning_report(
    matrix: Matrix, *, context: RecoveryExecutionContext
) -> dict[str, JSONValue]:
    """Singular values, effective numerical rank and kappa [AUTH: 00 §26, §24A.6].

    kappa = sigma_1 / sigma_2 for k = 2, and sigma_max / sigma_min_nonzero for k > 2. A zero
    direction matrix reports effective rank 0 and an explicit rank-deficient state; Inf and
    NaN are never returned as an ordinary kappa.
    """
    array = np.asarray(matrix, dtype=context.diagnostic)
    if array.ndim != 2:
        raise ConditioningError("a direction matrix is 2-D: rows are coordinates, columns are k")
    tolerance = context.conditioning_rank_tolerance
    singular = np.linalg.svd(array, compute_uv=False)
    k = int(array.shape[1])

    if singular.size == 0 or singular[0] == 0.0:
        return {
            "conditioning_version": CONDITIONING_VERSION,
            "k": k,
            "singular_values": [float(v) for v in singular],
            "effective_numerical_rank": 0,
            "numerical_rank_tolerance": tolerance,
            "kappa": RANK_DEFICIENT,
        }

    ratios = singular / singular[0]
    effective = int(np.count_nonzero(ratios >= tolerance))
    kept = singular[:effective]
    if k == 2:
        # 00 §26 fixes kappa = sigma_1 / sigma_2 for k = 2, when sigma_2 is numerically non-zero.
        second = float(singular[1]) if singular.size > 1 else 0.0
        kappa: JSONValue = (
            RANK_DEFICIENT if effective < 2 or second == 0.0 else float(singular[0]) / second
        )
    else:
        kappa = RANK_DEFICIENT if effective < 1 else float(kept[0]) / float(kept[effective - 1])
    return {
        "conditioning_version": CONDITIONING_VERSION,
        "k": k,
        "singular_values": [float(v) for v in singular],
        "effective_numerical_rank": effective,
        "numerical_rank_tolerance": tolerance,
        "kappa": kappa,
    }


def conditioning_from_observations(
    observations: Sequence[ObservedLineage], *, context: RecoveryExecutionContext
) -> dict[str, JSONValue]:
    """The known-partner conditioning report for one descendant family [AUTH: 00 §26]."""
    return conditioning_report(direction_matrix(observations), context=context)
