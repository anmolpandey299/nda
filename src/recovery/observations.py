"""The attack-time information firewall [AUTH: 00 §12, §13, §25].

This is S08's most important invariant. A recovery method must structurally receive only what
its lineage regime authorises, so no solver is ever handed an S07 `MergeResult` — that object
carries the protected constituent's identity, the partner's identity and the partner's
provenance, none of which an incomplete-lineage attacker holds.

Instead a factory reads a **genuine issued S07 object** and emits an **observation** carrying
only the authorised fields. The regime is derived from which factory ran, never from a caller
string:

    Z_HIGH                       descendant update · operator · partner update · coefficient
    Z_INCOMPLETE_FIXED           descendant updates · operator · one fixed known alpha
    Z_INCOMPLETE_VARYING_KNOWN   descendant updates · operator · per-descendant known alpha_i

    hidden alpha                 STRUCTURAL_NOT_AUTHORIZED — refused, not approximated

**Genuine S07 objects only.** The factories are typed on `MergeResult`, `ReleaseFamily` and
`TaskVector` and check with `isinstance`, so a `SimpleNamespace` or any other look-alike that
merely exposes `.update`, `.alpha`, `.operator` or `.partner_identity` cannot forge a public
coefficient, a partner attribution, an O3 retained rank or family membership. Verification
happens *during* issuance; the S07 object is then dropped, and the observation retains only
what the regime authorises.

**Canonical order.** Descendants are sorted by their public `descendant_id` before the
observation is built, so the same logical family presented in any caller order issues the same
observation identity and the same recovered bytes — a permutation cannot change float32
accumulation order [AUTH: 00 §25].

**Immutable storage.** Authoritative tensors live in an immutable tuple of
`(name, shape, dtype, bytes)` entries. No dict an object exposes can change the scientific
tensor set or its content, so a stored identity cannot go stale.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final

import numpy as np

from src.merge.family import MergeResult, ReleaseFamily
from src.merge.updates import Matrix, TaskVector
from src.provenance.hashing import JSONValue, sha256_canonical
from src.recovery.context import RecoveryExecutionContext

#: S08 shares one array alias with S07 [AUTH: 00 §22].
__all__ = [
    "LINEAGE_REGIMES",
    "Matrix",
    "NotAuthorizedError",
    "ObservationError",
    "ObservedLineage",
    "ObservedUpdate",
    "StructuralNAError",
    "observe_hidden_alpha",
    "observe_incomplete_fixed",
    "observe_incomplete_varying",
    "observe_known_partner",
]

OBSERVATION_SCHEMA: Final = "s08.observation.v2"

#: The authorised lineage regimes [AUTH: 00 §12].
Z_HIGH: Final = "Z_HIGH"
Z_INCOMPLETE_FIXED: Final = "Z_INCOMPLETE_FIXED"
Z_INCOMPLETE_VARYING_KNOWN: Final = "Z_INCOMPLETE_VARYING_KNOWN"
LINEAGE_REGIMES: Final[tuple[str, ...]] = (
    Z_HIGH,
    Z_INCOMPLETE_FIXED,
    Z_INCOMPLETE_VARYING_KNOWN,
)

#: Refused rather than approximated [AUTH: 00 §25].
STRUCTURAL_NOT_AUTHORIZED: Final = "STRUCTURAL_NOT_AUTHORIZED"
STRUCTURAL_NA: Final = "STRUCTURAL_NA"
METHOD_GATED: Final = "METHOD_GATED"

LINEAR: Final = "O1_LINEAR_TASK_ARITHMETIC"
DARE: Final = "O2_DARE"
SVD_TRUNC: Final = "O3_SVD_TRUNC_MERGE"

#: One immutable authoritative tensor: name, shape, dtype identity, bytes.
TensorEntry = tuple[str, tuple[int, int], str, bytes]


class ObservationError(ValueError):
    """An attacker observation cannot be formed as specified."""


class NotAuthorizedError(ObservationError):
    """The requested lineage regime or method is not authorised at S08."""


class StructuralNAError(ObservationError):
    """The quantity does not exist for this lineage; it is not a failed computation."""


# ----------------------------------------------------------------------------------------
# immutable observed tensors
# ----------------------------------------------------------------------------------------


def _canonical_entries(
    vector: TaskVector, context: RecoveryExecutionContext
) -> tuple[TensorEntry, ...]:
    """Freeze one task vector into immutable entries, in the frozen canonical name order."""
    entries: list[TensorEntry] = []
    for name in vector.names:
        array = np.asarray(vector[name], dtype=context.arithmetic)
        if not np.all(np.isfinite(array)):
            raise ObservationError(f"{name}: the observed update carries NaN or Inf")
        entries.append(
            (
                name,
                (int(array.shape[0]), int(array.shape[1])),
                context.arithmetic_dtype,
                np.ascontiguousarray(array, dtype=context.byte_order_code).tobytes("C"),
            )
        )
    return tuple(entries)


class ObservedUpdate:
    """One base-relative induced update as the attacker sees it.

    The authoritative storage is an immutable tuple of entries, so there is no ordinary
    write — through a public property or an underscore attribute — that can change the
    scientific bytes while a stored identity keeps claiming the old ones.
    """

    __slots__ = ("_context", "_entries")

    _entries: tuple[TensorEntry, ...]
    _context: RecoveryExecutionContext

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("ObservedUpdate is factory-issued")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("an observed update is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("an observed update is immutable")

    def _entry(self, name: str) -> TensorEntry:
        for entry in self._entries:
            if entry[0] == name:
                return entry
        raise ObservationError(f"{name!r} is not on this parameter surface")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry[0] for entry in self._entries)

    @property
    def shapes(self) -> dict[str, tuple[int, int]]:
        """A freshly built mapping. Mutating it cannot reach the authoritative entries."""
        return {entry[0]: entry[1] for entry in self._entries}

    @property
    def context(self) -> RecoveryExecutionContext:
        return self._context

    @property
    def dtype(self) -> str:
        return self._context.arithmetic_dtype

    def __getitem__(self, name: str) -> Matrix:
        _, shape, _, payload = self._entry(name)
        view = np.frombuffer(payload, dtype=self._context.byte_order_code)
        return view.reshape(shape)

    def copy_of(self, name: str) -> Matrix:
        """A writable copy, provably detached: it owns its own buffer."""
        return np.array(self[name], copy=True)

    def surface_identity(self) -> str:
        digest = hashlib.sha256(f"{OBSERVATION_SCHEMA}|surface".encode())
        for name, (rows, columns), dtype, _ in self._entries:
            digest.update(f"|{name}:{rows}x{columns}:{dtype}".encode())
        return digest.hexdigest()

    def content_identity(self) -> str:
        digest = hashlib.sha256(f"{OBSERVATION_SCHEMA}|content|{self.dtype}".encode())
        for name, _, _, payload in self._entries:
            digest.update(name.encode("utf-8"))
            digest.update(payload)
        return digest.hexdigest()


def _issue_update(vector: TaskVector, context: RecoveryExecutionContext) -> ObservedUpdate:
    observed = object.__new__(ObservedUpdate)
    object.__setattr__(observed, "_entries", _canonical_entries(vector, context))
    object.__setattr__(observed, "_context", context)
    return observed


# ----------------------------------------------------------------------------------------
# lineage observations
# ----------------------------------------------------------------------------------------


class ObservedLineage:
    """Exactly what one recovery method may see [AUTH: 00 §12, §25].

    For an incomplete regime the partner update and every hidden identity are simply absent:
    there is no slot holding them, so no attribute access and no `getattr` can reach them.
    """

    __slots__ = (
        "_alphas",
        "_context",
        "_descendant_ids",
        "_identity",
        "_operator",
        "_partner",
        "_regime",
        "_retained_rank",
        "_updates",
    )

    _regime: str
    _operator: str
    _updates: tuple[ObservedUpdate, ...]
    _descendant_ids: tuple[str, ...]
    _alphas: tuple[float, ...]
    _partner: ObservedUpdate | None
    _retained_rank: int | None
    _context: RecoveryExecutionContext
    _identity: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "ObservedLineage is factory-issued; use observe_known_partner,"
            " observe_incomplete_fixed or observe_incomplete_varying so the regime is derived"
            " from what was actually stripped [AUTH: 00 §12]"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("an observation is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("an observation is immutable")

    # ---------------------------------------------------------------- projections
    @property
    def regime(self) -> str:
        return self._regime

    @property
    def operator(self) -> str:
        return self._operator

    @property
    def k(self) -> int:
        return len(self._updates)

    @property
    def updates(self) -> tuple[ObservedUpdate, ...]:
        return self._updates

    @property
    def descendant_ids(self) -> tuple[str, ...]:
        """Public ids, in the canonical order the observation was built in."""
        return self._descendant_ids

    @property
    def alphas(self) -> tuple[float, ...]:
        """The public coefficients. Authorised in every implemented regime [00 §25]."""
        return self._alphas

    @property
    def fixed_alpha(self) -> float:
        """The one shared coefficient. Refuses when the family is not fixed-alpha."""
        if len(set(self._alphas)) != 1:
            raise ObservationError(
                "this lineage carries varying coefficients; use the varying-alpha path"
            )
        return self._alphas[0]

    @property
    def partner(self) -> ObservedUpdate:
        """The known partner. Only a Z-HIGH observation has one."""
        if self._partner is None:
            raise NotAuthorizedError(
                f"partner information is not authorised under {self._regime}; a"
                " known-partner method may not run on an incomplete-lineage observation"
                " [AUTH: 00 §12, §25]"
            )
        return self._partner

    @property
    def has_partner(self) -> bool:
        return self._partner is not None

    @property
    def retained_rank(self) -> int | None:
        """Public O3 metadata: the retained rank the descendants were truncated at."""
        return self._retained_rank

    @property
    def context(self) -> RecoveryExecutionContext:
        return self._context

    @property
    def surface_identity(self) -> str:
        return self._updates[0].surface_identity()

    def identity(self) -> str:
        """The observation identity every compared method must share [B23]."""
        return self._identity

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": OBSERVATION_SCHEMA,
            "regime": self._regime,
            "operator": self._operator,
            "k": self.k,
            "descendant_ids": list(self._descendant_ids),
            "alphas": list(self._alphas),
            "retained_rank": self._retained_rank,
            "has_partner": self._partner is not None,
            "parameter_surface_sha256": self.surface_identity,
            "observation_sha256": self._identity,
            "recovery_context_sha256": self._context.identity(),
        }


def _issue_lineage(
    *,
    regime: str,
    operator: str,
    updates: Sequence[ObservedUpdate],
    descendant_ids: Sequence[str],
    alphas: Sequence[float],
    partner: ObservedUpdate | None,
    retained_rank: int | None,
    context: RecoveryExecutionContext,
) -> ObservedLineage:
    surfaces = {update.surface_identity() for update in updates}
    if len(surfaces) != 1:
        raise ObservationError("the observed descendants do not share one parameter surface")
    for alpha in alphas:
        if not np.isfinite(alpha):
            raise ObservationError(f"public coefficient {alpha!r} is not finite")

    payload: dict[str, JSONValue] = {
        "schema": OBSERVATION_SCHEMA,
        "regime": regime,
        "operator": operator,
        "descendant_ids": list(descendant_ids),
        "alphas": [float(a) for a in alphas],
        "retained_rank": retained_rank,
        "update_identities": [update.content_identity() for update in updates],
        "partner_identity": partner.content_identity() if partner is not None else None,
        "parameter_surface_sha256": updates[0].surface_identity(),
        "recovery_context_sha256": context.identity(),
    }

    lineage = object.__new__(ObservedLineage)
    object.__setattr__(lineage, "_regime", regime)
    object.__setattr__(lineage, "_operator", operator)
    object.__setattr__(lineage, "_updates", tuple(updates))
    object.__setattr__(lineage, "_descendant_ids", tuple(descendant_ids))
    object.__setattr__(lineage, "_alphas", tuple(float(a) for a in alphas))
    object.__setattr__(lineage, "_partner", partner)
    object.__setattr__(lineage, "_retained_rank", retained_rank)
    object.__setattr__(lineage, "_context", context)
    object.__setattr__(lineage, "_identity", sha256_canonical(payload))
    return lineage


def _require_merge_result(candidate: object, *, what: str) -> MergeResult:
    """Only a genuine factory-issued S07 result may found an observation [AUTH: 01 §16].

    Duck typing is what makes a forged coefficient, partner attribution or retained rank
    possible at all: an object that merely exposes the right attribute names would otherwise
    be indistinguishable from a descendant S07 actually built.
    """
    if not isinstance(candidate, MergeResult):
        raise ObservationError(
            f"{what} must be a factory-issued S07 MergeResult, not"
            f" {type(candidate).__name__}; an object that merely exposes .update, .alpha,"
            " .operator or .partner_identity is not a descendant this study built"
            " [AUTH: 00 §12; 01 §16]"
        )
    return candidate


def _require_task_vector(candidate: object, *, what: str) -> TaskVector:
    if not isinstance(candidate, TaskVector):
        raise ObservationError(
            f"{what} must be a factory-issued S07 task vector, not {type(candidate).__name__}"
        )
    return candidate


def _canonical_order(results: Sequence[MergeResult]) -> tuple[MergeResult, ...]:
    """Sort by public descendant id, so caller order cannot change the observation.

    The key is public metadata, not content, so canonicalisation reveals nothing the regime
    does not already authorise.
    """
    return tuple(sorted(results, key=lambda result: result.descendant_id))


def _public_rank(results: Sequence[MergeResult]) -> int | None:
    """The O3 retained rank, which is public merge metadata, or None for other operators."""
    ranks = {result.retained_rank for result in results}
    if ranks == {None}:
        return None
    if len(ranks) != 1:
        raise ObservationError(
            "the descendants were truncated at different retained ranks"
            f" {sorted(r for r in ranks if r is not None)}; one recovery family shares one"
            " rank [AUTH: 00 §10.3]"
        )
    return next(iter(ranks))


def observe_known_partner(
    *, descendant: MergeResult, partner_update: TaskVector, context: RecoveryExecutionContext
) -> ObservedLineage:
    """Z-HIGH: descendant, operator, coefficient and the known partner [AUTH: 00 §12].

    Every scientific field is derived from the issued S07 result — the operator, the
    coefficient, the descendant id and the retained rank are read off the object that actually
    ran, never supplied. The partner's content identity is then checked against the
    descendant's own record, which is authorised precisely because the partner is known here.
    """
    result = _require_merge_result(descendant, what="the observed descendant")
    partner_vector = _require_task_vector(partner_update, what="the known partner")
    if partner_vector.content_identity() != result.partner_identity:
        raise ObservationError(
            "the supplied partner is not the partner this descendant was built from;"
            " a known-partner observation pairs a descendant with its own partner"
        )
    return _issue_lineage(
        regime=Z_HIGH,
        operator=result.operator,
        updates=[_issue_update(result.update, context)],
        descendant_ids=[result.descendant_id],
        alphas=[result.alpha],
        partner=_issue_update(partner_vector, context),
        retained_rank=_public_rank([result]),
        context=context,
    )


def _incomplete(
    family: ReleaseFamily, *, regime: str, context: RecoveryExecutionContext
) -> ObservedLineage:
    """Strip a genuine S07 release family down to what an incomplete attacker holds.

    Taking a `ReleaseFamily` rather than a loose sequence inherits S07's own guarantees: every
    descendant contains the same trained draw of A, on one surface, under one execution
    context, with distinct ids [AUTH: 00 §3.4, §11].
    """
    if not isinstance(family, ReleaseFamily):
        raise ObservationError(
            "an incomplete-lineage observation is built from a factory-issued S07"
            f" ReleaseFamily, not {type(family).__name__}; a loose sequence of descendants"
            " carries no guarantee that they share one protected constituent [AUTH: 00 §3.4]"
        )
    descendants = _canonical_order(family.descendants)
    if len(descendants) < 2:
        raise StructuralNAError(
            f"{STRUCTURAL_NA}: a shared-source recovery needs k >= 2 unknown-partner"
            " descendants; k = 1 cannot separate a common source from its residual"
            " [AUTH: 00 §11, §25]"
        )
    permitted = _permitted_k()
    if len(descendants) not in permitted:
        raise ObservationError(
            f"descendant count k = {len(descendants)} is not a registered value"
            f" {sorted(permitted)} [AUTH: 00 §11, §25]"
        )
    ids = [result.descendant_id for result in descendants]
    if len(set(ids)) != len(ids):
        raise ObservationError(f"the family repeats descendant id(s): {sorted(set(ids))}")
    identities = {result.identity() for result in descendants}
    if len(identities) != len(descendants):
        raise ObservationError(
            "the family lists the same descendant more than once; a repeated release is not"
            " an additional observation of A [AUTH: 00 §11]"
        )

    operators = {result.operator for result in descendants}
    if len(operators) != 1:
        raise ObservationError(f"the descendants mix operators {sorted(operators)}")
    operator = next(iter(operators))
    if operator == DARE:
        raise NotAuthorizedError(
            f"{METHOD_GATED}: no incomplete-lineage DARE recovery method is implemented at"
            " S08 [AUTH: 00 §25; 01 §39 S08]"
        )
    alphas = [result.alpha for result in descendants]
    if any(alpha == 0.0 for alpha in alphas):
        raise ObservationError("a zero coefficient carries no information about A")
    return _issue_lineage(
        regime=regime,
        operator=operator,
        updates=[_issue_update(result.update, context) for result in descendants],
        descendant_ids=ids,
        alphas=alphas,
        partner=None,
        retained_rank=_public_rank(descendants),
        context=context,
    )


def _permitted_k() -> frozenset[int]:
    """The registered descendant counts [AUTH: 00 §11]. Read from config, never hard-coded."""
    from pathlib import Path

    from src.recovery.settings import descendant_counts, recovery_settings

    root = Path(__file__).resolve().parents[2]
    return frozenset(descendant_counts(recovery_settings(root)))


def observe_incomplete_fixed(
    *, family: ReleaseFamily, context: RecoveryExecutionContext
) -> ObservedLineage:
    """Z-INCOMPLETE-FIXED: descendant updates, operator and one fixed known alpha.

    Partner bytes, partner identities and the protected identity are not copied across, so the
    solver has no attribute through which to reach them.
    """
    if not isinstance(family, ReleaseFamily):
        raise ObservationError(
            "an incomplete-lineage observation is built from a factory-issued S07"
            f" ReleaseFamily, not {type(family).__name__}"
        )
    alphas = {result.alpha for result in family.descendants}
    if len(alphas) != 1:
        raise ObservationError(
            f"a fixed-coefficient lineage requires one shared alpha, found {sorted(alphas)}"
        )
    return _incomplete(family, regime=Z_INCOMPLETE_FIXED, context=context)


def observe_incomplete_varying(
    *, family: ReleaseFamily, context: RecoveryExecutionContext
) -> ObservedLineage:
    """Z-INCOMPLETE-VARYING-KNOWN: descendant updates and each public alpha_i."""
    return _incomplete(family, regime=Z_INCOMPLETE_VARYING_KNOWN, context=context)


def observe_hidden_alpha(*args: object, **kwargs: object) -> ObservedLineage:
    """Unknown partner plus unknown coefficient. Refused, never approximated [00 §25]."""
    raise NotAuthorizedError(
        f"{STRUCTURAL_NOT_AUTHORIZED}: 00 §25 authorises known-coefficient recovery regimes"
        " only; there is no hidden-alpha method at S08 and none is inferred"
    )
