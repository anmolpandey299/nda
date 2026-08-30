"""The evaluator-only truth binding [AUTH: 00 §22, §24A, §25; 01 §16].

S08 emits scientific parameter-recovery quantities — e_F, the O3 error decomposition, the C1
pass decision — and every one of them is a statement about a *specific* protected constituent.
Letting an evaluator take `(result, some_task_vector)` means an A_1 recovery can be scored
against an unrelated A_2 of the same shape and come back with an ordinary-looking number. That
is a silent misattribution, not a caught error.

So truth never travels loose. A `EvaluationTruthBinding` is opaque, factory-issued and
immutable, and the factory proves the attribution before issuing:

1. the supplied genuine S07 source is **re-observed** through the same S08 factory, and the
   resulting observation identity must equal the identity the recovery actually ran on;
2. the supplied protected task vector's content identity must equal the protected identity the
   S07 source records;
3. for O3, the operator and the retained rank must match the recovery's own.

Only then is a binding issued. The attacker-visible observation gains nothing from any of
this: the protected identity and bytes live here, on the evaluator side of the firewall, and
`src.recovery.solvers` does not import this module.
"""

from __future__ import annotations

from typing import Final

from src.merge.family import MergeResult, ReleaseFamily
from src.merge.updates import TaskVector
from src.provenance.hashing import JSONValue, sha256_canonical
from src.recovery.context import RecoveryExecutionContext
from src.recovery.observations import (
    SVD_TRUNC,
    Z_HIGH,
    Z_INCOMPLETE_FIXED,
    Z_INCOMPLETE_VARYING_KNOWN,
    ObservedLineage,
    observe_incomplete_fixed,
    observe_incomplete_varying,
    observe_known_partner,
)

BINDING_SCHEMA: Final = "s08.evaluation-truth-binding.v1"


class TruthBindingError(ValueError):
    """The supplied truth cannot be attributed to the recovery being evaluated."""


class EvaluationTruthBinding:
    """One proven (observation, protected truth) pair. Evaluator-only.

    Opaque and factory-issued: `__init__` raises, `__setattr__` raises, and this is not a
    dataclass. No solver accepts one, and no solver module imports this one.
    """

    __slots__ = (
        "_context",
        "_identity",
        "_observation_identity",
        "_operator",
        "_protected_identity",
        "_protected_truth",
        "_regime",
        "_retained_rank",
        "_source_identity",
    )

    _observation_identity: str
    _protected_truth: TaskVector
    _protected_identity: str
    _source_identity: str
    _operator: str
    _regime: str
    _retained_rank: int | None
    _context: RecoveryExecutionContext
    _identity: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "EvaluationTruthBinding is factory-issued; use bind_truth(), which re-observes the"
            " genuine S07 source and refuses a truth the source did not produce"
            " [AUTH: 00 §22, §24A]"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a truth binding is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a truth binding is immutable")

    # ---------------------------------------------------------------- projections
    @property
    def observation_identity(self) -> str:
        return self._observation_identity

    @property
    def protected_truth(self) -> TaskVector:
        """The bound protected constituent. Reachable only from the evaluator side."""
        return self._protected_truth

    @property
    def protected_identity(self) -> str:
        return self._protected_identity

    @property
    def operator(self) -> str:
        return self._operator

    @property
    def regime(self) -> str:
        return self._regime

    @property
    def retained_rank(self) -> int | None:
        return self._retained_rank

    @property
    def context(self) -> RecoveryExecutionContext:
        return self._context

    @property
    def source_identity(self) -> str:
        """The genuine S07 source this binding was proved against."""
        return self._source_identity

    def identity(self) -> str:
        return self._identity

    def as_dict(self) -> dict[str, JSONValue]:
        """Evaluator-side provenance. Never handed to a solver."""
        return {
            "schema": BINDING_SCHEMA,
            "observation_sha256": self._observation_identity,
            "protected_update_sha256": self._protected_identity,
            "s07_source_sha256": self._source_identity,
            "operator": self._operator,
            "lineage_regime": self._regime,
            "retained_rank": self._retained_rank,
            "recovery_context_sha256": self._context.identity(),
            "truth_binding_sha256": self._identity,
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"EvaluationTruthBinding({self._regime}, {self._identity[:12]})"


def _reobserve(
    source: MergeResult | ReleaseFamily, observation: ObservedLineage
) -> ObservedLineage:
    """Re-issue the attacker view from the genuine source, through the same factories.

    Deriving the check from the real factories rather than from a comparison of fields is what
    makes it complete: anything the observation identity covers is covered here too.
    """
    context = observation.context
    if isinstance(source, MergeResult):
        if observation.regime != Z_HIGH:
            raise TruthBindingError(
                f"a single MergeResult founds a {Z_HIGH} observation; this recovery ran under"
                f" {observation.regime}, which comes from a ReleaseFamily"
            )
        # In Z-HIGH the partner is known, so re-deriving it is authorised; the descendant's own
        # record says which partner that is, and only that partner reproduces the observation.
        partner = _partner_of(source, observation)
        return observe_known_partner(descendant=source, partner_update=partner, context=context)
    if isinstance(source, ReleaseFamily):
        if observation.regime == Z_INCOMPLETE_FIXED:
            return observe_incomplete_fixed(family=source, context=context)
        if observation.regime == Z_INCOMPLETE_VARYING_KNOWN:
            return observe_incomplete_varying(family=source, context=context)
        raise TruthBindingError(
            f"a ReleaseFamily founds an incomplete-lineage observation; this one is"
            f" {observation.regime}"
        )
    raise TruthBindingError(
        "the evaluation source must be a factory-issued S07 MergeResult or ReleaseFamily, not"
        f" {type(source).__name__}"
    )


def _partner_of(source: MergeResult, observation: ObservedLineage) -> TaskVector:
    """The Z-HIGH partner, recovered from the observation itself.

    The observation already carries the partner it was issued with, and
    `observe_known_partner` verified at issuance that it is this descendant's own partner. So
    the re-observation reuses that object rather than asking the caller for one again — a
    second caller-supplied partner would be one more thing to forge.
    """
    from src.merge.updates import fixture_induced_update

    partner = observation.partner
    rebuilt = fixture_induced_update(
        {name: partner[name] for name in partner.names},
        context=source.context,
        origin="rebound-partner",
    )
    if rebuilt.content_identity() != source.partner_identity:
        raise TruthBindingError(
            "the observation's partner is not the partner this descendant was built from"
        )
    return rebuilt


def bind_truth(
    *,
    observation: ObservedLineage,
    source: MergeResult | ReleaseFamily,
    protected_truth: TaskVector,
) -> EvaluationTruthBinding:
    """Prove that `protected_truth` is what `observation` was actually built from.

    Two independent facts are checked, and both must hold:

    * re-observing the genuine S07 source reproduces the exact observation identity the
      recovery ran on — so a family for A_1 cannot be bound to an observation of A_2;
    * the supplied protected constituent's content identity equals the protected identity the
      S07 source itself recorded — so the right family cannot be bound to the wrong A.
    """
    if not isinstance(protected_truth, TaskVector):
        raise TruthBindingError(
            "the protected truth must be a factory-issued S07 task vector, not"
            f" {type(protected_truth).__name__}"
        )
    rebuilt = _reobserve(source, observation)
    if rebuilt.identity() != observation.identity():
        raise TruthBindingError(
            "the supplied S07 source does not reproduce this observation; the recovery being"
            " evaluated was not built from it [AUTH: 00 §22, §24A]"
        )
    if protected_truth.content_identity() != source.protected_identity:
        raise TruthBindingError(
            "the supplied protected constituent is not the one this lineage was built from;"
            " evaluating a recovery of A_1 against an unrelated A_2 is a misattribution, not a"
            " metric [AUTH: 00 §22]"
        )

    operator = observation.operator
    retained_rank = observation.retained_rank
    if operator == SVD_TRUNC and retained_rank is None:  # pragma: no cover - factory refuses
        raise TruthBindingError("an O3 lineage must carry its public retained rank")

    payload: dict[str, JSONValue] = {
        "schema": BINDING_SCHEMA,
        "observation_sha256": observation.identity(),
        "protected_update_sha256": protected_truth.content_identity(),
        "s07_source_sha256": _source_identity(source),
        "operator": operator,
        "lineage_regime": observation.regime,
        "retained_rank": retained_rank,
        "recovery_context_sha256": observation.context.identity(),
    }

    binding = object.__new__(EvaluationTruthBinding)
    object.__setattr__(binding, "_observation_identity", observation.identity())
    object.__setattr__(binding, "_protected_truth", protected_truth)
    object.__setattr__(binding, "_protected_identity", protected_truth.content_identity())
    object.__setattr__(binding, "_source_identity", _source_identity(source))
    object.__setattr__(binding, "_operator", operator)
    object.__setattr__(binding, "_regime", observation.regime)
    object.__setattr__(binding, "_retained_rank", retained_rank)
    object.__setattr__(binding, "_context", observation.context)
    object.__setattr__(binding, "_identity", sha256_canonical(payload))
    return binding


def _source_identity(source: MergeResult | ReleaseFamily) -> str:
    if isinstance(source, MergeResult):
        return source.identity()
    return sha256_canonical(
        {
            "schema": f"{BINDING_SCHEMA}|family",
            "protected_update_sha256": source.protected_identity,
            "descendants": [result.identity() for result in source.descendants],
        }
    )


def o3_source_for(binding: EvaluationTruthBinding, source: ReleaseFamily) -> MergeResult:
    """The genuine O3 release the floor must be computed from [AUTH: 00 §24A.5, §28B].

    The floor is a property of a specific truncation, so it is taken from a descendant of the
    bound family at the bound retained rank — never from an unrelated O3 result a caller
    happens to hold.
    """
    if binding.operator != SVD_TRUNC:
        raise TruthBindingError(
            f"an O3 floor needs an {SVD_TRUNC} lineage; this binding is {binding.operator}"
        )
    if _source_identity(source) != binding.source_identity:
        raise TruthBindingError("this release family is not the one the binding was issued for")
    for result in source.descendants:
        if result.retained_rank == binding.retained_rank:
            return result
    raise TruthBindingError(  # pragma: no cover - the family shares one rank by construction
        f"no descendant of the bound family was truncated at rank {binding.retained_rank}"
    )
