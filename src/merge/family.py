"""Factory-controlled merge results and release families [AUTH: 00 §3.4, §10, §11; 01 §16].

00 §3.4 fixes the structural claim a release family makes: every descendant contains the
**same trained draw of A**, and A is not re-trained per descendant.

Two things are enforced structurally here rather than by convention.

**Scientific identity is derived, never supplied.** A `MergeResult` cannot be constructed by
an ordinary caller: `__init__` is generated but refuses without a module-private capability
token, so the only ways to obtain one are `build_linear_descendant`, `build_dare_descendant`
and `build_o3_descendant`. Each recomputes every identity from the immutable inputs and the
execution context. There is no generic factory whose optional arguments choose an operator,
so DARE bytes cannot be issued as a LINEAR result, and `dataclasses.replace` cannot promote
`provenance_class` because it is derived from the class, not stored.

**A merge is not a run.** `provenance_class` is fixed at `NON_EVIDENTIARY_S07_MERGE`. S09 may
later bind a production merge into a complete run manifest; it may not promote this object by
replacing a string.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from src.merge.context import MergeExecutionContext
from src.merge.masks import DareMask
from src.merge.operators import (
    DARE,
    LINEAR,
    SVD_TRUNC,
    Truncation,
    dare_merge,
    linear_merge,
    require_supported_operator,
    svd_trunc_merge,
    truncation_set_identity,
)
from src.merge.updates import TaskVector, require_compatible
from src.provenance.hashing import JSONValue, sha256_canonical

FAMILY_SCHEMA: Final = "s07.merge-family.v2"

#: An S07 merge artifact is never a final evidentiary run artifact [AUTH: 01 §14, §16].
NON_EVIDENTIARY: Final = "NON_EVIDENTIARY_S07_MERGE"

#: The capability token. A result refuses to exist without it, and only the three factories
#: below hold it.
_ISSUER: Final = object()


class MergeFamilyError(ValueError):
    """A merge family cannot be constructed as specified."""


@dataclass(frozen=True)
class MergeSpec:
    """What one descendant was asked for, before it was built.

    Only the *request* lives here — ids, operator, coefficient, p, merge seed, rank. No
    identity hash is a spec field, because a caller must not be able to state one.
    """

    descendant_id: str
    operator: str
    alpha: float
    partner_id: str
    drop_probability: float | None = None
    dare_merge_seed: int | None = None
    retained_rank: int | None = None

    def __post_init__(self) -> None:
        require_supported_operator(self.operator)
        if not self.descendant_id or not self.partner_id:
            raise MergeFamilyError("a descendant and its partner both need identities")
        if self.operator == DARE:
            if self.drop_probability is None:
                raise MergeFamilyError(f"{self.descendant_id}: a DARE merge needs its p")
            if self.dare_merge_seed is None:
                raise MergeFamilyError(
                    f"{self.descendant_id}: a DARE merge needs its one declared merge seed;"
                    " an undeclared mask is not reproducible [AUTH: 01 §30]"
                )
        elif self.drop_probability is not None or self.dare_merge_seed is not None:
            raise MergeFamilyError(
                f"{self.descendant_id}: DARE parameters were given for a {self.operator} merge"
            )
        if self.operator == SVD_TRUNC:
            if self.retained_rank is None:
                raise MergeFamilyError(f"{self.descendant_id}: an O3 merge needs its rank")
        elif self.retained_rank is not None:
            raise MergeFamilyError(
                f"{self.descendant_id}: a retained rank was given for a {self.operator} merge"
            )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "descendant_id": self.descendant_id,
            "operator": self.operator,
            "alpha": self.alpha,
            "partner_id": self.partner_id,
            "drop_probability": self.drop_probability,
            "dare_merge_seed": self.dare_merge_seed,
            "retained_rank": self.retained_rank,
        }


class MergeResult:
    """One built descendant. Opaque, immutable and factory-issued.

    `__init__` raises, `__setattr__` raises, and this is not a dataclass, so there is no
    `dataclasses.replace` path and no supported way to supply an operator, an input identity,
    a surface identity, a coefficient, a context hash, a result hash, mask evidence, rank
    evidence or a provenance class. Every one of those is derived by a factory from the
    immutable inputs and the operation that actually ran.
    """

    __slots__ = (
        "_alpha",
        "_context",
        "_descendant_id",
        "_identity",
        "_operator",
        "_partner_identity",
        "_partner_label",
        "_partner_mask",
        "_partner_provenance",
        "_partner_truncation_identity",
        "_partner_truncations",
        "_protected_identity",
        "_protected_mask",
        "_protected_provenance",
        "_protected_truncation_identity",
        "_protected_truncations",
        "_retained_rank",
        "_dare_merge_seed",
        "_drop_probability",
        "_surface_identity",
        "_update",
    )

    _operator: str
    _descendant_id: str
    _partner_label: str
    _alpha: float
    _update: TaskVector
    _protected_identity: str
    _partner_identity: str
    _protected_provenance: str
    _partner_provenance: str
    _surface_identity: str
    _context: MergeExecutionContext
    _drop_probability: float | None
    _dare_merge_seed: int | None
    _protected_mask: DareMask | None
    _partner_mask: DareMask | None
    _retained_rank: int | None
    _protected_truncations: Mapping[str, Truncation] | None
    _partner_truncations: Mapping[str, Truncation] | None
    _protected_truncation_identity: str | None
    _partner_truncation_identity: str | None
    _identity: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "MergeResult is factory-issued; use build_linear_descendant,"
            " build_dare_descendant or build_o3_descendant. Scientific identity derives from"
            " the inputs and the operation, and is never supplied [AUTH: 01 §16]"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a merge result is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a merge result is immutable")

    # ---------------------------------------------------------------- projections
    @property
    def provenance_class(self) -> str:
        """Derived and permanent. S09 binds a run manifest; it does not promote this."""
        return NON_EVIDENTIARY

    @property
    def operator(self) -> str:
        return self._operator

    @property
    def descendant_id(self) -> str:
        return self._descendant_id

    @property
    def partner_label(self) -> str:
        """A display label only. Scientific partner identity is `partner_identity`."""
        return self._partner_label

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def update(self) -> TaskVector:
        return self._update

    @property
    def protected_identity(self) -> str:
        return self._protected_identity

    @property
    def partner_identity(self) -> str:
        return self._partner_identity

    @property
    def surface_identity(self) -> str:
        return self._surface_identity

    @property
    def context(self) -> MergeExecutionContext:
        return self._context

    @property
    def execution_context_sha256(self) -> str:
        return self._context.identity()

    @property
    def protected_mask(self) -> DareMask | None:
        return self._protected_mask

    @property
    def partner_mask(self) -> DareMask | None:
        return self._partner_mask

    @property
    def retained_rank(self) -> int | None:
        return self._retained_rank

    @property
    def protected_truncations(self) -> Mapping[str, Truncation] | None:
        return self._protected_truncations

    @property
    def partner_truncations(self) -> Mapping[str, Truncation] | None:
        return self._partner_truncations

    @property
    def protected_truncation_identity(self) -> str | None:
        return self._protected_truncation_identity

    @property
    def result_identity(self) -> str:
        return self._update.content_identity()

    def comparison_identity(self) -> str:
        """What makes two descendants the same comparison at different operators [F12].

        Deliberately excludes the operator and its parameters: two results share this identity
        exactly when they merge the same A and B at the same α on the same surface under the
        same context.
        """
        return sha256_canonical(
            {
                "schema": f"{FAMILY_SCHEMA}|comparison",
                "protected_update_sha256": self._protected_identity,
                "partner_update_sha256": self._partner_identity,
                "alpha": self._alpha,
                "parameter_surface_sha256": self._surface_identity,
                "execution_context_sha256": self._context.identity(),
                "arithmetic_dtype": self._context.arithmetic_dtype,
            }
        )

    def as_dict(self) -> dict[str, JSONValue]:
        """The S09 handoff: operator facts only, no scientific outcome [AUTH: 01 §16]."""
        document: dict[str, JSONValue] = {
            "schema": FAMILY_SCHEMA,
            "provenance_class": NON_EVIDENTIARY,
            "descendant_id": self._descendant_id,
            "operator": self._operator,
            "operator_version": self._context.operator_version,
            "execution_context_sha256": self._context.identity(),
            "config_sha256": self._context.config_sha256,
            "arithmetic_dtype": self._context.arithmetic_dtype,
            "alpha": self._alpha,
            "partner_id": self._partner_label,
            "protected_update_sha256": self._protected_identity,
            "partner_update_sha256": self._partner_identity,
            "protected_provenance_class": self._protected_provenance,
            "partner_provenance_class": self._partner_provenance,
            "parameter_surface_sha256": self._surface_identity,
            "result_update_sha256": self.result_identity,
            "dtype": self._update.dtype,
            "drop_probability": self._drop_probability,
            "dare_merge_seed": self._dare_merge_seed,
            "retained_rank": self._retained_rank,
            "merge_result_sha256": self._identity,
        }
        if self._protected_mask is not None and self._partner_mask is not None:
            document["mask_scheme"] = self._context.mask_scheme
            document["protected_mask_sha256"] = self._protected_mask.identity()
            document["partner_mask_sha256"] = self._partner_mask.identity()
            document["retained_coordinates"] = (
                self._protected_mask.retained + self._partner_mask.retained
            )
            document["dropped_coordinates"] = (
                self._protected_mask.dropped + self._partner_mask.dropped
            )
        if self._protected_truncations is not None and self._partner_truncations is not None:
            document["svd_backend"] = self._context.svd_backend
            document["svd_workspace_dtype"] = self._context.svd_workspace_dtype
            document["protected_truncation_sha256"] = self._protected_truncation_identity
            document["partner_truncation_sha256"] = self._partner_truncation_identity
        return document

    def identity(self) -> str:
        """The canonical result identity [F5]. Computed at issue, never recomputed here."""
        return self._identity

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"MergeResult({self._descendant_id!r}, {self._operator}, {self._identity[:12]})"


def _issue_result(
    *,
    operator: str,
    spec: MergeSpec,
    protected: TaskVector,
    partner: TaskVector,
    update: TaskVector,
    surface_identity: str,
    context: MergeExecutionContext,
    protected_mask: DareMask | None = None,
    partner_mask: DareMask | None = None,
    protected_truncations: Mapping[str, Truncation] | None = None,
    partner_truncations: Mapping[str, Truncation] | None = None,
) -> MergeResult:
    """Derive every scientific field from the inputs and the operation, then issue [F5]."""
    protected_identity = protected.content_identity()
    partner_identity = partner.content_identity()
    protected_truncation_identity = (
        truncation_set_identity(protected_truncations) if protected_truncations else None
    )
    partner_truncation_identity = (
        truncation_set_identity(partner_truncations) if partner_truncations else None
    )

    payload: dict[str, JSONValue] = {
        "schema": FAMILY_SCHEMA,
        "operator": operator,
        "operator_version": context.operator_version,
        "execution_context_sha256": context.identity(),
        "protected_update_sha256": protected_identity,
        "partner_update_sha256": partner_identity,
        "parameter_surface_sha256": surface_identity,
        "alpha": spec.alpha,
        "arithmetic_dtype": context.arithmetic_dtype,
        "result_update_sha256": update.content_identity(),
        "drop_probability": spec.drop_probability,
        "dare_merge_seed": spec.dare_merge_seed,
        "mask_scheme": context.mask_scheme if protected_mask is not None else None,
        "protected_mask_sha256": (
            protected_mask.identity() if protected_mask is not None else None
        ),
        "partner_mask_sha256": partner_mask.identity() if partner_mask is not None else None,
        "retained_rank": spec.retained_rank,
        "protected_truncation_sha256": protected_truncation_identity,
        "partner_truncation_sha256": partner_truncation_identity,
    }

    result = object.__new__(MergeResult)
    object.__setattr__(result, "_operator", operator)
    object.__setattr__(result, "_descendant_id", spec.descendant_id)
    object.__setattr__(result, "_partner_label", spec.partner_id)
    object.__setattr__(result, "_alpha", float(spec.alpha))
    object.__setattr__(result, "_update", update)
    object.__setattr__(result, "_protected_identity", protected_identity)
    object.__setattr__(result, "_partner_identity", partner_identity)
    object.__setattr__(result, "_protected_provenance", protected.provenance_class)
    object.__setattr__(result, "_partner_provenance", partner.provenance_class)
    object.__setattr__(result, "_surface_identity", surface_identity)
    object.__setattr__(result, "_context", context)
    object.__setattr__(result, "_drop_probability", spec.drop_probability)
    object.__setattr__(result, "_dare_merge_seed", spec.dare_merge_seed)
    object.__setattr__(result, "_protected_mask", protected_mask)
    object.__setattr__(result, "_partner_mask", partner_mask)
    object.__setattr__(result, "_retained_rank", spec.retained_rank)
    object.__setattr__(result, "_protected_truncations", protected_truncations)
    object.__setattr__(result, "_partner_truncations", partner_truncations)
    object.__setattr__(result, "_protected_truncation_identity", protected_truncation_identity)
    object.__setattr__(result, "_partner_truncation_identity", partner_truncation_identity)
    object.__setattr__(result, "_identity", sha256_canonical(payload))
    return result


def build_linear_descendant(
    *, protected: TaskVector, partner: TaskVector, spec: MergeSpec, context: MergeExecutionContext
) -> MergeResult:
    """The ONLY way to obtain a LINEAR result [AUTH: 00 §10.1]."""
    if spec.operator != LINEAR:
        raise MergeFamilyError(f"{spec.descendant_id}: this factory builds {LINEAR} only")
    surface = require_compatible([protected, partner], what="linear merge")
    update = linear_merge(
        protected, partner, alpha=spec.alpha, context=context, origin=spec.descendant_id
    )
    return _issue_result(
        operator=LINEAR,
        spec=spec,
        protected=protected,
        partner=partner,
        update=update,
        surface_identity=surface,
        context=context,
    )


def build_dare_descendant(
    *, protected: TaskVector, partner: TaskVector, spec: MergeSpec, context: MergeExecutionContext
) -> MergeResult:
    """The ONLY way to obtain a DARE result [AUTH: 00 §10.2].

    Masks are drawn here from the descendant's one merge seed, and their hashes are computed
    from the actual mask bytes. No caller supplies a mask or a mask hash, so evidence drawn
    under seed 101 cannot be recorded against seed 202.
    """
    if spec.operator != DARE:
        raise MergeFamilyError(f"{spec.descendant_id}: this factory builds {DARE} only")
    assert spec.drop_probability is not None and spec.dare_merge_seed is not None
    surface = require_compatible([protected, partner], what="DARE merge")
    update, protected_transform, partner_transform = dare_merge(
        protected,
        partner,
        alpha=spec.alpha,
        drop_probability=spec.drop_probability,
        merge_seed=spec.dare_merge_seed,
        context=context,
        origin=spec.descendant_id,
    )
    return _issue_result(
        operator=DARE,
        spec=spec,
        protected=protected,
        partner=partner,
        update=update,
        surface_identity=surface,
        context=context,
        protected_mask=protected_transform.mask,
        partner_mask=partner_transform.mask,
    )


def build_o3_descendant(
    *, protected: TaskVector, partner: TaskVector, spec: MergeSpec, context: MergeExecutionContext
) -> MergeResult:
    """The ONLY way to obtain an O3 result [AUTH: 00 §10.3]."""
    if spec.operator != SVD_TRUNC:
        raise MergeFamilyError(f"{spec.descendant_id}: this factory builds {SVD_TRUNC} only")
    assert spec.retained_rank is not None
    surface = require_compatible([protected, partner], what="SVD-trunc merge")
    update, protected_parts, partner_parts = svd_trunc_merge(
        protected,
        partner,
        alpha=spec.alpha,
        rank=spec.retained_rank,
        context=context,
        origin=spec.descendant_id,
    )
    return _issue_result(
        operator=SVD_TRUNC,
        spec=spec,
        protected=protected,
        partner=partner,
        update=update,
        surface_identity=surface,
        context=context,
        protected_truncations=protected_parts,
        partner_truncations=partner_parts,
    )


#: Operator -> its single factory. Dispatch on the declared operator, never on which optional
#: arguments happened to be supplied.
_FACTORIES: Final = {
    LINEAR: build_linear_descendant,
    DARE: build_dare_descendant,
    SVD_TRUNC: build_o3_descendant,
}


def build_descendant(
    *, protected: TaskVector, partner: TaskVector, spec: MergeSpec, context: MergeExecutionContext
) -> MergeResult:
    """Route to the operator's own factory. Convenience only; it adds no behaviour."""
    factory = _FACTORIES[require_supported_operator(spec.operator)]
    return factory(protected=protected, partner=partner, spec=spec, context=context)


class ReleaseFamily:
    """C_1..C_k built from ONE protected constituent A [AUTH: 00 §3.4, §11].

    Opaque and factory-issued. The protected identity is DERIVED from the first issued result
    and then required to match every other; there is no parameter through which a caller can
    claim A1 for a descendant built from A2.
    """

    __slots__ = ("_context", "_descendants", "_protected_identity", "_surface_identity")

    _protected_identity: str
    _surface_identity: str
    _context: MergeExecutionContext
    _descendants: tuple[MergeResult, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("ReleaseFamily is factory-issued; use build_release_family()")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a release family is immutable")

    @property
    def protected_identity(self) -> str:
        return self._protected_identity

    @property
    def surface_identity(self) -> str:
        return self._surface_identity

    @property
    def context(self) -> MergeExecutionContext:
        return self._context

    @property
    def descendants(self) -> tuple[MergeResult, ...]:
        return self._descendants

    @property
    def k(self) -> int:
        return len(self._descendants)

    def by_id(self, descendant_id: str) -> MergeResult:
        for result in self._descendants:
            if result.descendant_id == descendant_id:
                return result
        raise MergeFamilyError(f"{descendant_id!r} is not in this family")

    def mask_identities(self) -> dict[str, tuple[str, str]]:
        return {
            result.descendant_id: (
                result.protected_mask.identity(),
                result.partner_mask.identity(),
            )
            for result in self._descendants
            if result.protected_mask is not None and result.partner_mask is not None
        }

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": FAMILY_SCHEMA,
            "provenance_class": NON_EVIDENTIARY,
            "protected_update_sha256": self._protected_identity,
            "parameter_surface_sha256": self._surface_identity,
            "execution_context_sha256": self._context.identity(),
            "k": self.k,
            "descendants": [result.as_dict() for result in self._descendants],
        }


def build_release_family(
    *,
    protected: TaskVector,
    partners: Mapping[str, TaskVector],
    specs: Sequence[MergeSpec],
    permitted_k: Sequence[int],
    context: MergeExecutionContext,
) -> ReleaseFamily:
    """Build C_1..C_k from one A [AUTH: 00 §3.4, §11].

    `specs` carries the (partner, coefficient, descendant-id) association explicitly, so
    partners are never sorted independently of their coefficients. Reordering complete spec
    records changes only listing order; each descendant's bytes come from its own record.
    """
    if not specs:
        raise MergeFamilyError("a release family needs at least one descendant spec")
    if len(specs) not in set(permitted_k):
        raise MergeFamilyError(
            f"descendant count k = {len(specs)} is not a registered value"
            f" {sorted(set(permitted_k))} [AUTH: 00 §11]"
        )
    unknown = sorted({spec.partner_id for spec in specs} - set(partners))
    if unknown:
        raise MergeFamilyError(f"no task vector supplied for partner(s): {unknown}")

    results = tuple(
        build_descendant(
            protected=protected, partner=partners[spec.partner_id], spec=spec, context=context
        )
        for spec in specs
    )

    # Derived from the first ISSUED result, then required to match every other. There is no
    # parameter through which a caller can claim A1 for a descendant built from A2.
    protected_identity = results[0].protected_identity
    surface_identity = results[0].surface_identity
    context_identity = results[0].execution_context_sha256

    ids = [result.descendant_id for result in results]
    if len(set(ids)) != len(ids):
        repeated = sorted({name for name in ids if ids.count(name) > 1})
        raise MergeFamilyError(f"duplicate descendant id(s): {repeated}")
    for result in results:
        if result.protected_identity != protected_identity:
            raise MergeFamilyError(
                f"{result.descendant_id} was built from a different protected constituent;"
                " every descendant contains the same trained draw of A and A is not"
                " re-trained per descendant [AUTH: 00 §3.4]"
            )
        if result.surface_identity != surface_identity:
            raise MergeFamilyError(f"{result.descendant_id} adapts a different parameter surface")
        if result.execution_context_sha256 != context_identity:
            raise MergeFamilyError(
                f"{result.descendant_id} ran under a different execution context"
            )

    family = object.__new__(ReleaseFamily)
    object.__setattr__(family, "_protected_identity", protected_identity)
    object.__setattr__(family, "_surface_identity", surface_identity)
    object.__setattr__(family, "_context", context)
    object.__setattr__(family, "_descendants", results)
    return family
