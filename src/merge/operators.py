"""The three frozen S07 merge operators [AUTH: 00 §10.1-§10.3; 01 §39 S07].

    O1  LINEAR / TASK-ARITHMETIC   information-preserving baseline
    O2  DARE                       coordinate drop then survivor rescale
    O3  SVD-TRUNC-MERGE            pre-registered, conditional

Operator identity is explicit and carried on every result. There is deliberately no function
called `merge` that switches operator on an optional argument: a LINEAR artifact and a DARE
artifact of the same inputs are different scientific objects, and the type system should not
let one be produced where the other was requested.

**Availability is not authorisation.** O3 exists here because 00 §10.3 pre-registers it
before P0-D runs, so the project cannot pick a replacement operator after learning whether
DARE succeeded. Nothing in this module may select p*, select s*, mark either gate passed, or
name a primary lossy operator — those are later empirical decisions and their config fields
are `REQUIRED_NOT_CALIBRATED` precisely so that reading one fails closed.

TIES is not implemented and is refused by the factory [AUTH: 00 §10.4; 01 §39 S07].

Every operator takes a resolved `MergeExecutionContext`, so the operator version, mask scheme,
SVD backend and arithmetic dtype that a run records are the ones it actually executed. Merge
arithmetic runs at the context's arithmetic dtype — float32 for scientific artifacts
[AUTH: 01 §10; 00 §25 P0-C1] — while the SVD numerical workspace may run wider and is cast
back deterministically before anything is hashed or composed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Final

import numpy as np

from src.merge.context import MergeExecutionContext
from src.merge.masks import PARTNER_ROLE, PROTECTED_ROLE, DareMask, derive_mask
from src.merge.updates import (
    FixtureTaskVector,
    Matrix,
    TaskVector,
    combine,
    fixture_induced_update,
    require_compatible,
)

#: The version this source implements. The version a run *records* comes from the resolved
#: context; `MergeExecutionContext` refuses a config naming anything this code cannot execute.
OPERATOR_VERSION: Final = "s07.operators.v1"

LINEAR: Final = "O1_LINEAR_TASK_ARITHMETIC"
DARE: Final = "O2_DARE"
SVD_TRUNC: Final = "O3_SVD_TRUNC_MERGE"
OPERATORS: Final[tuple[str, ...]] = (LINEAR, DARE, SVD_TRUNC)

#: Deferred to P2, and refused here rather than stubbed [AUTH: 00 §10.4].
TIES: Final = "TIES"
DEFERRED_OPERATORS: Final[tuple[str, ...]] = (TIES, "SLERP", "O4_QUANTIZATION")

#: O3 is implemented and pre-registered; it is not activated.
O3_STATUS: Final = "PRE_REGISTERED_CONDITIONAL"
#: The two empirical gates. S07 never moves either off NOT_RUN [AUTH: 00 §24, §24A].
DARE_GATE_STATUS: Final = "NOT_RUN"
O3_GATE_STATUS: Final = "NOT_RUN"


class OperatorError(ValueError):
    """A merge operator cannot be applied as specified."""


class DeferredOperatorError(OperatorError):
    """An operator that is not part of P0/P1 was requested."""


# ----------------------------------------------------------------------------------------
# O1 — linear / task arithmetic
# ----------------------------------------------------------------------------------------


def linear_merge(
    protected: TaskVector,
    partner: TaskVector,
    *,
    alpha: float,
    context: MergeExecutionContext,
    origin: str = "C_linear",
) -> FixtureTaskVector:
    """ΔW_C = α·τ_A + (1-α)·τ_B, in base-relative induced-update space [AUTH: 00 §10.1].

    The descendant is conceptually C = θ0 + ΔW_C; θ0 never enters, because the operator is
    defined on task vectors and adding a base back would make the object a checkpoint rather
    than an update.

    α is not clamped to [0, 1]: 00 §10.1 is task arithmetic, and the generic operator should
    not be narrower than the authority. Production experiment configs supply only registered
    coefficient schedules; that restriction belongs to S09's registry, not to the operator.
    """
    if not np.isfinite(alpha):
        raise OperatorError(f"merge coefficient {alpha!r} is not finite")
    require_compatible([protected, partner], what="linear merge")
    return combine(
        [(float(alpha), protected), (1.0 - float(alpha), partner)],
        context=context,
        origin=origin,
    )


# ----------------------------------------------------------------------------------------
# O2 — DARE
# ----------------------------------------------------------------------------------------


class DareTransform:
    """One task vector after drop-and-rescale, with the mask that produced it.

    An internal carrier between `dare_transform` and `dare_merge`. Opaque like the evidence
    it holds, so no intermediate step can swap a mask for another before a result is issued.
    """

    __slots__ = ("_mask", "_operator_version", "_role", "_source_identity", "_vector")

    _vector: FixtureTaskVector
    _mask: DareMask
    _source_identity: str
    _role: str
    _operator_version: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("DareTransform is factory-issued; use dare_transform()")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a DARE transform is immutable")

    @property
    def vector(self) -> FixtureTaskVector:
        return self._vector

    @property
    def mask(self) -> DareMask:
        return self._mask

    @property
    def source_identity(self) -> str:
        return self._source_identity

    @property
    def role(self) -> str:
        return self._role

    @property
    def operator_version(self) -> str:
        return self._operator_version

    @property
    def drop_probability(self) -> float:
        return self._mask.drop_probability

    def as_dict(self) -> dict[str, object]:
        return {
            "operator": DARE,
            "operator_version": self._operator_version,
            "source_identity": self._source_identity,
            "constituent_role": self._role,
            "transformed_identity": self._vector.content_identity(),
            **self._mask.as_dict(),
        }


def _issue_transform(
    *,
    vector: FixtureTaskVector,
    mask: DareMask,
    source_identity: str,
    role: str,
    operator_version: str,
) -> DareTransform:
    transform = object.__new__(DareTransform)
    object.__setattr__(transform, "_vector", vector)
    object.__setattr__(transform, "_mask", mask)
    object.__setattr__(transform, "_source_identity", source_identity)
    object.__setattr__(transform, "_role", role)
    object.__setattr__(transform, "_operator_version", operator_version)
    return transform


def dare_transform(
    vector: TaskVector,
    *,
    drop_probability: float,
    merge_seed: int,
    role: str,
    context: MergeExecutionContext,
    origin: str = "dare",
) -> DareTransform:
    """Drop coordinates with probability p, then rescale survivors by 1/(1-p).

    The order is fixed and is the whole operator: dropping alone would shrink the expected
    task vector, and the survivor rescale is what keeps E[τ_DARE] = τ. p = 0 retains every
    coordinate and rescales by exactly 1, so it is the identity transformation; p = 1 is
    refused because it would erase the vector entirely [AUTH: 00 §10.2].
    """
    if not 0.0 <= drop_probability < 1.0:
        raise OperatorError(
            f"DARE drop probability {drop_probability!r} is outside [0, 1); p = 1 is refused"
        )
    mask = derive_mask(
        vector,
        drop_probability=drop_probability,
        merge_seed=merge_seed,
        role=role,
        context=context,
    )
    dtype = context.arithmetic
    scale = np.asarray(1.0 / (1.0 - float(drop_probability)), dtype=dtype)
    zero = np.asarray(0.0, dtype=dtype)
    transformed: dict[str, Matrix] = {
        name: np.asarray(
            np.where(
                mask.keep_array(name).reshape(vector.shapes[name]),
                vector[name] * scale,
                zero,
            ),
            dtype=dtype,
        )
        for name in vector.names
    }
    return _issue_transform(
        vector=fixture_induced_update(transformed, context=context, origin=origin),
        mask=mask,
        source_identity=vector.content_identity(),
        role=role,
        operator_version=context.operator_version,
    )


def dare_merge(
    protected: TaskVector,
    partner: TaskVector,
    *,
    alpha: float,
    drop_probability: float,
    merge_seed: int,
    context: MergeExecutionContext,
    origin: str = "C_dare",
) -> tuple[FixtureTaskVector, DareTransform, DareTransform]:
    """Transform both constituents, then compose linearly [AUTH: 00 §10.2].

    One merge-level seed per descendant, split into independent role substreams: they are
    different task vectors, and one shared *draw* would correlate which coordinates survive
    on the two sides, while two independently chosen seeds would be a pairing nobody
    declared [architect adjudication]. The composed result is returned with both transforms,
    so the artifact is reconstructable from input identities, α, p, the merge seed, the two
    mask identities and the execution context.
    """
    require_compatible([protected, partner], what="DARE merge")
    if not np.isfinite(alpha):
        raise OperatorError(f"merge coefficient {alpha!r} is not finite")
    protected_transform = dare_transform(
        protected,
        drop_probability=drop_probability,
        merge_seed=merge_seed,
        role=PROTECTED_ROLE,
        context=context,
        origin="dare_A",
    )
    partner_transform = dare_transform(
        partner,
        drop_probability=drop_probability,
        merge_seed=merge_seed,
        role=PARTNER_ROLE,
        context=context,
        origin="dare_B",
    )
    merged = combine(
        [
            (float(alpha), protected_transform.vector),
            (1.0 - float(alpha), partner_transform.vector),
        ],
        context=context,
        origin=origin,
    )
    return merged, protected_transform, partner_transform


# ----------------------------------------------------------------------------------------
# O3 — SVD-TRUNC-MERGE (pre-registered, conditional)
# ----------------------------------------------------------------------------------------


class Truncation:
    """One matrix's rank-s truncation and the spectrum it came from.

    Opaque and factory-issued: truncations are computed by `truncate` and may not be
    assembled by a caller, because the O3 information floor is defined on the protected
    constituent's own truncation [AUTH: 00 §24A.1].
    """

    __slots__ = (
        "_arithmetic_dtype",
        "_bytes",
        "_identity",
        "_requested_rank",
        "_retained_rank",
        "_shape",
        "_singular",
    )

    _bytes: bytes
    _shape: tuple[int, int]
    _singular: bytes
    _retained_rank: int
    _requested_rank: int
    _arithmetic_dtype: str
    _identity: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Truncation is factory-issued; use truncate()")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a truncation is immutable")

    @property
    def matrix(self) -> Matrix:
        """A read-only view of the float32 artifact bytes."""
        code = "<f4" if self._arithmetic_dtype == "float32" else "<f8"
        return np.frombuffer(self._bytes, dtype=code).reshape(self._shape)

    @property
    def singular_values(self) -> Matrix:
        return np.frombuffer(self._singular, dtype="<f8")

    @property
    def retained_rank(self) -> int:
        return self._retained_rank

    @property
    def requested_rank(self) -> int:
        return self._requested_rank

    @property
    def arithmetic_dtype(self) -> str:
        return self._arithmetic_dtype

    def identity(self) -> str:
        """Hash of the truncated artifact bytes and the rank that produced them."""
        return self._identity

    @property
    def discarded_energy(self) -> float:
        return float(np.sum(self.singular_values[self._retained_rank :] ** 2))

    @property
    def total_energy(self) -> float:
        return float(np.sum(self.singular_values**2))


def _issue_truncation(
    *,
    matrix: Matrix,
    singular: Matrix,
    retained_rank: int,
    requested_rank: int,
    context: MergeExecutionContext,
) -> Truncation:
    payload = np.ascontiguousarray(matrix, dtype=context.byte_order_code).tobytes("C")
    spectrum = np.ascontiguousarray(singular, dtype="<f8").tobytes("C")
    digest = hashlib.sha256(
        f"s07.truncation.v1|{context.operator_version}|{context.arithmetic_dtype}"
        f"|s={requested_rank}|kept={retained_rank}".encode()
    )
    digest.update(payload)

    truncation = object.__new__(Truncation)
    object.__setattr__(truncation, "_bytes", payload)
    object.__setattr__(truncation, "_shape", (int(matrix.shape[0]), int(matrix.shape[1])))
    object.__setattr__(truncation, "_singular", spectrum)
    object.__setattr__(truncation, "_retained_rank", retained_rank)
    object.__setattr__(truncation, "_requested_rank", requested_rank)
    object.__setattr__(truncation, "_arithmetic_dtype", context.arithmetic_dtype)
    object.__setattr__(truncation, "_identity", digest.hexdigest())
    return truncation


def truncation_set_identity(parts: Mapping[str, Truncation]) -> str:
    """One identity over a whole surface's truncations, in canonical name order."""
    digest = hashlib.sha256(b"s07.truncation-set.v1")
    for name in sorted(parts):
        digest.update(name.encode("utf-8"))
        digest.update(parts[name].identity().encode("ascii"))
    return digest.hexdigest()


def truncate(matrix: Matrix, *, rank: int, context: MergeExecutionContext) -> Truncation:
    """T_s(W) = U[:, :s] Σ[:s] V[:, :s]^T [AUTH: 00 §10.3].

    The reconstruction is returned, never the factors: U and V carry a sign (and, at repeated
    singular values, a basis) ambiguity that no backend fixes, while T_s(W) itself is
    algebraically determined. Hashing a reconstruction is stable; hashing U would not be
    [AUTH: 01 §12].

    The decomposition runs in the context's frozen workspace dtype for numerical stability
    and the reconstruction is then cast deterministically to the arithmetic dtype, so the
    artifact bytes are float32 while the factorisation itself is not fighting float32
    conditioning [AUTH: 01 §10]. Singular values are kept at workspace precision because they
    feed diagnostics, not artifact bytes.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise OperatorError(f"retained rank {rank!r} must be a positive integer")
    if matrix.ndim != 2:
        raise OperatorError("truncation applies to a 2-D induced update")
    array = np.asarray(matrix, dtype=context.workspace)
    if not np.all(np.isfinite(array)):
        raise OperatorError("cannot truncate a matrix carrying NaN or Inf")
    left, singular, right = np.linalg.svd(array, full_matrices=False)
    retained = min(rank, int(singular.size))
    reconstructed = (left[:, :retained] * singular[:retained]) @ right[:retained, :]
    return _issue_truncation(
        matrix=np.ascontiguousarray(reconstructed, dtype=context.arithmetic),
        singular=np.asarray(singular, dtype=context.diagnostic),
        retained_rank=retained,
        requested_rank=rank,
        context=context,
    )


def truncate_vector(
    vector: TaskVector, *, rank: int, context: MergeExecutionContext, origin: str = "T_s"
) -> FixtureTaskVector:
    """Truncate every target matrix of a task vector at the same rank."""
    return fixture_induced_update(
        {name: truncate(vector[name], rank=rank, context=context).matrix for name in vector.names},
        context=context,
        origin=origin,
    )


def truncations(
    vector: TaskVector, *, rank: int, context: MergeExecutionContext
) -> dict[str, Truncation]:
    """Per-matrix truncations, keeping the spectra so diagnostics need no second SVD."""
    return {name: truncate(vector[name], rank=rank, context=context) for name in vector.names}


def svd_trunc_merge(
    protected: TaskVector,
    partner: TaskVector,
    *,
    alpha: float,
    rank: int,
    context: MergeExecutionContext,
    origin: str = "C_svd",
) -> tuple[FixtureTaskVector, dict[str, Truncation], dict[str, Truncation]]:
    """ΔW_C^SVD = α·T_s(τ_A) + (1-α)·T_s(τ_B) [AUTH: 00 §10.3].

    Truncation happens per constituent BEFORE composition, which is what makes the discarded
    singular directions a property of each constituent rather than of the sum.
    """
    require_compatible([protected, partner], what="SVD-trunc merge")
    if not np.isfinite(alpha):
        raise OperatorError(f"merge coefficient {alpha!r} is not finite")
    protected_parts = truncations(protected, rank=rank, context=context)
    partner_parts = truncations(partner, rank=rank, context=context)
    merged = combine(
        [
            (
                float(alpha),
                fixture_induced_update(
                    {n: p.matrix for n, p in protected_parts.items()},
                    context=context,
                    origin="T_s(A)",
                ),
            ),
            (
                1.0 - float(alpha),
                fixture_induced_update(
                    {n: p.matrix for n, p in partner_parts.items()},
                    context=context,
                    origin="T_s(B)",
                ),
            ),
        ],
        context=context,
        origin=origin,
    )
    return merged, protected_parts, partner_parts


# ----------------------------------------------------------------------------------------
# factory
# ----------------------------------------------------------------------------------------


def require_supported_operator(name: str) -> str:
    """The single gate on operator identity. TIES is refused, not stubbed."""
    if name in DEFERRED_OPERATORS:
        raise DeferredOperatorError(
            f"operator {name!r} is deferred and is not implemented at S07; it may enter only"
            " under the expansion rules, and no placeholder stands in for it"
            " [AUTH: 00 §10.4, §10.5; 01 §39 S07]"
        )
    if name not in OPERATORS:
        raise OperatorError(f"{name!r} is not a declared S07 operator; declared: {OPERATORS}")
    return name


def operator_status(context: MergeExecutionContext) -> dict[str, object]:
    """What S07 may and may not assert about the operator axis [AUTH: 00 §24, §24A.7].

    Version and backend come from the resolved context, not from source constants, so a
    status report cannot describe a configuration the run did not execute.
    """
    return {
        "operator_version": context.operator_version,
        "implemented": list(OPERATORS),
        "deferred": list(DEFERRED_OPERATORS),
        "O3_STATUS": O3_STATUS,
        "DARE_GATE_STATUS": DARE_GATE_STATUS,
        "O3_GATE_STATUS": O3_GATE_STATUS,
        "svd_backend": context.svd_backend,
        "arithmetic_dtype": context.arithmetic_dtype,
        "execution_context_sha256": context.identity(),
    }
