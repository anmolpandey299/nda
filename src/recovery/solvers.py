"""The S08 recovery methods [AUTH: 00 §25; 01 §39 S08].

    C1    KNOWN PARTNER + KNOWN COEFFICIENT EXACT LINEAR INVERSION      oracle control
    C2    SPECTRAL_DETUNING_CORE_FIXED_RANK_REPRODUCTION                reproduction baseline
    C2B   KNOWN-VARYING-ALPHA RESCALING + THE SAME C2 SOLVER
    O3    CONDITIONAL SHARED-SOURCE RECOVERY OF T_s(A)

Every solver takes an `ObservedLineage` and nothing else, so what it can see is decided by the
observation factory rather than by discipline here. A `RecoveryResult` is opaque, immutable,
factory-issued and carries **no** ground truth: no protected identity, no hidden partner
identity, no recovery error, no success label. Truth is joined only in `src.recovery.evaluation`,
which no solver imports.

C2 is a reproduction of the published Spectral DeTuning shared-source / low-rank-residual
baseline (ICML 2024, Algorithm 1 style), implemented independently from the mathematical
description. It is an occupied-regime control, not a novel algorithm, and it makes no claim to
a new common-source method or low-rank inverse theorem.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from src.provenance.hashing import JSONValue, sha256_canonical
from src.recovery.context import RecoveryExecutionContext
from src.recovery.observations import (
    LINEAR,
    METHOD_GATED,
    STRUCTURAL_NA,
    STRUCTURAL_NOT_AUTHORIZED,
    SVD_TRUNC,
    Z_HIGH,
    Z_INCOMPLETE_FIXED,
    Z_INCOMPLETE_VARYING_KNOWN,
    Matrix,
    NotAuthorizedError,
    ObservedLineage,
    StructuralNAError,
    TensorEntry,
)

RESULT_SCHEMA: Final = "s08.recovery-result.v2"

#: A recovery never attributes a scientific run [AUTH: 01 §14]. A result issued under a
#: fixture context says so, so a planted-truth solve can never be read as the accepted method.
NON_EVIDENTIARY: Final = "NON_EVIDENTIARY_S08_RECOVERY"
NON_EVIDENTIARY_FIXTURE: Final = "NON_EVIDENTIARY_S08_RECOVERY_FIXTURE"

#: What a method is trying to recover. Derived from the factory, never from a caller label.
ORIGINAL_PROTECTED_CONSTITUENT: Final = "ORIGINAL_PROTECTED_CONSTITUENT"
TRUNCATED_PROTECTED_SOURCE: Final = "TRUNCATED_PROTECTED_SOURCE"

C1_METHOD: Final = "C1_EXACT_LINEAR_ORACLE"
C2_SPECTRAL_FIXED: Final = "C2_SPECTRAL_FIXED"
C2_NAIVE_SHARED_MEAN: Final = "C2_NAIVE_SHARED_MEAN"
C2B_RESCALED: Final = "C2B_RESCALED_SPECTRAL_REPRODUCTION"
NAIVE_UNRESCALED_FIXED_ALPHA: Final = "NAIVE_UNRESCALED_FIXED_ALPHA"
O3_SPECTRAL_FIXED: Final = "O3_SPECTRAL_FIXED_ALPHA"
O3_C2B_RESCALED: Final = "O3_C2B_RESCALED"

METHODS: Final[tuple[str, ...]] = (
    C1_METHOD,
    C2_SPECTRAL_FIXED,
    C2_NAIVE_SHARED_MEAN,
    C2B_RESCALED,
    NAIVE_UNRESCALED_FIXED_ALPHA,
    O3_SPECTRAL_FIXED,
    O3_C2B_RESCALED,
)

#: C1 is an implementation oracle; C2 reproduces an occupied regime. Neither is novelty.
METHOD_ROLE: Final[Mapping[str, str]] = {
    C1_METHOD: "ORACLE_CONTROL",
    C2_SPECTRAL_FIXED: "OCCUPIED_REGIME_REPRODUCTION",
    C2_NAIVE_SHARED_MEAN: "NAIVE_COMPARATOR",
    C2B_RESCALED: "OCCUPIED_REGIME_REPRODUCTION",
    NAIVE_UNRESCALED_FIXED_ALPHA: "NAIVE_COMPARATOR",
    O3_SPECTRAL_FIXED: "OCCUPIED_REGIME_REPRODUCTION",
    O3_C2B_RESCALED: "OCCUPIED_REGIME_REPRODUCTION",
}


class RecoveryError(ValueError):
    """A recovery cannot be performed as specified."""


class RecoveryInputError(RecoveryError):
    """A public recovery entry point was handed something other than an issued observation."""


# ----------------------------------------------------------------------------------------
# the opaque result
# ----------------------------------------------------------------------------------------


class RecoveryResult:
    """One recovered induced update and the attack-time facts that produced it.

    Opaque and factory-issued: `__init__` raises, `__setattr__` raises, and this is not a
    dataclass, so no `dataclasses.replace` can move the method, the observation identity or the
    recovered bytes. It deliberately carries no ground truth — a solver never holds any.
    """

    __slots__ = (
        "_context",
        "_entries",
        "_identity",
        "_iterations",
        "_method",
        "_method_parameters",
        "_observation_identity",
        "_operator",
        "_public_alphas",
        "_regime",
        "_residual_rank",
        "_retained_rank",
        "_target_class",
    )

    _method: str
    _target_class: str
    _observation_identity: str
    _regime: str
    _operator: str
    _entries: tuple[TensorEntry, ...]
    _method_parameters: tuple[tuple[str, JSONValue], ...]
    _context: RecoveryExecutionContext
    _residual_rank: int | None
    _iterations: int | None
    _retained_rank: int | None
    _public_alphas: tuple[float, ...]
    _identity: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "RecoveryResult is factory-issued; use recover_c1, recover_c2_fixed_alpha,"
            " recover_c2b_varying_alpha or their naive comparators. Method identity and"
            " recovered bytes derive from the solver that ran [AUTH: 00 §25]"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a recovery result is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a recovery result is immutable")

    # ---------------------------------------------------------------- projections
    @property
    def provenance_class(self) -> str:
        """Derived and permanent. S09 binds run-level provenance; this cannot self-promote.

        A result solved under a fixture context is labelled as such, so a planted-truth solve
        can never be mistaken for the accepted method [AUTH: 01 §14].
        """
        return NON_EVIDENTIARY if self._context.is_scientific else NON_EVIDENTIARY_FIXTURE

    @property
    def method(self) -> str:
        return self._method

    @property
    def method_role(self) -> str:
        return METHOD_ROLE[self._method]

    @property
    def target_class(self) -> str:
        """What was being recovered: the original constituent, or T_s(A) for O3."""
        return self._target_class

    @property
    def observation_identity(self) -> str:
        return self._observation_identity

    @property
    def regime(self) -> str:
        return self._regime

    @property
    def operator(self) -> str:
        return self._operator

    @property
    def context(self) -> RecoveryExecutionContext:
        return self._context

    @property
    def residual_rank(self) -> int | None:
        return self._residual_rank

    @property
    def iterations(self) -> int | None:
        return self._iterations

    @property
    def retained_rank(self) -> int | None:
        return self._retained_rank

    @property
    def method_parameters(self) -> dict[str, JSONValue]:
        """Method-specific scientific parameters, freshly built. Bound into the identity."""
        return dict(self._method_parameters)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry[0] for entry in self._entries)

    @property
    def shapes(self) -> dict[str, tuple[int, int]]:
        """A freshly built mapping. Mutating it cannot reach the authoritative entries."""
        return {entry[0]: entry[1] for entry in self._entries}

    @property
    def dtype(self) -> str:
        return self._context.arithmetic_dtype

    def _entry(self, name: str) -> TensorEntry:
        for entry in self._entries:
            if entry[0] == name:
                return entry
        raise RecoveryError(f"{name!r} is not on this parameter surface")

    def __getitem__(self, name: str) -> Matrix:
        """A fresh read-only ndarray over the immutable recovered bytes."""
        _, shape, _, payload = self._entry(name)
        view = np.frombuffer(payload, dtype=self._context.byte_order_code)
        return view.reshape(shape)

    def copy_of(self, name: str) -> Matrix:
        """A writable copy, provably detached: it owns its own buffer."""
        return np.array(self[name], copy=True)

    def surface_identity(self) -> str:
        digest = hashlib.sha256(f"{RESULT_SCHEMA}|surface".encode())
        for name, (rows, columns), dtype, _ in self._entries:
            digest.update(f"|{name}:{rows}x{columns}:{dtype}".encode())
        return digest.hexdigest()

    def content_identity(self) -> str:
        digest = hashlib.sha256(f"{RESULT_SCHEMA}|content|{self.dtype}".encode())
        for name, _, _, payload in self._entries:
            digest.update(name.encode("utf-8"))
            digest.update(payload)
        return digest.hexdigest()

    def identity(self) -> str:
        return self._identity

    def as_dict(self) -> dict[str, JSONValue]:
        """The S09 handoff: attack-time facts only, no truth and no privacy result."""
        return {
            "schema": RESULT_SCHEMA,
            "provenance_class": self.provenance_class,
            "recovery_profile": self._context.profile,
            "method": self._method,
            "method_role": self.method_role,
            "method_parameters": dict(self._method_parameters),
            "recovery_version": self._context.recovery_version,
            "c2_method_version": self._context.c2_method_version,
            "target_class": self._target_class,
            "observation_sha256": self._observation_identity,
            "lineage_regime": self._regime,
            "operator": self._operator,
            "recovery_context_sha256": self._context.identity(),
            "parameter_surface_sha256": self.surface_identity(),
            "recovered_update_sha256": self.content_identity(),
            "arithmetic_dtype": self.dtype,
            "residual_rank": self._residual_rank,
            "iterations": self._iterations,
            "retained_rank": self._retained_rank,
            "public_alphas": list(self._public_alphas),
            "recovery_result_sha256": self._identity,
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"RecoveryResult({self._method}, {self._identity[:12]})"


def _issue_result(
    *,
    method: str,
    target_class: str,
    tensors: Mapping[str, Matrix],
    observation: ObservedLineage,
    context: RecoveryExecutionContext,
    residual_rank: int | None,
    iterations: int | None,
    method_parameters: Mapping[str, JSONValue] | None = None,
) -> RecoveryResult:
    """Freeze the recovered tensors and derive every scientific field, then issue.

    `method_parameters` carries anything a method was *told* rather than read off the
    observation — today only the naive comparator's predeclared coefficient. It is bound into
    the result identity and the serialized provenance, so two runs of the same method under
    different predeclared parameters are distinguishable before their bytes are compared
    [AUTH: 00 §25; 01 §16].
    """
    _require_observed_lineage(observation)
    entries: list[TensorEntry] = []
    for name in sorted(tensors):
        cast = np.asarray(tensors[name], dtype=context.arithmetic)
        if not np.all(np.isfinite(cast)):
            raise RecoveryError(f"{name}: the recovered update carries NaN or Inf")
        entries.append(
            (
                name,
                (int(cast.shape[0]), int(cast.shape[1])),
                context.arithmetic_dtype,
                np.ascontiguousarray(cast, dtype=context.byte_order_code).tobytes("C"),
            )
        )
    parameters = tuple(sorted((str(k), v) for k, v in dict(method_parameters or {}).items()))

    result = object.__new__(RecoveryResult)
    object.__setattr__(result, "_method", method)
    object.__setattr__(result, "_target_class", target_class)
    object.__setattr__(result, "_observation_identity", observation.identity())
    object.__setattr__(result, "_regime", observation.regime)
    object.__setattr__(result, "_operator", observation.operator)
    object.__setattr__(result, "_entries", tuple(entries))
    object.__setattr__(result, "_method_parameters", parameters)
    object.__setattr__(result, "_context", context)
    object.__setattr__(result, "_residual_rank", residual_rank)
    object.__setattr__(result, "_iterations", iterations)
    object.__setattr__(result, "_retained_rank", observation.retained_rank)
    object.__setattr__(result, "_public_alphas", observation.alphas)

    document: dict[str, JSONValue] = {
        "schema": RESULT_SCHEMA,
        "method": method,
        "method_parameters": dict(parameters),
        "recovery_profile": context.profile,
        "recovery_version": context.recovery_version,
        "c2_method_version": context.c2_method_version,
        "target_class": target_class,
        "observation_sha256": observation.identity(),
        "recovery_context_sha256": context.identity(),
        "parameter_surface_sha256": result.surface_identity(),
        "recovered_update_sha256": result.content_identity(),
        "arithmetic_dtype": context.arithmetic_dtype,
        "residual_rank": residual_rank,
        "iterations": iterations,
        "retained_rank": observation.retained_rank,
        "public_alphas": list(observation.alphas),
    }
    object.__setattr__(result, "_identity", sha256_canonical(document))
    return result


# ----------------------------------------------------------------------------------------
# numerical core
# ----------------------------------------------------------------------------------------


def _svd(work: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    """The pinned SVD, with one deterministic escalation on non-convergence [AUTH: 01 §12].

    `numpy.linalg.svd` calls LAPACK's divide-and-conquer driver (`gesdd`). The C2 iteration
    reaches ill-conditioned residual iterates on which `gesdd` genuinely fails to converge —
    finite input, ordinary magnitudes, condition number around 1e8 — while LAPACK's
    QR-iteration driver (`gesvd`) decomposes the same matrix without difficulty. The
    escalation is deterministic (one input, one path, one output), computes the SVD of the
    same matrix, and never runs unless the primary raises, so every iterate that already
    converges keeps its exact bytes. Rank and truncation semantics are untouched.

    A non-finite input is a different failure and is refused rather than escalated: no SVD
    driver can rescue a matrix that is not a parameter object [AUTH: 00 §22].
    """
    if not np.all(np.isfinite(work)):
        raise RecoveryError(
            "cannot decompose a residual carrying NaN or Inf; this is a malformed iterate,"
            " not an SVD backend failure [AUTH: 00 §22]"
        )
    try:
        left, singular, right = np.linalg.svd(work, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as scipy_svd  # type: ignore[import-untyped]

        left, singular, right = scipy_svd(
            work, full_matrices=False, lapack_driver="gesvd", check_finite=False
        )
    return left, singular, right


def truncate_to_rank(matrix: Matrix, *, rank: int, context: RecoveryExecutionContext) -> Matrix:
    """T_r(W): rank-r truncated-SVD reconstruction, cast back to the artifact dtype.

    The decomposition runs in the frozen workspace dtype for stability; the reconstruction is
    cast deterministically to float32 before it re-enters the iteration, so the common-source
    state never silently promotes [AUTH: 01 §10].
    """
    if rank < 1:
        raise RecoveryError(f"residual rank {rank!r} must be positive")
    work = np.asarray(matrix, dtype=context.workspace)
    left, singular, right = _svd(work)
    kept = min(rank, int(singular.size))
    reconstructed = (left[:, :kept] * singular[:kept]) @ right[:kept, :]
    return np.ascontiguousarray(reconstructed, dtype=context.arithmetic)


def _mean(stack: Sequence[Matrix], context: RecoveryExecutionContext) -> Matrix:
    """Mean in the artifact dtype. Accumulated explicitly to avoid float64 promotion."""
    total = np.zeros(stack[0].shape, dtype=context.arithmetic)
    for array in stack:
        total = np.asarray(total + array, dtype=context.arithmetic)
    return np.asarray(
        total / np.asarray(len(stack), dtype=context.arithmetic), dtype=context.arithmetic
    )


def _spectral_detuning_core(
    observations: Sequence[Matrix],
    *,
    residual_rank: int,
    n_iters: int,
    context: RecoveryExecutionContext,
) -> Matrix:
    """The frozen C2 core for ONE target matrix [AUTH: 00 §25 P0-C2].

    Model: X_i = S + L_i with rank(L_i) <= r. Alternating, with a fixed iteration count:

        S^(0) = mean_i X_i
        R_i^(t) = X_i - S^(t-1)
        L_i^(t) = T_r(R_i^(t))
        S^(t)   = mean_i(X_i - L_i^(t))

    No gradients, no training data, no function queries, no true A, no partner bytes, and no
    stopping rule that could read an evaluation error [AUTH: 01 §29].
    """
    if len(observations) < 2:
        raise StructuralNAError(
            f"{STRUCTURAL_NA}: a shared-source recovery needs at least two observations;"
            " one observation cannot separate a common source from its residual"
        )
    if n_iters < 1:
        raise RecoveryError("the C2 iteration count must be at least one")
    stack = [np.asarray(x, dtype=context.arithmetic) for x in observations]
    source = _mean(stack, context)
    for _ in range(n_iters):
        residuals = [
            truncate_to_rank(
                np.asarray(x - source, dtype=context.arithmetic),
                rank=residual_rank,
                context=context,
            )
            for x in stack
        ]
        source = _mean(
            [
                np.asarray(x - low, dtype=context.arithmetic)
                for x, low in zip(stack, residuals, strict=True)
            ],
            context,
        )
    return source


# ----------------------------------------------------------------------------------------
# shared guards
# ----------------------------------------------------------------------------------------


def _require_observed_lineage(observation: object) -> ObservedLineage:
    """THE entry guard. Every public recovery path runs this before reading anything.

    The observation factories already refuse a duck-typed S07 descendant, but that only closes
    the front door: an object exposing `regime`, `k`, `updates`, `alphas`, `context`,
    `retained_rank`, `operator`, `fixed_alpha` and `identity()` could otherwise be handed
    straight to a solver and skip issuance entirely — carrying raw matrices, a forged regime,
    or the true protected constituent, under a borrowed observation identity.

    So the type is checked BEFORE any scientific field is read. Nothing about the attacker
    view is inferred from what an object happens to expose [AUTH: 00 §12, §25; 01 §16].
    """
    if not isinstance(observation, ObservedLineage):
        raise RecoveryInputError(
            "a recovery method runs on a factory-issued ObservedLineage, not"
            f" {type(observation).__name__}; an object that merely exposes .regime, .updates,"
            " .alphas or .identity() has not been through observation issuance and its"
            " scientific fields were never stripped [AUTH: 00 §12, §25]"
        )
    return observation


def _require_incomplete(observation: ObservedLineage, *, regime: str, method: str) -> None:
    if observation.regime != regime:
        raise NotAuthorizedError(
            f"{method} runs under {regime}; this observation is {observation.regime}"
        )
    if observation.has_partner:  # pragma: no cover - the factory never attaches one
        raise NotAuthorizedError(f"{method} may not receive partner information")


def _require_k(observation: ObservedLineage, *, method: str) -> None:
    """00 §11 / §25: an unknown-partner shared-source recovery needs k >= 2."""
    if observation.k < 2:
        raise StructuralNAError(
            f"{STRUCTURAL_NA}: {method} needs k >= 2 unknown-partner descendants;"
            " k = 1 cannot separate a common source from its residual [AUTH: 00 §11, §25]"
        )


def _residual_rank_for(observation: ObservedLineage, context: RecoveryExecutionContext) -> int:
    """The structural residual rank. For O3 it is the public retained rank s, never a choice."""
    if observation.operator == SVD_TRUNC:
        rank = observation.retained_rank
        if rank is None:
            raise RecoveryError("an O3 lineage must carry its public retained rank")
        return int(rank)
    return context.linear_residual_rank


def _target_class_for(observation: ObservedLineage) -> str:
    return (
        TRUNCATED_PROTECTED_SOURCE
        if observation.operator == SVD_TRUNC
        else ORIGINAL_PROTECTED_CONSTITUENT
    )


def _method_for(observation: ObservedLineage, *, spectral: bool, rescaled: bool) -> str:
    if observation.operator == SVD_TRUNC:
        return O3_C2B_RESCALED if rescaled else O3_SPECTRAL_FIXED
    if rescaled:
        return C2B_RESCALED
    return C2_SPECTRAL_FIXED if spectral else C2_NAIVE_SHARED_MEAN


# ----------------------------------------------------------------------------------------
# C1 — exact linear oracle
# ----------------------------------------------------------------------------------------


def recover_c1(observation: ObservedLineage) -> RecoveryResult:
    """A_hat = [C - (1-alpha)B] / alpha [AUTH: 00 §25 P0-C1]. An implementation oracle.

    Alpha is read off the observation's public merge metadata, not accepted from a caller, so
    a coefficient that disagrees with the issued descendant cannot be supplied. The partner
    came through `observe_known_partner`, which already checked it is this descendant's own
    partner.
    """
    _require_observed_lineage(observation)
    if observation.regime != Z_HIGH:
        raise NotAuthorizedError(
            f"{C1_METHOD} requires a known-partner {Z_HIGH} observation; this one is"
            f" {observation.regime} [AUTH: 00 §12, §25]"
        )
    if observation.operator != LINEAR:
        raise RecoveryError(
            f"{C1_METHOD} inverts the linear merge only; this descendant is {observation.operator}"
        )
    if observation.k != 1:
        raise RecoveryError("C1 inverts one descendant at a time")

    context = observation.context
    alpha = observation.fixed_alpha
    if not np.isfinite(alpha):
        raise RecoveryError(f"coefficient {alpha!r} is not finite")
    if alpha == 0.0:
        raise RecoveryError(
            "alpha = 0 leaves no trace of A in the descendant; the inversion is undefined"
        )

    descendant, partner = observation.updates[0], observation.partner
    if descendant.surface_identity() != partner.surface_identity():
        raise RecoveryError("the descendant and partner cover different parameter surfaces")

    dtype = context.arithmetic
    a32 = np.asarray(alpha, dtype=dtype)
    one_minus = np.asarray(1.0 - alpha, dtype=dtype)
    recovered = {
        name: np.asarray((descendant[name] - one_minus * partner[name]) / a32, dtype=dtype)
        for name in descendant.names
    }
    return _issue_result(
        method=C1_METHOD,
        target_class=ORIGINAL_PROTECTED_CONSTITUENT,
        tensors=recovered,
        observation=observation,
        context=context,
        residual_rank=None,
        iterations=None,
    )


# ----------------------------------------------------------------------------------------
# C2 — fixed known alpha, unknown partners
# ----------------------------------------------------------------------------------------


def recover_c2_fixed_alpha(observation: ObservedLineage) -> RecoveryResult:
    """Spectral shared-source recovery at fixed known alpha [AUTH: 00 §25 P0-C2].

    On base-relative updates D_i = alpha*A + (1-alpha)*B_i the common source is S = alpha*A,
    so the solver recovers S and the method divides once: A_hat = S_hat / alpha.
    """
    _require_observed_lineage(observation)
    method = _method_for(observation, spectral=True, rescaled=False)
    _require_incomplete(observation, regime=Z_INCOMPLETE_FIXED, method=method)
    _require_k(observation, method=method)

    context = observation.context
    alpha = observation.fixed_alpha
    if alpha == 0.0:
        raise RecoveryError("alpha = 0 leaves no trace of A in the descendants")
    rank = _residual_rank_for(observation, context)
    a32 = np.asarray(alpha, dtype=context.arithmetic)

    recovered: dict[str, Matrix] = {}
    for name in observation.updates[0].names:
        source = _spectral_detuning_core(
            [update[name] for update in observation.updates],
            residual_rank=rank,
            n_iters=context.c2_n_iters,
            context=context,
        )
        recovered[name] = np.asarray(source / a32, dtype=context.arithmetic)
    return _issue_result(
        method=method,
        target_class=_target_class_for(observation),
        tensors=recovered,
        observation=observation,
        context=context,
        residual_rank=rank,
        iterations=context.c2_n_iters,
    )


def recover_c2_naive_shared_mean(observation: ObservedLineage) -> RecoveryResult:
    """S_naive = mean_i D_i ; A_hat = S_naive / alpha [AUTH: 00 §25 P0-C2].

    The naive shared-mean baseline. Kept structurally separate: the spectral method never
    calls it and never carries its label.
    """
    _require_observed_lineage(observation)
    _require_incomplete(observation, regime=Z_INCOMPLETE_FIXED, method=C2_NAIVE_SHARED_MEAN)
    _require_k(observation, method=C2_NAIVE_SHARED_MEAN)
    context = observation.context
    alpha = observation.fixed_alpha
    if alpha == 0.0:
        raise RecoveryError("alpha = 0 leaves no trace of A in the descendants")
    a32 = np.asarray(alpha, dtype=context.arithmetic)
    recovered = {
        name: np.asarray(
            _mean([update[name] for update in observation.updates], context) / a32,
            dtype=context.arithmetic,
        )
        for name in observation.updates[0].names
    }
    return _issue_result(
        method=C2_NAIVE_SHARED_MEAN,
        target_class=_target_class_for(observation),
        tensors=recovered,
        observation=observation,
        context=context,
        residual_rank=None,
        iterations=None,
    )


# ----------------------------------------------------------------------------------------
# C2B — known varying alpha
# ----------------------------------------------------------------------------------------


def beta_for(alphas: Sequence[float]) -> tuple[float, ...]:
    """beta_i = (1 - alpha_i) / alpha_i [AUTH: 00 §25 P0-C2B]."""
    for alpha in alphas:
        if not np.isfinite(alpha):
            raise RecoveryError(f"coefficient {alpha!r} is not finite")
        if alpha == 0.0:
            raise RecoveryError("a zero coefficient cannot be rescaled")
    return tuple((1.0 - float(a)) / float(a) for a in alphas)


def rescale_observations(observation: ObservedLineage) -> dict[str, list[Matrix]]:
    """Dtilde_i = D_i / alpha_i, giving Dtilde_i = A + beta_i * B_i [AUTH: 00 §25 P0-C2B].

    Deterministic preconditioning on public coefficients only. No coefficient is inferred.
    """
    _require_observed_lineage(observation)
    context = observation.context
    beta_for(observation.alphas)
    rescaled: dict[str, list[Matrix]] = {}
    for name in observation.updates[0].names:
        rescaled[name] = [
            np.asarray(
                update[name] / np.asarray(alpha, dtype=context.arithmetic),
                dtype=context.arithmetic,
            )
            for update, alpha in zip(observation.updates, observation.alphas, strict=True)
        ]
    return rescaled


def recover_c2b_varying_alpha(observation: ObservedLineage) -> RecoveryResult:
    """Rescale by the public alpha_i, then run THE SAME C2 solver [AUTH: 00 §25 P0-C2B].

    After preconditioning the common source is A itself, so no second division follows. There
    is no new optimizer: `spectral_detuning_core` is the identical implementation C2 uses, at
    the same `c2_method_version`.
    """
    _require_observed_lineage(observation)
    method = _method_for(observation, spectral=True, rescaled=True)
    _require_incomplete(observation, regime=Z_INCOMPLETE_VARYING_KNOWN, method=method)
    _require_k(observation, method=method)

    context = observation.context
    rank = _residual_rank_for(observation, context)
    rescaled = rescale_observations(observation)
    recovered = {
        name: _spectral_detuning_core(
            stack, residual_rank=rank, n_iters=context.c2_n_iters, context=context
        )
        for name, stack in rescaled.items()
    }
    return _issue_result(
        method=method,
        target_class=_target_class_for(observation),
        tensors=recovered,
        observation=observation,
        context=context,
        residual_rank=rank,
        iterations=context.c2_n_iters,
    )


def recover_naive_unrescaled_fixed_alpha(
    observation: ObservedLineage, *, assumed_fixed_alpha: float
) -> RecoveryResult:
    """The 00 §25 NAIVE_UNRESCALED_FIXED_ALPHA comparator.

    00 §25 names this comparator without freezing any coefficient-estimation rule, so none is
    invented: the assumed alpha must be predeclared by the caller (S09 supplies it per cell),
    and the method deliberately does NOT rescale per descendant. It applies the fixed-alpha
    naive shared mean to the UNRESCALED D_i. No optimisation, no truth, no selection.
    """
    _require_observed_lineage(observation)
    _require_incomplete(
        observation,
        regime=Z_INCOMPLETE_VARYING_KNOWN,
        method=NAIVE_UNRESCALED_FIXED_ALPHA,
    )
    _require_k(observation, method=NAIVE_UNRESCALED_FIXED_ALPHA)
    if not np.isfinite(assumed_fixed_alpha) or assumed_fixed_alpha == 0.0:
        raise RecoveryError(
            f"the predeclared comparator alpha {assumed_fixed_alpha!r} must be finite and"
            " non-zero; there is no default and none is estimated [AUTH: 00 §25]"
        )
    context = observation.context
    a32 = np.asarray(assumed_fixed_alpha, dtype=context.arithmetic)
    recovered = {
        name: np.asarray(
            _mean([update[name] for update in observation.updates], context) / a32,
            dtype=context.arithmetic,
        )
        for name in observation.updates[0].names
    }
    return _issue_result(
        method=NAIVE_UNRESCALED_FIXED_ALPHA,
        target_class=_target_class_for(observation),
        tensors=recovered,
        observation=observation,
        context=context,
        residual_rank=None,
        iterations=None,
        method_parameters={"assumed_fixed_alpha": float(assumed_fixed_alpha)},
    )


# ----------------------------------------------------------------------------------------
# refusals
# ----------------------------------------------------------------------------------------


def recover_hidden_alpha(*args: object, **kwargs: object) -> RecoveryResult:
    """Unknown partner + unknown coefficient. Refused [AUTH: 00 §25]."""
    raise NotAuthorizedError(
        f"{STRUCTURAL_NOT_AUTHORIZED}: no hidden-alpha recovery method exists at S08, and no"
        " coefficient is estimated"
    )


def recover_dare_incomplete(*args: object, **kwargs: object) -> RecoveryResult:
    """Incomplete-lineage DARE recovery. Gated [AUTH: 00 §25; 01 §39 S08]."""
    raise NotAuthorizedError(
        f"{METHOD_GATED}: no incomplete-lineage DARE recovery method is implemented at S08"
    )


def recover_ties(*args: object, **kwargs: object) -> RecoveryResult:
    """TIES recovery. Not implemented [AUTH: 00 §10.4]."""
    raise NotAuthorizedError(
        "TIES is deferred and has no recovery method at S08; no placeholder stands in for it"
    )


def recovery_status(context: RecoveryExecutionContext) -> dict[str, Any]:
    """What S08 may and may not assert about the recovery axis [AUTH: 00 §24A.7, §25]."""
    return {
        "recovery_version": context.recovery_version,
        "implemented": list(METHODS),
        "c2_method": context.c2_method,
        "c2_method_version": context.c2_method_version,
        "c2_n_iters": context.c2_n_iters,
        "c2_rank_scheduler": context.c2_rank_scheduler,
        "O3_RECOVERY_IMPLEMENTED": True,
        # Scope, not tuning: read from the resolved config so the reported authorisation and
        # the executed one cannot drift apart [AUTH: 01 §17].
        "O3_RECOVERY_AUTHORIZATION": context.o3_recovery_authorization,
        "DARE_INCOMPLETE_RECOVERY": context.dare_incomplete_recovery_status,
        "HIDDEN_ALPHA_RECOVERY": context.hidden_alpha_recovery_status,
        "TIES_RECOVERY": "NOT_IMPLEMENTED",
        "recovery_context_sha256": context.identity(),
    }
