"""DP-LoRA contract and privacy accounting [AUTH: 00 §8.2, §8.4, §9; 01 §8B, §17, §30].

Two separable things live here.

**The mechanism contract** — adjacency, clipping norm, noise multiplier, sample rate, step
count, delta and the DP seed — is what a DP run must declare before it runs, and what its
manifest must carry afterwards. It is enforced here.

**The accountant** is the analytic Renyi accountant for the Sampled Gaussian Mechanism
(Mironov, Talwar and Zhang 2019), composed over steps and converted to (epsilon, delta) by
the standard RDP conversion. It is exact at integer Renyi orders; restricting the order grid
to integers can only over-state epsilon, never under-state it, which is the safe direction
for a privacy claim.

The per-sample-gradient *training* backend is Opacus, which is not installed in the CPU/dev
lane, so it is recorded as dependency-deferred rather than imitated. The clipping and noise
primitives below are the reference semantics that backend must reproduce, and they are what
the tiny-fixture tests exercise.

Nothing here decides whether a run may be believed. 00 §8.4 gives the DP arm one smoke seed
whose only job is to show the pipeline runs; `privacy_endpoint_verdict` is how that fact
reaches the Block B reporting gate without any Block B code changing.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.analysis.metrics import ELIGIBLE, INELIGIBLE, EligibilityVerdict
from src.provenance.hashing import JSONValue, sha256_canonical

FloatArray = NDArray[np.float64]

ACCOUNTANT_VERSION: Final = "s06.rdp-sgm-integer-orders.v1"
#: The per-sample-gradient backend this contract is written against.
DP_BACKEND_STATUS: Final = "NOT_RUN_DEPENDENCY(OPACUS_BACKEND)"

#: 00 §8.2: the unit of privacy is one training record, added or removed.
SAMPLE_LEVEL_ADJACENCY: Final = "SAMPLE_LEVEL_ADD_REMOVE_ONE_RECORD"
#: The adjacency relations this project explicitly does NOT claim.
UNSUPPORTED_ADJACENCY: Final[tuple[str, ...]] = ("USER_LEVEL", "GROUP_LEVEL", "DOCUMENT_LEVEL")

#: The assumptions the epsilon is only valid under. Recorded, because an epsilon quoted
#: without them is not a statement about anything [AUTH: 00 §8.2].
ACCOUNTING_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "POISSON_SUBSAMPLING_WITH_FIXED_RATE",
    "PER_SAMPLE_GRADIENT_CLIPPING_BEFORE_NOISE",
    "GAUSSIAN_NOISE_SCALED_BY_CLIPPING_NORM",
    "HOMOGENEOUS_COMPOSITION_OVER_STEPS",
    "ONE_RECORD_APPEARS_ONCE_PER_EPOCH_NO_REPETITION",
)

INFERENTIAL: Final = "INFERENTIAL"
NON_INFERENTIAL: Final = "NON_INFERENTIAL"
SMOKE_ONLY: Final = "SMOKE_ONLY"

#: A run fact that has not been bound to a real artifact yet. A set cannot check consistency
#: across runs that do not say what they are, so an unbound field refuses inferential standing.
UNBOUND: Final = "UNBOUND"

#: A backend that actually ran. Mirrors `src.training.execution.BACKEND_READY`; declared here
#: rather than imported so `src.dp` does not depend on `src.training`.
BACKEND_READY: Final = "READY"

#: Renyi orders. An accountant grid is a numerical choice, not a scientific threshold.
RDP_ORDERS: Final[tuple[int, ...]] = tuple(range(2, 257))

_MANTISSA: Final = float(1 << 53)


class DPError(ValueError):
    """A DP configuration is incoherent, or an epsilon was claimed without being computed."""


class DPUsageError(RuntimeError):
    """A non-inferential DP run was used where an inferential one is required."""


# ---------------------------------------------------------------------------------------
# mechanism contract
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DPMechanism:
    """Everything that must be declared BEFORE a DP run [AUTH: 00 §8.2; 01 §17].

    `requested_epsilon` is the target the noise multiplier was chosen for. It is a *request*
    and is kept in its own field for exactly that reason: the achieved epsilon is whatever
    the accountant returns for the parameters actually used, and the two are allowed to
    differ. See `DPAccounting`.
    """

    adjacency: str
    delta: float
    clipping_norm: float
    noise_multiplier: float
    sample_rate: float
    steps: int
    dp_seed: int
    requested_epsilon: float | None = None
    assumptions: tuple[str, ...] = ACCOUNTING_ASSUMPTIONS

    def __post_init__(self) -> None:
        problems = self.problems()
        if problems:
            raise DPError("; ".join(problems))

    def problems(self) -> tuple[str, ...]:
        found: list[str] = []
        if self.adjacency != SAMPLE_LEVEL_ADJACENCY:
            found.append(
                f"adjacency {self.adjacency!r} is not the declared sample-level relation"
                f" {SAMPLE_LEVEL_ADJACENCY!r} [AUTH: 00 §8.2]"
            )
        if not 0.0 < self.delta < 1.0:
            found.append("delta must lie strictly between 0 and 1")
        if self.clipping_norm <= 0.0:
            found.append("the per-sample clipping norm must be positive")
        if self.noise_multiplier <= 0.0:
            found.append("the noise multiplier must be positive; sigma = 0 is not DP")
        if not 0.0 < self.sample_rate <= 1.0:
            found.append("the sample rate must lie in (0, 1]")
        if self.steps < 1:
            found.append("a DP run must take at least one step")
        if isinstance(self.dp_seed, bool) or not isinstance(self.dp_seed, int):
            found.append("the DP noise seed must be an integer and must be recorded")
        return tuple(found)

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "adjacency": self.adjacency,
            "delta": self.delta,
            "clipping_norm": self.clipping_norm,
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
            "steps": self.steps,
            "dp_seed": self.dp_seed,
            "requested_epsilon": self.requested_epsilon,
            "accounting_assumptions": list(self.assumptions),
        }


def poisson_sample_rate(*, batch_size: int, dataset_size: int) -> float:
    """q = B / N. The rate the accountant assumes, derived rather than asserted."""
    if batch_size < 1 or dataset_size < 1:
        raise DPError("batch size and dataset size must be positive")
    if batch_size > dataset_size:
        raise DPError("the expected batch cannot exceed the dataset")
    return batch_size / dataset_size


def steps_for(*, epochs: int, dataset_size: int, batch_size: int) -> int:
    """Optimiser steps under a fixed expected batch size, so `steps` is not free-floating."""
    if epochs < 1:
        raise DPError("epochs must be positive")
    return epochs * math.ceil(dataset_size / batch_size)


# ---------------------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------------------


def _log_binomial(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _rdp_sgm_step(order: int, *, sample_rate: float, noise_multiplier: float) -> float:
    """RDP of one Sampled Gaussian step at an integer order [Mironov et al. 2019, Thm 4]."""
    if sample_rate == 1.0:
        return order / (2.0 * noise_multiplier**2)
    terms: list[float] = []
    log_q = math.log(sample_rate)
    log_1mq = math.log1p(-sample_rate)
    for k in range(order + 1):
        terms.append(
            _log_binomial(order, k)
            + (order - k) * log_1mq
            + k * log_q
            + (k * k - k) / (2.0 * noise_multiplier**2)
        )
    top = max(terms)
    total = top + math.log(sum(math.exp(t - top) for t in terms))
    return total / (order - 1)


def _epsilon_at(order: int, rdp_total: float, delta: float) -> float:
    """RDP -> (eps, delta), with the standard log((a-1)/a) improvement."""
    return (
        rdp_total
        - (math.log(delta) + math.log(order)) / (order - 1)
        + math.log((order - 1) / order)
    )


@dataclass(frozen=True)
class DPAccounting:
    """What the accountant returned, and under what version it returned it.

    `achieved_epsilon` is computed from `mechanism` alone. It is never assigned from
    `requested_epsilon`; `agreement_gap` exists so a report can show how far the realised
    guarantee sits from the target instead of quietly presenting the target as the result.
    """

    achieved_epsilon: float
    optimal_order: int
    delta: float
    accountant_version: str
    mechanism: DPMechanism

    @property
    def requested_epsilon(self) -> float | None:
        return self.mechanism.requested_epsilon

    @property
    def agreement_gap(self) -> float | None:
        if self.mechanism.requested_epsilon is None:
            return None
        return self.achieved_epsilon - self.mechanism.requested_epsilon

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "achieved_epsilon": self.achieved_epsilon,
            "requested_epsilon": self.requested_epsilon,
            "epsilon_agreement_gap": self.agreement_gap,
            "optimal_rdp_order": self.optimal_order,
            "delta": self.delta,
            "accountant_version": self.accountant_version,
            "dp_backend_status": DP_BACKEND_STATUS,
        }


def account(mechanism: DPMechanism, *, orders: Sequence[int] = RDP_ORDERS) -> DPAccounting:
    """Achieved (epsilon, delta) for the parameters actually used. Deterministic."""
    usable = [int(order) for order in orders if int(order) > 1]
    if not usable:
        raise DPError("the Renyi order grid is empty")
    best_epsilon = math.inf
    best_order = usable[0]
    for order in usable:
        step = _rdp_sgm_step(
            order,
            sample_rate=mechanism.sample_rate,
            noise_multiplier=mechanism.noise_multiplier,
        )
        candidate = _epsilon_at(order, mechanism.steps * step, mechanism.delta)
        if candidate < best_epsilon:
            best_epsilon, best_order = candidate, order
    return DPAccounting(
        achieved_epsilon=max(best_epsilon, 0.0),
        optimal_order=best_order,
        delta=mechanism.delta,
        accountant_version=ACCOUNTANT_VERSION,
        mechanism=mechanism,
    )


def calibrate_noise_multiplier(
    *,
    target_epsilon: float,
    delta: float,
    sample_rate: float,
    steps: int,
    clipping_norm: float,
    dp_seed: int,
    tolerance: float = 1e-4,
    max_iterations: int = 200,
) -> DPMechanism:
    """Smallest sigma on a bisection grid whose ACHIEVED epsilon is at most the target.

    The returned mechanism records `target_epsilon` as `requested_epsilon`. The achieved
    value still has to be obtained from `account`, and will generally be slightly below the
    target because the search returns a feasible sigma rather than an exact solution.
    """
    if target_epsilon <= 0.0:
        raise DPError("the target epsilon must be positive")
    low, high = 0.3, 0.6
    for _ in range(max_iterations):
        trial = DPMechanism(
            adjacency=SAMPLE_LEVEL_ADJACENCY,
            delta=delta,
            clipping_norm=clipping_norm,
            noise_multiplier=high,
            sample_rate=sample_rate,
            steps=steps,
            dp_seed=dp_seed,
        )
        if account(trial).achieved_epsilon <= target_epsilon:
            break
        low, high = high, high * 2.0
    else:  # pragma: no cover - only reachable with an absurd target
        raise DPError("no noise multiplier in the search range attains the target epsilon")

    for _ in range(max_iterations):
        if high - low <= tolerance:
            break
        middle = 0.5 * (low + high)
        trial = DPMechanism(
            adjacency=SAMPLE_LEVEL_ADJACENCY,
            delta=delta,
            clipping_norm=clipping_norm,
            noise_multiplier=middle,
            sample_rate=sample_rate,
            steps=steps,
            dp_seed=dp_seed,
        )
        if account(trial).achieved_epsilon <= target_epsilon:
            high = middle
        else:
            low = middle
    return DPMechanism(
        adjacency=SAMPLE_LEVEL_ADJACENCY,
        delta=delta,
        clipping_norm=clipping_norm,
        noise_multiplier=high,
        sample_rate=sample_rate,
        steps=steps,
        dp_seed=dp_seed,
        requested_epsilon=target_epsilon,
    )


# ---------------------------------------------------------------------------------------
# reference mechanism primitives
# ---------------------------------------------------------------------------------------


def clip_to_norm(gradient: FloatArray, max_norm: float) -> FloatArray:
    """Per-sample clipping: scale by min(1, C/||g||). Bounds one record's contribution."""
    if max_norm <= 0.0:
        raise DPError("the clipping norm must be positive")
    norm = float(np.linalg.norm(gradient))
    if norm <= max_norm or norm == 0.0:
        return np.array(gradient, dtype=np.float64)
    return np.asarray(gradient, dtype=np.float64) * (max_norm / norm)


def clipped_sum(gradients: Sequence[FloatArray], max_norm: float) -> FloatArray:
    """Sum of per-sample clipped gradients. Clipping happens BEFORE the sum, never after."""
    if not gradients:
        raise DPError("no per-sample gradients to aggregate")
    total = np.zeros_like(np.asarray(gradients[0], dtype=np.float64))
    for gradient in gradients:
        total = total + clip_to_norm(np.asarray(gradient, dtype=np.float64), max_norm)
    return total


def gaussian_noise(shape: tuple[int, ...], *, scale: float, dp_seed: int, step: int) -> FloatArray:
    """Deterministic N(0, scale^2) noise from the recorded DP seed [AUTH: 01 §30].

    Box-Muller over a SHA256 counter stream rather than a library sampler, so the same seed
    reproduces the same noise on any platform and any NumPy version.
    """
    count = int(np.prod(shape)) if shape else 1
    words: list[int] = []
    needed = 2 * count
    for index in range((needed + 3) // 4):
        digest = hashlib.sha256(f"dp|{dp_seed}|{step}|{index}".encode()).digest()
        words.extend(struct.unpack(">4Q", digest))
    raw = (np.array(words[:needed], dtype=np.uint64) >> np.uint64(11)).astype(np.float64)
    uniforms = (raw + 0.5) / _MANTISSA
    first, second = uniforms[:count], uniforms[count:]
    values = np.sqrt(-2.0 * np.log(first)) * np.cos(2.0 * np.pi * second)
    return (scale * values).reshape(shape)


def privatise(gradients: Sequence[FloatArray], *, mechanism: DPMechanism, step: int) -> FloatArray:
    """One DP-SGD aggregation step: clip per sample, sum, add sigma*C noise, average."""
    total = clipped_sum(gradients, mechanism.clipping_norm)
    noise = gaussian_noise(
        total.shape,
        scale=mechanism.noise_multiplier * mechanism.clipping_norm,
        dp_seed=mechanism.dp_seed,
        step=step,
    )
    return (total + noise) / len(gradients)


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# the DP arm: one run describes facts, a validated SET confers inferential standing
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DPRun:
    """One DP execution and the facts it can attest to about itself.

    A run may NOT confer inferential standing on itself. 00 §8.4 gives the DP arm one smoke
    seed, and a single seed supports no across-seed variance estimate, so
    `inferential_status` is a derived property that is always `NON_INFERENTIAL` — there is no
    field a caller can set to `INFERENTIAL`. Standing comes from `InferentialDPSet`, which
    validates that the required seeds are present, consistent and evidentiary-ready.

    The scientific-condition fields are carried so the set validator can check that the runs
    it is asked to combine are the same condition at different seeds rather than different
    conditions [AUTH: 00 §8.2, §8.4, §18.1].
    """

    label: str
    training_seed: int
    accounting: DPAccounting
    execution_contract_sha256: str = UNBOUND
    model_identity: str = UNBOUND
    data_manifest_sha256: str = UNBOUND
    training_config_sha256: str = UNBOUND
    lora_policy: str = UNBOUND
    backend_status: str = DP_BACKEND_STATUS
    accounting_backend_status: str = DP_BACKEND_STATUS

    @property
    def inferential_status(self) -> str:
        """Always NON_INFERENTIAL. Derived, so no label string can override it."""
        return NON_INFERENTIAL

    @property
    def may_enter_privacy_endpoints(self) -> bool:
        """False for every individual run, including one in a valid set.

        Endpoint eligibility is a property of the set, so the question is asked of the set.
        """
        return False

    @property
    def evidentiary_ready(self) -> bool:
        return self.backend_status == BACKEND_READY and (
            self.accounting_backend_status == BACKEND_READY
        )

    def condition_identity(self) -> str:
        """Everything except the seed. Two runs of one condition must agree on all of it."""
        mechanism = self.accounting.mechanism
        return sha256_canonical(
            {
                "model_identity": self.model_identity,
                "data_manifest_sha256": self.data_manifest_sha256,
                "training_config_sha256": self.training_config_sha256,
                "lora_policy": self.lora_policy,
                "adjacency": mechanism.adjacency,
                "delta": mechanism.delta,
                "clipping_norm": mechanism.clipping_norm,
                "noise_multiplier": mechanism.noise_multiplier,
                "sample_rate": mechanism.sample_rate,
                "steps": mechanism.steps,
                "accountant_version": self.accounting.accountant_version,
                "assumptions": list(mechanism.assumptions),
            }
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "label": self.label,
            "training_seed": self.training_seed,
            "inferential_status": self.inferential_status,
            "execution_contract_sha256": self.execution_contract_sha256,
            "model_identity": self.model_identity,
            "data_manifest_sha256": self.data_manifest_sha256,
            "training_config_sha256": self.training_config_sha256,
            "lora_policy": self.lora_policy,
            "backend_status": self.backend_status,
            "accounting_backend_status": self.accounting_backend_status,
            "condition_identity": self.condition_identity(),
            **self.accounting.as_dict(),
        }


def dp_smoke_run(*, smoke_seed: int, mechanism: DPMechanism, **facts: str) -> DPRun:
    """The 00 §8.4 DP arm: one seed, and structurally never inferential."""
    return DPRun(label=SMOKE_ONLY, training_seed=smoke_seed, accounting=account(mechanism), **facts)


@dataclass(frozen=True)
class InferentialDPSet:
    """A collection of DP runs that may — together — support an inferential statement.

    00 §8.2 fixes the minimum inferential DP set at seeds {101, 202, 303} for one frozen
    scientific condition. This validates all three requirements at once: the seeds are
    present, every member is the same condition, and every member is evidentiary-ready.
    """

    runs: tuple[DPRun, ...]
    required_seeds: tuple[int, ...]

    def problems(self) -> tuple[str, ...]:
        found: list[str] = []
        if not self.runs:
            return ("an inferential DP set contains no runs",)
        seeds = [run.training_seed for run in self.runs]
        if len(set(seeds)) != len(seeds):
            found.append("the set repeats a training seed; three runs at one seed are one seed")
        missing = sorted(set(self.required_seeds) - set(seeds))
        if missing:
            found.append(
                f"the inferential DP set is missing seed(s) {missing}; 00 §8.2 requires at"
                f" least {sorted(self.required_seeds)} before any inferential DP-arm statement"
            )
        conditions = {run.condition_identity() for run in self.runs}
        if len(conditions) > 1:
            found.append(
                f"the {len(self.runs)} runs describe {len(conditions)} different scientific"
                " conditions; an across-seed statement requires one frozen condition"
                " [AUTH: 00 §8.2]"
            )
        unbound = sorted(
            run.label
            for run in self.runs
            if UNBOUND
            in {
                run.model_identity,
                run.data_manifest_sha256,
                run.training_config_sha256,
                run.lora_policy,
                run.execution_contract_sha256,
            }
        )
        if unbound:
            found.append(
                f"{len(unbound)} run(s) do not bind their model/data/config/contract identity,"
                f" first {unbound[0]!r}; consistency cannot be checked"
            )
        not_ready = sorted(str(run.training_seed) for run in self.runs if not run.evidentiary_ready)
        if not_ready:
            found.append(
                f"seed(s) {', '.join(not_ready)} ran on a backend or accountant that is not"
                " READY; a dependency-deferred run cannot support inference"
                " [AUTH: 01 §12, §16]"
            )
        return tuple(found)

    @property
    def eligible(self) -> bool:
        return not self.problems()

    @property
    def status(self) -> str:
        return INFERENTIAL if self.eligible else NON_INFERENTIAL

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "status": self.status,
            "training_seeds": sorted(run.training_seed for run in self.runs),
            "required_seeds": sorted(self.required_seeds),
            "condition_identity": (self.runs[0].condition_identity() if self.runs else None),
            "problems": list(self.problems()),
        }


def inferential_dp_set(runs: Sequence[DPRun], *, required_seeds: Sequence[int]) -> InferentialDPSet:
    return InferentialDPSet(runs=tuple(runs), required_seeds=tuple(required_seeds))


def require_inferential_dp_set(dp_set: InferentialDPSet) -> None:
    problems = dp_set.problems()
    if problems:
        raise DPUsageError(
            "this DP collection may be reported as a feasibility result only, and may not"
            f" enter R_priv, G_L or G_R [AUTH: 00 §8.4, §18.1]: {'; '.join(problems)}"
        )


def assert_inferential(dp_set: InferentialDPSet) -> None:
    """The only gate. It takes the set, because standing is a set-level property."""
    require_inferential_dp_set(dp_set)


def privacy_endpoint_verdict(
    base: EligibilityVerdict, *, dp_set: InferentialDPSet | None
) -> EligibilityVerdict:
    """Fold DP inferential standing into the Block B eligibility gate.

    `dp_set=None` means the condition is not a DP condition and the base verdict stands. A DP
    condition must pass a validated set; anything else downgrades an eligible verdict, and can
    never upgrade an ineligible one. Returns a verdict rather than a boolean so the existing
    `report_normalised_privacy` refuses on its own terms with no change to Block B.
    """
    if dp_set is None or dp_set.eligible:
        return base
    reason = (
        "the DP arm has no validated inferential seed set "
        f"({'; '.join(dp_set.problems())}) [AUTH: 00 §8.4]"
    )
    if base.status == ELIGIBLE:
        return EligibilityVerdict(
            status=INELIGIBLE,
            oracle_median=base.oracle_median,
            signal_ci=base.signal_ci,
            reasons=(reason,),
        )
    return EligibilityVerdict(
        status=base.status,
        oracle_median=base.oracle_median,
        signal_ci=base.signal_ci,
        reasons=(*base.reasons, reason),
    )


def mechanism_from_config(values: Mapping[str, JSONValue], *, dp_seed: int) -> DPMechanism:
    """Build the mechanism from resolved config; every material constant comes from there.

    `delta` is read from the real field, never from a fixture field: it is a DP scientific
    parameter, and a fixture value silently becoming the real delta would change what the
    epsilon means [AUTH: 00 §8.2; 01 §17].
    """
    required = ("delta", "clipping_norm", "noise_multiplier", "sample_rate", "steps")
    missing = [name for name in required if values.get(name) is None]
    if missing:
        raise DPError(f"DP config is missing required field(s): {missing}")
    requested = values.get("requested_epsilon")
    return DPMechanism(
        adjacency=str(values.get("adjacency", SAMPLE_LEVEL_ADJACENCY)),
        delta=float(str(values["delta"])),
        clipping_norm=float(str(values["clipping_norm"])),
        noise_multiplier=float(str(values["noise_multiplier"])),
        sample_rate=float(str(values["sample_rate"])),
        steps=int(str(values["steps"])),
        dp_seed=dp_seed,
        requested_epsilon=None if requested is None else float(str(requested)),
    )
