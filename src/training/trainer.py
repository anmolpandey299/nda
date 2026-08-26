"""Rank-32 LoRA training over the Block A tiny fixture [AUTH: 00 §8.3; 01 §8B, §20, §30].

The production backend is torch/PEFT, which is Linux-only in the `science` extra and absent
from the CPU/dev lock, so it is dependency-deferred. What ships here is the full training
*contract* plus a deterministic NumPy reference trainer over the Block A fixture, which is
what 01 §20 asks for: unit and integration testing must not depend on downloading a research
model.

The reference objective is a ridge-regularised least-squares fit of the adapted MLP
projections onto per-record targets. It is not a language model and its loss is not evidence
of anything scientific; it exists so the mechanics that DO carry downstream meaning can be
tested — adapter-only optimisation, exact optimiser-step counts, frozen base weights, the
induced update ΔW = scaling · B·A, deterministic data order, and save/load fidelity.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.data.membership import TrainingPlan
from src.provenance.hashing import JSONValue, sha256_bytes
from src.training.execution import TrainingExecutionContract

Matrix = NDArray[np.float64]

TRAINER_VERSION: Final = "s06.reference-lora-trainer.v1"
#: The real backend this contract is written against, and which is not installed here.
PRODUCTION_BACKEND_STATUS: Final = "NOT_RUN_DEPENDENCY(TORCH_PEFT_BACKEND)"

_MANTISSA: Final = float(1 << 53)


class TrainingError(RuntimeError):
    """A training run cannot proceed as configured."""


def _stream(label: str, count: int) -> Matrix:
    """The Block A SHA256 counter stream, so fixtures stay reproducible anywhere."""
    words: list[int] = []
    for index in range((count + 3) // 4):
        digest = hashlib.sha256(f"{label}|{index}".encode()).digest()
        words.extend(struct.unpack(">4Q", digest))
    raw = np.array(words[:count], dtype=np.uint64)
    return (raw >> np.uint64(11)).astype(np.float64) / _MANTISSA


@dataclass(frozen=True)
class LoraAdapter:
    """One adapter: A (r x in) and B (out x r) per target parameter, plus the scaling."""

    factors: Mapping[str, tuple[Matrix, Matrix]]
    rank: int
    scaling: float

    def induced_update(self, name: str) -> Matrix:
        a, b = self.factors[name]
        return self.scaling * (b @ a)

    def induced_updates(self) -> dict[str, Matrix]:
        return {name: self.induced_update(name) for name in sorted(self.factors)}

    def identity(self) -> str:
        """Hash the numbers, not a container format [AUTH: 01 §16, §19]."""
        digest = hashlib.sha256()
        for name in sorted(self.factors):
            a, b = self.factors[name]
            digest.update(name.encode("utf-8"))
            for array in (a, b):
                digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes(order="C"))
        return digest.hexdigest()

    def update_identity(self) -> str:
        digest = hashlib.sha256()
        for name, delta in self.induced_updates().items():
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(delta, dtype="<f8").tobytes(order="C"))
        return digest.hexdigest()


def initialise_adapter(
    shapes: Mapping[str, tuple[int, int]], *, rank: int, scaling: float, init_seed: int
) -> LoraAdapter:
    """A drawn from the init seed, B at zero.

    B = 0 means the adapter starts as an exact no-op, so an untrained adapter cannot change
    the model — the standard LoRA initialisation, and the reason a trained ΔW is attributable
    to training rather than to initialisation.
    """
    if rank < 1:
        raise TrainingError("LoRA rank must be positive")
    factors: dict[str, tuple[Matrix, Matrix]] = {}
    for name, (rows, columns) in sorted(shapes.items()):
        if min(rows, columns) < rank:
            raise TrainingError(
                f"{name}: shape {(rows, columns)} cannot carry rank {rank}; the induced update"
                " could not attain the declared rank"
            )
        a = (2.0 * _stream(f"{init_seed}|A|{name}", rank * columns) - 1.0).reshape(rank, columns)
        factors[name] = (a * 0.02, np.zeros((rows, rank), dtype=np.float64))
    return LoraAdapter(factors=factors, rank=rank, scaling=scaling)


def data_order(record_ids: Sequence[str], *, order_seed: int) -> tuple[str, ...]:
    """Deterministic epoch order, keyed on the data-order family alone [AUTH: 01 §30]."""
    keys = {
        record_id: hashlib.sha256(f"{order_seed}|{record_id}".encode()).hexdigest()
        for record_id in record_ids
    }
    return tuple(sorted(record_ids, key=lambda record_id: keys[record_id]))


@dataclass(frozen=True)
class TrainingSchedule:
    """The exact batches a run executes, and which records that reaches [AUTH: 00 §7.3].

    Built from the plan's own corpus under the frozen data-order policy. A matched control
    reaches the treated run's update budget by cycling its natural order — batch `k` of the
    run is batch `k mod len(batches)` of one epoch — so it repeats natural records it was
    given and never gains one it was not.
    """

    order: tuple[str, ...]
    batches: tuple[tuple[str, ...], ...]
    executed: tuple[tuple[str, ...], ...]
    role: str

    @property
    def consumed_ids(self) -> frozenset[str]:
        return frozenset(record_id for batch in self.executed for record_id in batch)

    @property
    def cycles(self) -> float:
        return len(self.executed) / len(self.batches)


def build_schedule(
    plan: TrainingPlan, *, order_seed: int, batch_size: int, budget: int, role: str
) -> TrainingSchedule:
    """The canonical schedule: every natural record and every included canary, once per epoch.

    The corpus holds one copy of each included canary [AUTH: 00 §7.3]. Epoch iteration
    revisits that corpus, which is ordinary training and does not make a canary appear twice
    in the corpus.
    """
    order = data_order(
        [*plan.natural_ids, *(canary.canary_id for canary in plan.canaries)],
        order_seed=order_seed,
    )
    if not order:
        raise TrainingError("the training plan carries no records")
    if batch_size < 1 or budget < 1:
        raise TrainingError("batch size and update budget must be positive")
    batches = tuple(
        tuple(order[index : index + batch_size]) for index in range(0, len(order), batch_size)
    )
    executed = tuple(batches[step % len(batches)] for step in range(budget))
    return TrainingSchedule(order=order, batches=batches, executed=executed, role=role)


@dataclass(frozen=True)
class TrainingOutcome:
    """What one reference run produced, and the evidence it may be checked against."""

    adapter: LoraAdapter
    optimizer_steps: int
    initial_loss: float
    final_loss: float
    schedule: TrainingSchedule
    contract: TrainingExecutionContract
    base_identity: str
    trainer_version: str = TRAINER_VERSION
    production_backend_status: str = PRODUCTION_BACKEND_STATUS

    @property
    def order(self) -> tuple[str, ...]:
        return self.schedule.order

    @property
    def audit(self) -> object:
        return self.contract.audit

    @property
    def seeds(self) -> object:
        return self.contract.seeds

    @property
    def natural_ids(self) -> tuple[str, ...]:
        return self.contract.plan.natural_ids

    @property
    def included_canary_ids(self) -> tuple[str, ...]:
        """Canaries the inclusion mask put in the training corpus [AUTH: 00 §7.2]."""
        return tuple(sorted(canary.canary_id for canary in self.contract.plan.canaries))

    @property
    def consumed_canary_ids(self) -> tuple[str, ...]:
        """Canaries the executed schedule actually reached."""
        return tuple(sorted(set(self.included_canary_ids) & self.schedule.consumed_ids))

    @property
    def unconsumed_canary_ids(self) -> tuple[str, ...]:
        """Planned but never reached. A successful run leaves this empty."""
        return tuple(sorted(set(self.included_canary_ids) - self.schedule.consumed_ids))

    @property
    def consumed_natural_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.natural_ids) & self.schedule.consumed_ids))

    def consumption_audit(self) -> dict[str, JSONValue]:
        """The membership evidence, available BEFORE any privacy evaluation [00 §7.3]."""
        return {
            "schedule_role": self.schedule.role,
            "optimizer_steps": self.optimizer_steps,
            "batches_per_epoch": len(self.schedule.batches),
            "schedule_cycles": self.schedule.cycles,
            "n_natural": len(self.natural_ids),
            "n_consumed_natural": len(self.consumed_natural_ids),
            "included_canary_ids": list(self.included_canary_ids),
            "consumed_canary_ids": list(self.consumed_canary_ids),
            "unconsumed_canary_ids": list(self.unconsumed_canary_ids),
        }

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "optimizer_steps": self.optimizer_steps,
            "execution_contract_sha256": self.contract.identity(),
            "consumption_audit": self.consumption_audit(),
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "adapter_identity": self.adapter.identity(),
            "induced_update_identity": self.adapter.update_identity(),
            "base_identity": self.base_identity,
            "trainer_version": self.trainer_version,
            "production_backend_status": self.production_backend_status,
        }


def _base_identity(base: Mapping[str, Matrix]) -> str:
    digest = hashlib.sha256()
    for name in sorted(base):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(base[name], dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def train_reference_lora(
    contract: TrainingExecutionContract, *, base: Mapping[str, Matrix]
) -> TrainingOutcome:
    """Adapter-only gradient descent under ONE authorised contract [AUTH: 00 §8.3; 01 §8B].

    Rank, scaling, targets, batch size, learning rate, seeds and the update budget all come
    from `contract`. There is no second place to pass them, so execution cannot disagree with
    the audit the contract carries.

    The run executes exactly `contract.optimizer_step_budget` updates — not "epochs' worth" —
    so two arms holding different numbers of records still perform the same number of
    optimiser updates [AUTH: 00 §7.5].

    Base weights are read but never written: the function holds them in a local copy and
    returns only adapter factors, so a caller can assert bitwise equality afterwards.
    """
    targets = list(contract.target_names)
    missing = [name for name in targets if name not in base]
    if missing:
        raise TrainingError(f"target parameter(s) absent from the base model: {missing[:3]}")

    plan, seeds = contract.plan, contract.seeds
    scaling, batch_size = contract.scaling, contract.batch_size
    shapes = {name: (base[name].shape[0], base[name].shape[1]) for name in targets}
    adapter = initialise_adapter(
        shapes, rank=contract.rank, scaling=scaling, init_seed=seeds["lora_init"]
    )
    schedule = build_schedule(
        plan,
        order_seed=seeds["data_order"],
        batch_size=batch_size,
        budget=contract.optimizer_step_budget,
        role=contract.schedule_role,
    )
    unconsumed = sorted({canary.canary_id for canary in plan.canaries} - schedule.consumed_ids)
    if unconsumed:
        raise TrainingError(
            f"{len(unconsumed)} included canary/canaries would never be consumed by this"
            f" schedule, first {unconsumed[0]!r}; a canary the run never reaches is not a"
            " trained member [AUTH: 00 §7.3, §7.5]"
        )

    factors = {name: (a.copy(), b.copy()) for name, (a, b) in adapter.factors.items()}
    order = schedule.order
    initial_loss = _loss(base, factors, targets, order, scaling)
    for batch in schedule.executed:
        signal = _batch_signal(batch, targets, base, scaling)
        for name in targets:
            a, b = factors[name]
            residual = scaling * (b @ a) - signal[name]
            grad_b = scaling * (residual @ a.T) / len(batch)
            grad_a = scaling * (b.T @ residual) / len(batch)
            factors[name] = (
                a - contract.learning_rate * grad_a,
                b - contract.learning_rate * grad_b,
            )
    trained = LoraAdapter(factors=factors, rank=contract.rank, scaling=scaling)
    return TrainingOutcome(
        adapter=trained,
        optimizer_steps=len(schedule.executed),
        initial_loss=initial_loss,
        final_loss=_loss(base, factors, targets, order, scaling),
        schedule=schedule,
        contract=contract,
        base_identity=_base_identity(base),
    )


def _batch_signal(
    batch: Sequence[str], targets: Sequence[str], base: Mapping[str, Matrix], scaling: float
) -> dict[str, Matrix]:
    """The per-record target the reference objective fits. Deterministic in the record ids."""
    signal: dict[str, Matrix] = {}
    for name in targets:
        rows, columns = base[name].shape
        accumulated = np.zeros((rows, columns), dtype=np.float64)
        for record_id in batch:
            draw = _stream(f"signal|{name}|{record_id}", rows * columns).reshape(rows, columns)
            accumulated += 2.0 * draw - 1.0
        signal[name] = 0.05 * accumulated / len(batch)
    return signal


def _loss(
    base: Mapping[str, Matrix],
    factors: Mapping[str, tuple[Matrix, Matrix]],
    targets: Sequence[str],
    order: Sequence[str],
    scaling: float,
) -> float:
    signal = _batch_signal(order, targets, base, scaling)
    total = 0.0
    for name in targets:
        a, b = factors[name]
        total += float(np.mean((scaling * (b @ a) - signal[name]) ** 2))
    return total / len(targets)


def save_adapter(adapter: LoraAdapter) -> dict[str, JSONValue]:
    """Serialise to plain lists; identity travels with it [AUTH: 01 §16]."""
    return {
        "rank": adapter.rank,
        "scaling": adapter.scaling,
        "adapter_identity": adapter.identity(),
        "factors": {
            name: {"A": a.tolist(), "B": b.tolist()}
            for name, (a, b) in sorted(adapter.factors.items())
        },
    }


def load_adapter(document: Mapping[str, JSONValue]) -> LoraAdapter:
    raw = document.get("factors")
    if not isinstance(raw, Mapping):
        raise TrainingError("adapter document carries no factors")
    factors = {
        str(name): (
            np.asarray(entry["A"], dtype=np.float64),  # type: ignore[index,call-overload]
            np.asarray(entry["B"], dtype=np.float64),  # type: ignore[index,call-overload]
        )
        for name, entry in raw.items()
    }
    adapter = LoraAdapter(
        factors=factors, rank=int(str(document["rank"])), scaling=float(str(document["scaling"]))
    )
    declared = document.get("adapter_identity")
    if isinstance(declared, str) and declared != adapter.identity():
        raise TrainingError("the reloaded adapter does not match its recorded identity")
    return adapter


def adapter_file_hash(document: Mapping[str, JSONValue]) -> str:
    from src.provenance.hashing import canonical_json_bytes

    return sha256_bytes(canonical_json_bytes(dict(document)))
