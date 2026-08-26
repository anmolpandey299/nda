"""One immutable training execution contract [AUTH: 00 §8.3; 01 §8B, §17, §30].

Before this, three things could disagree with each other while every individual object stayed
valid: the attached audit said MLP_ONLY while `lm_head.weight` trained; the audit recorded
rank 32 while execution ran rank 8; and a treated arm ran ten optimiser updates against its
control's eight because the two had different record counts at the same epoch count.

The fix is that execution takes ONE object. `TrainingExecutionContract` binds the plan, the
model contract, the target policy, the actually selected module names, the audit derived from
those names, the rank/scaling/dropout that execution will use, the resolved settings, the
complete seed families, the precision, the sequence length, the optimiser-update budget, the
privacy regime and the backend state. `train_reference_lora` consumes it and takes no
independent rank, scaling or target list, so there is nothing left to disagree with.

The update budget is explicit for the same reason. "Same epochs" is not "same optimiser
steps" once the two arms hold different numbers of records, so the contract carries an
`optimizer_step_budget` and both arms execute exactly that many updates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from src.data.membership import TrainingPlan
from src.provenance.hashing import JSONValue, sha256_canonical
from src.training.lora import (
    FORBIDDEN_ROLES,
    TargetMapping,
    TrainableAudit,
    audit_trainable_surface,
    select_target_parameters,
)
from src.training.model_contract import PanelMember
from src.training.seeds import SeedFamilies

CONTRACT_VERSION: Final = "s06.training-execution-contract.v1"

NON_DP: Final = "NON_DP"
DP_SAMPLE_LEVEL: Final = "DP_SAMPLE_LEVEL"
PRIVACY_REGIMES: Final[tuple[str, ...]] = (NON_DP, DP_SAMPLE_LEVEL)

#: What the backend that will actually run this is. Only READY may carry an evidentiary run.
BACKEND_READY: Final = "READY"
BACKEND_DEPENDENCY_DEFERRED: Final = "NOT_RUN_DEPENDENCY"


class ExecutionContractError(ValueError):
    """A training execution cannot be authorised as specified."""


def adapter_parameter_names(target_names: Sequence[str]) -> tuple[str, ...]:
    """The adapter surface implied by the selected targets. Not a caller's list."""
    return tuple(f"{name}.lora_{part}" for name in target_names for part in ("A", "B"))


@dataclass(frozen=True)
class BackendState:
    """What will execute, and whether it is actually available."""

    identity: str
    status: str
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.status == BACKEND_READY

    def as_dict(self) -> dict[str, JSONValue]:
        return {"identity": self.identity, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class TrainingExecutionContract:
    """Everything one training execution is authorised to do, bound into one object.

    Build with `authorise_training`. Direct construction is possible but pointless: the
    validation that makes the contract meaningful lives in `__post_init__`, so an inconsistent
    contract cannot exist either way.
    """

    plan: TrainingPlan
    model: PanelMember
    mapping: TargetMapping
    target_names: tuple[str, ...]
    audit: TrainableAudit
    rank: int
    scaling: float
    dropout: float
    learning_rate: float
    batch_size: int
    optimizer_step_budget: int
    precision: str
    sequence_length: int
    seeds: SeedFamilies
    privacy_regime: str
    backend: BackendState
    resolved_config_sha256: str
    version: str = CONTRACT_VERSION

    @property
    def schedule_role(self) -> str:
        """`MATCHED_CONTROL_CYCLED_SCHEDULE` when the control repeats its natural order."""
        return schedule_role(
            plan=self.plan, budget=self.optimizer_step_budget, batch_size=self.batch_size
        )

    def __post_init__(self) -> None:
        problems = self.problems()
        if problems:
            raise ExecutionContractError("; ".join(problems))

    def problems(self) -> tuple[str, ...]:
        found: list[str] = []
        if self.privacy_regime not in PRIVACY_REGIMES:
            found.append(f"{self.privacy_regime!r} is not a declared privacy regime")
        if self.seeds.training_seed != self.plan.training_seed:
            found.append(
                f"the seed families are for training seed {self.seeds.training_seed}, the plan"
                f" is seed {self.plan.training_seed} [AUTH: 01 §30]"
            )
        if (self.privacy_regime == DP_SAMPLE_LEVEL) != ("dp_noise" in self.seeds.values):
            found.append(
                "the recorded seed families and the declared privacy regime disagree about"
                " whether a DP noise stream was drawn [AUTH: 01 §30]"
            )
        if self.audit.rank != self.rank:
            found.append(f"audit rank {self.audit.rank} is not the execution rank {self.rank}")
        if self.audit.scaling != self.scaling:
            found.append(
                f"audit scaling {self.audit.scaling} is not the execution scaling {self.scaling}"
            )
        if self.audit.dropout != self.dropout:
            found.append("audit dropout is not the execution dropout")
        if self.audit.policy != self.mapping.policy:
            found.append("the audit records a different target policy than the mapping")
        if tuple(sorted(self.audit.adapter_parameters)) != tuple(
            sorted(adapter_parameter_names(self.target_names))
        ):
            found.append(
                "the audited adapter surface is not the surface the selected targets imply"
            )
        forbidden = sorted(
            name
            for name in self.target_names
            if any(marker in name for marker in FORBIDDEN_ROLES)
            or any(marker in name for marker in self.mapping.forbidden_modules)
        )
        if forbidden:
            found.append(
                f"{len(forbidden)} selected target(s) are forbidden by the policy, first"
                f" {forbidden[0]!r} [AUTH: 01 §8B]"
            )
        if self.optimizer_step_budget < 1:
            found.append("the optimiser-update budget must be at least one step")
        elif self.batch_size >= 1 and self.plan.n_training_records >= 1:
            required = batches_per_epoch(self.plan.n_training_records, self.batch_size)
            if self.plan.canaries_permitted and self.optimizer_step_budget < required:
                found.append(
                    f"the {self.optimizer_step_budget}-update budget cannot cover the"
                    f" {self.plan.n_training_records}-record treated corpus, which needs"
                    f" {required} updates at batch size {self.batch_size}; included canaries"
                    " would never be consumed [AUTH: 00 §7.3, §7.5]"
                )
        if self.batch_size < 1:
            found.append("the batch size must be positive")
        if self.sequence_length < 1:
            found.append("the sequence length must be positive")
        if not self.precision:
            found.append("the execution precision must be recorded")
        return tuple(found)

    def identity(self) -> str:
        """The hash a manifest binds, so a run cannot claim a contract it did not execute."""
        return sha256_canonical(
            {
                "version": self.version,
                "plan": self.plan.as_dict(),
                "model_id": self.model.model_id,
                "model_revision": self.model.revision,
                "policy": self.mapping.policy,
                "target_names": list(self.target_names),
                "audit": self.audit.as_dict(),
                "rank": self.rank,
                "scaling": self.scaling,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "optimizer_step_budget": self.optimizer_step_budget,
                "precision": self.precision,
                "sequence_length": self.sequence_length,
                "seeds": self.seeds.as_dict(),
                "privacy_regime": self.privacy_regime,
                "backend": self.backend.as_dict(),
                "schedule_role": self.schedule_role,
                "resolved_config_sha256": self.resolved_config_sha256,
            }
        )


def authorise_training(
    *,
    plan: TrainingPlan,
    model: PanelMember,
    mapping: TargetMapping,
    base_parameter_names: Sequence[str],
    rank: int,
    scaling: float,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    optimizer_step_budget: int,
    precision: str,
    sequence_length: int,
    seeds: SeedFamilies,
    privacy_regime: str,
    backend: BackendState,
    resolved_config_sha256: str,
) -> TrainingExecutionContract:
    """Derive the target list and the audit from the mapping, then bind everything.

    The audit is not a parameter. It is computed here from the parameters execution will
    actually adapt, which is what makes "the audit says MLP_ONLY while lm_head trains"
    unrepresentable rather than merely discouraged.
    """
    target_names = tuple(select_target_parameters(list(base_parameter_names), mapping))
    adapters = adapter_parameter_names(target_names)
    audit = audit_trainable_surface(
        base_parameter_names=list(base_parameter_names),
        adapter_parameter_names=list(adapters),
        trainable_parameter_names=list(adapters),
        mapping=mapping,
        rank=rank,
        scaling=scaling,
        dropout=dropout,
    )
    return TrainingExecutionContract(
        plan=plan,
        model=model,
        mapping=mapping,
        target_names=target_names,
        audit=audit,
        rank=rank,
        scaling=scaling,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        optimizer_step_budget=optimizer_step_budget,
        precision=precision,
        sequence_length=sequence_length,
        seeds=seeds,
        privacy_regime=privacy_regime,
        backend=backend,
        resolved_config_sha256=resolved_config_sha256,
    )


#: How a run's optimiser updates are laid over its own corpus.
CANONICAL_TREATED_SCHEDULE: Final = "CANONICAL_TREATED_SCHEDULE"
MATCHED_CONTROL_CYCLED_SCHEDULE: Final = "MATCHED_CONTROL_CYCLED_SCHEDULE"
SCHEDULE_ROLES: Final[tuple[str, ...]] = (
    CANONICAL_TREATED_SCHEDULE,
    MATCHED_CONTROL_CYCLED_SCHEDULE,
)


def batches_per_epoch(n_records: int, batch_size: int) -> int:
    """The frozen batching convention: a final partial batch is still an update."""
    if n_records < 1:
        raise ExecutionContractError("a training plan carries no records")
    if batch_size < 1:
        raise ExecutionContractError("the batch size must be positive")
    return (n_records + batch_size - 1) // batch_size


def treated_update_budget(*, treated: TrainingPlan, epochs: int, batch_size: int) -> int:
    """The canonical treated schedule's update count [AUTH: 00 §7.3, §7.5].

    Derived from the COMPLETE treated corpus — every natural record plus every included
    canary — so one pass covers all of it. This is the whole point: 00 §7.3 puts each included
    canary in the training corpus, and a canary the schedule never reaches is not a trained
    member, however the plan labelled it.
    """
    if epochs < 1:
        raise ExecutionContractError("epochs must be positive")
    return epochs * batches_per_epoch(treated.n_training_records, batch_size)


def matched_update_budget(*, treated: TrainingPlan, epochs: int, batch_size: int) -> int:
    """The single budget BOTH arms execute, defined by the treated run [AUTH: 00 §7.5].

    The control matches the treated run, never the other way round. Deriving the budget from
    the smaller no-canary corpus truncated the treated run: at 20 natural records plus 13
    included canaries with batch size 4, the control's 5 updates would leave 9 of the 13
    canaries unconsumed while they still counted as trained members.

    The control reaches this budget by deterministically cycling its own natural data order.
    It never gains a canary, loses a natural record, or receives filler.
    """
    return treated_update_budget(treated=treated, epochs=epochs, batch_size=batch_size)


def schedule_role(*, plan: TrainingPlan, budget: int, batch_size: int) -> str:
    """Which schedule a contract describes, derived rather than asserted."""
    if plan.canaries_permitted:
        return CANONICAL_TREATED_SCHEDULE
    return (
        MATCHED_CONTROL_CYCLED_SCHEDULE
        if budget > batches_per_epoch(plan.n_training_records, batch_size)
        else CANONICAL_TREATED_SCHEDULE
    )


def matched_execution_problems(
    treated: TrainingExecutionContract, control: TrainingExecutionContract
) -> list[str]:
    """Every way two arms fail to be a matched execution [AUTH: 00 §7.5].

    Complements `membership.matched_control_problems`, which matches the *plans*. This
    matches what will actually run, including the update budget the 10-vs-8 defect exposed.
    """
    problems: list[str] = []
    if treated.optimizer_step_budget != control.optimizer_step_budget:
        problems.append(
            f"the arms execute different optimiser-update budgets:"
            f" {treated.optimizer_step_budget} against {control.optimizer_step_budget}"
            " [AUTH: 00 §7.5]"
        )
    for name in (
        "rank",
        "scaling",
        "dropout",
        "learning_rate",
        "batch_size",
        "precision",
        "sequence_length",
        "privacy_regime",
    ):
        if getattr(treated, name) != getattr(control, name):
            problems.append(f"the arms differ in {name}")
    if treated.target_names != control.target_names:
        problems.append("the arms adapt different modules")
    if treated.mapping.policy != control.mapping.policy:
        problems.append("the arms use different target policies")
    if treated.model.model_id != control.model.model_id:
        problems.append("the arms use different models")
    if treated.seeds.training_seed != control.seeds.training_seed:
        problems.append("the arms use different training seeds")
    for family in ("data_order", "lora_init", "numpy_rng", "python_rng"):
        if treated.seeds.values.get(family) != control.seeds.values.get(family):
            problems.append(f"the arms use different {family} seeds")
    if treated.resolved_config_sha256 != control.resolved_config_sha256:
        problems.append("the arms resolved different configuration")
    problems.extend(_plan_problems(treated.plan, control.plan))
    return problems


def _plan_problems(treated: TrainingPlan, control: TrainingPlan) -> list[str]:
    from src.data.membership import matched_control_problems

    return matched_control_problems(treated, control)


def matched_outcome_problems(treated: object, control: object) -> list[str]:
    """Every way two EXECUTED arms fail to be a matched comparison [AUTH: 00 §7.5].

    `matched_execution_problems` compares what two contracts authorise; this compares what
    two runs actually did. The distinction is the point: a planned budget both arms agree on
    proves nothing if one of them consumed a different corpus.

    Typed loosely because `src.training.trainer` imports this module, not the other way round.
    """
    problems: list[str] = []
    treated_steps = int(getattr(treated, "optimizer_steps", -1))
    control_steps = int(getattr(control, "optimizer_steps", -2))
    if treated_steps != control_steps:
        problems.append(
            f"the arms executed different optimiser-update counts: {treated_steps} against"
            f" {control_steps} [AUTH: 00 §7.5]"
        )
    treated_natural = tuple(getattr(treated, "natural_ids", ()))
    control_natural = tuple(getattr(control, "natural_ids", ()))
    if treated_natural != control_natural:
        problems.append("the arms trained different natural corpora")
    consumed_treated = frozenset(getattr(treated, "consumed_natural_ids", ()))
    consumed_control = frozenset(getattr(control, "consumed_natural_ids", ()))
    if consumed_treated != consumed_control:
        problems.append("the arms consumed different natural records")
    if tuple(getattr(control, "included_canary_ids", ())):
        problems.append("the no-canary control carries synthetic canaries")
    unconsumed = tuple(getattr(treated, "unconsumed_canary_ids", ()))
    if unconsumed:
        problems.append(
            f"{len(unconsumed)} included canary/canaries were never consumed by the treated"
            f" run, first {unconsumed[0]!r} [AUTH: 00 §7.3]"
        )
    if not tuple(getattr(treated, "included_canary_ids", ())):
        problems.append("the treated arm carries no canaries")
    return problems


def deferred_backend(identity: str, dependency: str) -> BackendState:
    """The honest state of a backend that is not installed."""
    return BackendState(
        identity=identity,
        status=f"{BACKEND_DEPENDENCY_DEFERRED}({dependency})",
        detail=f"{dependency} is not present in the resolved environment",
    )


def backend_problems(backend: BackendState, *, evidentiary: bool) -> list[str]:
    """A dependency-deferred backend may describe a reference run and never an evidentiary one."""
    if not evidentiary:
        return []
    if not backend.ready:
        return [
            f"backend {backend.identity!r} is {backend.status}; an evidentiary run requires a"
            " READY production backend [AUTH: 01 §12, §16]"
        ]
    return []


#: Bound into every contract that names an unimplemented backend, so a reader never has to
#: infer that a reference run is not the production one.
def reference_backend() -> BackendState:
    return deferred_backend("s06.reference-lora-trainer.v1", "TORCH_PEFT_BACKEND")


def reference_dp_backend() -> BackendState:
    return deferred_backend("s06.rdp-sgm-integer-orders.v1", "OPACUS_BACKEND")
