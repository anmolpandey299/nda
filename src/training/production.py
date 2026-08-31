"""The production LoRA training loop [AUTH: 00 §8.1, §8.3, §9; 01 §8B, §12, §17, §30].

S06 froze the training *contract* — what one run is authorised to adapt, with which rank,
schedule, seed families and privacy regime — and shipped a reference trainer over numeric
fixtures. This is the executor that contract was written for: it takes an authorised
`TrainingExecutionContract` and runs it against a real Hugging Face causal LM with real PEFT
adapters, then persists the result through the accepted adapter-persistence path.

It adds no policy. Every material constant is read from `configs/training/**` through
`src.materials`, which refuses an uncalibrated constant rather than defaulting it — so a run
that would have invented a learning rate fails closed here instead of producing a plausible
adapter [AUTH: 01 §17].

The split matters for testability and is deliberate:

* `plan_production_training` is pure. It resolves the frozen constants, derives the batch
  schedule and the RNG streams, and decides everything a run will do — on plain data, with no
  torch import. It is fully exercised on the CPU/dev lane.
* `train_production_lora` executes that plan. It is the only function here that imports torch,
  transformers or peft, and it refuses immediately when they are absent [AUTH: 02 §C6].

Nothing in this module chooses a scientific value, and nothing writes a run manifest: the
caller owns the S01 lifecycle, so one training job is one claimed attempt and no more.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from src.backend.adapters import LoraSpecification, PersistedAdapter, save_adapter_directory
from src.backend.loader import require_backend
from src.data.membership import TrainingPlan
from src.provenance.hashing import JSONValue, sha256_canonical
from src.training.execution import DP_SAMPLE_LEVEL, TrainingExecutionContract
from src.training.seeds import SeedFamilies

TRAINER_VERSION: Final = "s06.production-lora-trainer.v1"

#: 00 §8.3's data-order policy. The only one this executor implements; a different frozen
#: policy must be implemented rather than approximated.
SHUFFLE_ONCE_PER_EPOCH: Final = "SHUFFLE_ONCE_PER_EPOCH_FROM_DATA_ORDER_SEED"

#: 00 §8.3's checkpoint policy: the final adapter only. No intermediate checkpoint is written,
#: so no "best" checkpoint can be selected after seeing an outcome [AUTH: 00 §34B.1A].
FINAL_ADAPTER_ONLY: Final = "FINAL_ADAPTER_ONLY_NO_INTERMEDIATE_SELECTION"


class ProductionTrainingError(RuntimeError):
    """A production training run cannot be planned or executed as specified."""


@dataclass(frozen=True)
class ResolvedTrainingConstants:
    """The frozen optimisation constants, read from config and never defaulted."""

    optimizer: str
    learning_rate: float
    epochs: int
    batch_size: int
    precision: str
    sequence_length: int
    gradient_accumulation_steps: int
    gradient_clipping_norm: float
    data_order_policy: str
    checkpoint_policy: str

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "precision": self.precision,
            "sequence_length": self.sequence_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping_norm": self.gradient_clipping_norm,
            "data_order_policy": self.data_order_policy,
            "checkpoint_policy": self.checkpoint_policy,
        }


def resolve_training_constants(root: Path) -> ResolvedTrainingConstants:
    """Read every material training constant [AUTH: 00 §8.3; 01 §17].

    `UncalibratedConstantError` propagates deliberately: an unfrozen learning rate is a
    scientific gap, and substituting a fixture value here would hide it behind a run that
    looks complete.
    """
    from src.materials import material_integer, material_number, material_text
    from src.training.settings import lora_settings

    settings = lora_settings(root)
    return ResolvedTrainingConstants(
        optimizer=material_text(settings, "optimizer"),
        learning_rate=material_number(settings, "learning_rate"),
        epochs=material_integer(settings, "epochs"),
        batch_size=material_integer(settings, "batch_size"),
        precision=material_text(settings, "precision"),
        sequence_length=material_integer(settings, "sequence_length"),
        gradient_accumulation_steps=material_integer(settings, "gradient_accumulation_steps"),
        gradient_clipping_norm=material_number(settings, "gradient_clipping_norm"),
        data_order_policy=material_text(settings, "data_order_policy"),
        checkpoint_policy=material_text(settings, "checkpoint_policy"),
    )


@dataclass(frozen=True)
class ResolvedDPConstants:
    """The frozen DP constants. Read only for a DP run [AUTH: 00 §8.2, §8.3]."""

    clipping_norm: float
    noise_multiplier: float
    delta: float
    target_epsilon: float
    adjacency: str

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "clipping_norm": self.clipping_norm,
            "noise_multiplier": self.noise_multiplier,
            "delta": self.delta,
            "target_epsilon": self.target_epsilon,
            "adjacency": self.adjacency,
        }


def resolve_dp_constants(root: Path) -> ResolvedDPConstants:
    """Read the DP constants. Fails closed while any of them is uncalibrated."""
    from src.materials import material_number, material_text
    from src.training.settings import dp_settings

    settings = dp_settings(root)
    return ResolvedDPConstants(
        clipping_norm=material_number(settings, "clipping_norm"),
        noise_multiplier=material_number(settings, "noise_multiplier"),
        delta=material_number(settings, "delta"),
        target_epsilon=material_number(settings, "target_epsilon"),
        adjacency=material_text(settings, "adjacency"),
    )


def epoch_order(record_ids: Sequence[str], *, data_order_seed: int, epoch: int) -> tuple[str, ...]:
    """One epoch's record order under `SHUFFLE_ONCE_PER_EPOCH_FROM_DATA_ORDER_SEED`.

    The permutation is a pure function of (data-order seed, epoch), so a resumed or repeated
    run sees the same order without carrying RNG state across process boundaries.
    """
    import numpy as np

    stream = np.random.Generator(np.random.PCG64([data_order_seed, epoch]))
    order = list(record_ids)
    permutation = stream.permutation(len(order))
    return tuple(order[int(index)] for index in permutation)


@dataclass(frozen=True)
class ProductionTrainingPlan:
    """Everything one production run will do, decided before anything executes."""

    contract: TrainingExecutionContract
    constants: ResolvedTrainingConstants
    dp: ResolvedDPConstants | None
    record_ids: tuple[str, ...]
    batches_per_epoch: int
    optimizer_steps: int
    trainer_version: str = TRAINER_VERSION

    @property
    def differentially_private(self) -> bool:
        return self.contract.privacy_regime == DP_SAMPLE_LEVEL

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "trainer_version": self.trainer_version,
            "contract": self.contract.identity(),
            "constants": self.constants.as_dict(),
            "dp": None if self.dp is None else self.dp.as_dict(),
            "n_records": len(self.record_ids),
            "batches_per_epoch": self.batches_per_epoch,
            "optimizer_steps": self.optimizer_steps,
            "differentially_private": self.differentially_private,
        }

    def identity(self) -> str:
        return sha256_canonical(self.as_dict())


def plan_production_training(
    root: Path, contract: TrainingExecutionContract, *, record_ids: Sequence[str]
) -> ProductionTrainingPlan:
    """Decide the whole run on plain data. Pure: imports no backend and trains nothing."""
    from src.training.execution import batches_per_epoch

    constants = resolve_training_constants(root)
    if constants.data_order_policy != SHUFFLE_ONCE_PER_EPOCH:
        raise ProductionTrainingError(
            f"the frozen data-order policy is {constants.data_order_policy!r}; this executor"
            f" implements {SHUFFLE_ONCE_PER_EPOCH!r} and does not approximate another"
        )
    if constants.checkpoint_policy != FINAL_ADAPTER_ONLY:
        raise ProductionTrainingError(
            f"the frozen checkpoint policy is {constants.checkpoint_policy!r}; this executor"
            " writes the final adapter only [AUTH: 00 §8.3]"
        )
    if not record_ids:
        raise ProductionTrainingError("a training run needs at least one record")
    if len(set(record_ids)) != len(record_ids):
        raise ProductionTrainingError("the training corpus carries a duplicate record id")
    if constants.batch_size != contract.batch_size:
        raise ProductionTrainingError(
            f"the contract authorises batch size {contract.batch_size} and the frozen config"
            f" says {constants.batch_size}; the two must agree before anything runs"
        )

    per_epoch = batches_per_epoch(len(record_ids), constants.batch_size)
    dp = resolve_dp_constants(root) if contract.privacy_regime == DP_SAMPLE_LEVEL else None
    return ProductionTrainingPlan(
        contract=contract,
        constants=constants,
        dp=dp,
        record_ids=tuple(record_ids),
        batches_per_epoch=per_epoch,
        optimizer_steps=contract.optimizer_step_budget,
    )


@dataclass(frozen=True)
class ProductionTrainingOutcome:
    """What one executed run produced. Hash-bound, so it can be verified later."""

    plan_sha256: str
    adapter: PersistedAdapter
    optimizer_steps_run: int
    final_loss: float
    seeds: Mapping[str, int]
    trainer_version: str = TRAINER_VERSION

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "trainer_version": self.trainer_version,
            "plan_sha256": self.plan_sha256,
            "adapter_sha256": self.adapter.update_identity(),
            "optimizer_steps_run": self.optimizer_steps_run,
            "final_loss": self.final_loss,
            "seeds": dict(self.seeds),
        }


def _seed_everything(seeds: SeedFamilies) -> None:
    """Set every RNG stream 01 §30 records, from its own recorded family."""
    import random

    import numpy as np
    import torch

    random.seed(seeds["python_rng"])
    np.random.seed(seeds["numpy_rng"])
    torch.manual_seed(seeds["torch_cpu_rng"])
    if torch.cuda.is_available():  # pragma: no cover - requires the H100 image
        torch.cuda.manual_seed_all(seeds["torch_cuda_rng"])


def _lora_specification(root: Path, contract: TrainingExecutionContract) -> LoraSpecification:
    """The PEFT specification, from the accepted factory [AUTH: PRE_S09 §8].

    `resolve_lora_specification` is the one builder that reads the frozen LoRA constants and
    refuses a policy or rank the spec did not freeze. A second builder here would be a second
    answer to "what adapter was trained".
    """
    from src.backend.adapters import resolve_lora_specification

    return resolve_lora_specification(
        root,
        architecture_family=contract.model.architecture_family,
        target_modules=contract.mapping.module_names(),
    )


def train_production_lora(
    root: Path,
    plan: ProductionTrainingPlan,
    *,
    corpus: Mapping[str, Sequence[int]],
    adapter_directory: Path,
    training_plan: TrainingPlan,
) -> ProductionTrainingOutcome:
    """Execute one authorised training run against the real backend.

    `corpus` maps record id to that record's token ids, already tokenized under the frozen
    tokenizer identity — tokenization is the data block's job, not the trainer's, so the same
    ids that were hashed into the corpus manifest are the ids that train.

    Refuses immediately without torch/transformers/peft rather than emulating them: an adapter
    produced by a stand-in would be indistinguishable from a real one afterwards.
    """
    require_backend()
    _require_peft()

    import torch

    contract = plan.contract
    _seed_everything(contract.seeds)
    missing = [record for record in plan.record_ids if record not in corpus]
    if missing:
        raise ProductionTrainingError(
            f"{len(missing)} planned record(s) are absent from the tokenized corpus,"
            f" first {missing[0]!r}"
        )
    if training_plan.training_seed != contract.seeds.training_seed:
        raise ProductionTrainingError(
            f"the corpus plan is seed {training_plan.training_seed} and the contract is seed"
            f" {contract.seeds.training_seed} [AUTH: 00 §7.2; 01 §30]"
        )

    model = _attach_adapter(root, plan)
    if plan.differentially_private:
        from src.backend.dp import attach_grad_sample_module, build_dp_plan
        from src.dp.mechanism import DPMechanism, poisson_sample_rate, steps_for

        if plan.dp is None:
            raise ProductionTrainingError("a DP run reached execution with no DP constants")
        size = len(plan.record_ids)
        mechanism = DPMechanism(
            adjacency=plan.dp.adjacency,
            delta=plan.dp.delta,
            clipping_norm=plan.dp.clipping_norm,
            noise_multiplier=plan.dp.noise_multiplier,
            sample_rate=poisson_sample_rate(
                batch_size=plan.constants.batch_size, dataset_size=size
            ),
            steps=steps_for(
                epochs=plan.constants.epochs,
                dataset_size=size,
                batch_size=plan.constants.batch_size,
            ),
            dp_seed=contract.seeds["dp_noise"],
            requested_epsilon=plan.dp.target_epsilon,
        )
        model = attach_grad_sample_module(
            model, build_dp_plan(root, mechanism, model_alias=contract.model.alias)
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = _build_optimizer(plan.constants, trainable)
    device = next(model.parameters()).device

    steps = 0
    final_loss = float("nan")
    order_seed = contract.seeds["data_order"]
    model.train()
    for epoch in range(plan.constants.epochs):
        if steps >= plan.optimizer_steps:
            break
        order = epoch_order(plan.record_ids, data_order_seed=order_seed, epoch=epoch)
        for start in range(0, len(order), plan.constants.batch_size):
            if steps >= plan.optimizer_steps:
                break
            batch = order[start : start + plan.constants.batch_size]
            loss = _batch_loss(model, corpus, batch, device=device, torch_module=torch)
            (loss / plan.constants.gradient_accumulation_steps).backward()
            if (steps + 1) % plan.constants.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, plan.constants.gradient_clipping_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            final_loss = float(loss.detach().item())
            steps += 1

    adapter = _persist_adapter(root, plan, model, adapter_directory)
    return ProductionTrainingOutcome(
        plan_sha256=plan.identity(),
        adapter=adapter,
        optimizer_steps_run=steps,
        final_loss=final_loss,
        seeds=contract.seeds.as_manifest_seeds(),
    )


def peft_available() -> bool:
    """True when PEFT resolves. It belongs to the CUDA-coupled science set [AUTH: 02 §C6]."""
    import importlib.util

    return importlib.util.find_spec("peft") is not None


def _require_peft() -> None:
    if not peft_available():
        raise ProductionTrainingError(
            "BACKEND_NOT_INSTALLED: peft is resolved on the H100 image and is deliberately"
            " absent from the CPU/dev lane [AUTH: 01 §12; 02 §C6]"
        )


def _attach_adapter(root: Path, plan: ProductionTrainingPlan) -> Any:  # pragma: no cover
    """Load the frozen base checkpoint and attach a fresh LoRA adapter to it."""
    from peft import LoraConfig, get_peft_model

    from src.backend.loader import build_load_plan, load_causal_lm

    contract = plan.contract
    specification = _lora_specification(root, contract)
    identity = _model_identity(contract)
    base = load_causal_lm(build_load_plan(root, identity))
    configuration = LoraConfig(
        r=specification.rank,
        lora_alpha=specification.lora_alpha,
        lora_dropout=specification.lora_dropout,
        bias=specification.bias,
        task_type=specification.task_type,
        target_modules=list(specification.target_modules),
        use_rslora=specification.use_rslora,
    )
    return get_peft_model(base, configuration)


def _model_identity(contract: TrainingExecutionContract) -> Any:  # pragma: no cover
    """The frozen model identity for the contract's panel member.

    `evidentiary=True` refuses a floating or unresolved revision, so a production adapter can
    never be trained against `main` [AUTH: 01 §8G].
    """
    from src.backend.revision import resolve_model_identity

    return resolve_model_identity(contract.model.as_dict(), evidentiary=True)


def _build_optimizer(
    constants: ResolvedTrainingConstants, parameters: Sequence[Any]
) -> Any:  # pragma: no cover
    """The frozen optimiser. An unrecognised name is refused, never silently substituted."""
    import torch

    name = constants.optimizer.strip().lower()
    if name in ("adamw", "torch.adamw"):
        return torch.optim.AdamW(parameters, lr=constants.learning_rate)
    if name in ("sgd", "torch.sgd"):
        return torch.optim.SGD(parameters, lr=constants.learning_rate)
    raise ProductionTrainingError(
        f"the frozen optimizer {constants.optimizer!r} is not one this executor implements;"
        " add it deliberately rather than falling back to AdamW"
    )


def _batch_loss(
    model: Any,
    corpus: Mapping[str, Sequence[int]],
    batch: Sequence[str],
    *,
    device: Any,
    torch_module: Any,
) -> Any:  # pragma: no cover - requires the H100 image
    """Causal-LM loss over one batch, with right padding and ignored pad targets."""
    width = max(len(corpus[record]) for record in batch)
    ids = [list(corpus[record]) + [0] * (width - len(corpus[record])) for record in batch]
    mask = [[1] * len(corpus[record]) + [0] * (width - len(corpus[record])) for record in batch]
    input_ids = torch_module.tensor(ids, dtype=torch_module.long, device=device)
    attention = torch_module.tensor(mask, dtype=torch_module.long, device=device)
    labels = input_ids.masked_fill(attention == 0, -100)
    return model(input_ids=input_ids, attention_mask=attention, labels=labels).loss


def _persist_adapter(
    root: Path, plan: ProductionTrainingPlan, model: Any, directory: Path
) -> PersistedAdapter:  # pragma: no cover - requires the H100 image
    """Write the trained factors through the accepted persistence path.

    The adapter is saved by the same writer the compatibility contract validates, so a
    production adapter and a fixture adapter are the same artifact kind [AUTH: PRE_S09 §8].
    """
    import numpy as np

    from src.backend.settings import backend_settings
    from src.materials import material_text

    contract = plan.contract
    specification = _lora_specification(root, contract)
    factors: dict[str, tuple[Any, Any]] = {}
    for module in specification.target_modules:
        a = _named_tensor(model, module, "lora_A")
        b = _named_tensor(model, module, "lora_B")
        factors[module] = (
            np.asarray(a.detach().to("cpu"), dtype=np.float32),
            np.asarray(b.detach().to("cpu"), dtype=np.float32),
        )
    settings = backend_settings(root)
    return save_adapter_directory(
        directory,
        specification=specification,
        factors=factors,
        base_model_id=contract.model.model_id,
        base_revision=contract.model.revision,
        expected_selection=list(contract.target_names),
        module_prefix=material_text(settings, "peft_module_prefix"),
        weight_filename=material_text(settings, "adapter_weight_filename"),
        config_filename=material_text(settings, "adapter_config_filename"),
    )


def _named_tensor(model: Any, module: str, factor: str) -> Any:  # pragma: no cover
    """Find one LoRA factor on the attached model by its PEFT parameter name."""
    suffix = f".{module}.{factor}.default.weight"
    for name, parameter in model.named_parameters():
        if name.endswith(suffix):
            return parameter
    raise ProductionTrainingError(
        f"the attached adapter exposes no {factor} factor for target module {module!r}"
    )
