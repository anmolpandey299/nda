"""PEFT LoRA configuration, persistence and induced-update extraction [AUTH: 00 §22; 01 §8B].

This is the science-critical seam. Everything downstream — merge, recovery, the whole P0
parameter axis — is defined on the **induced update** ΔW = BA, never on the raw factors: the
factorisation is not unique, so (A, B) and (SA, BS⁻¹) name the same object while comparing as
different matrices [AUTH: 00 §22].

Three properties are enforced here rather than assumed:

* **PEFT's own scaling, applied exactly once.** PEFT stores unscaled factors and scales at
  forward time by `lora_alpha / r` (`use_rslora` is disabled in config, so that is the whole
  rule). The induced update must therefore be `scaling · B · A`. Omitting the factor or
  applying it twice both produce a matrix that is a plausible-looking multiple of the truth,
  which no shape or finiteness check would catch.
* **Exactly one A and one B per target.** A missing, duplicated or unexpected factor is a
  refusal, not a best effort.
* **The declared base is the loaded base.** An adapter whose recorded base model or revision
  differs from the one it is being attached to is rejected before any tensor is read.

Reading and writing use the real PEFT save layout — `adapter_config.json` plus
`adapter_model.safetensors` with `base_model.model.<module>.lora_{A,B}.weight` naming — so the
path exercised on CPU fixtures is the path a real `PeftModel.save_pretrained` produces.

**The adapter's own config is evidence, not authority.** `adapter_config.json` travels with
the adapter and can be edited, so reading LoRA semantics out of it means an edited file can
change alpha, the scaling rule or the target policy and produce a different ΔW while staying
internally hash-consistent. The reader therefore resolves the frozen specification from
version-controlled config and *compares* the declaration against it, field by field; a
mismatch is a refusal, never an override.

**The target surface is checked against the model, not the adapter.** `target_modules` in the
adapter config is equally untrusted, and the tensor keys are what actually enter ΔW. So the
reader takes the canonical `TrunkSelection` derived from the base model's own parameter names
under `LANGUAGE_TRUNK_MLP_ONLY` and requires the adapted parameter set to equal it exactly —
which is what keeps attention projections, the LM head, embeddings and every multimodal tower
out of the induced update [AUTH: 01 §8B; 00 §22].
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from src.provenance.hashing import JSONValue, sha256_canonical, sha256_file

Matrix = NDArray[Any]

ADAPTER_SCHEMA: Final = "backend.peft-adapter.v1"

#: PEFT's peft_type for the LoRA method.
PEFT_TYPE: Final = "LORA"

_LORA_SUFFIX_RE: Final = re.compile(r"^(?P<module>.+)\.lora_(?P<factor>[AB])\.weight$")


class AdapterError(ValueError):
    """A PEFT adapter cannot be configured, persisted or read as a scientific object."""


class ScalingError(AdapterError):
    """The LoRA scaling semantics are not the frozen ones [AUTH: 01 §8B]."""


# ----------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LoraSpecification:
    """Every material LoRA constant, resolved from config [AUTH: 01 §17; 00 §8.3]."""

    rank: int
    lora_alpha: float
    lora_dropout: float
    bias: str
    task_type: str
    target_policy: str
    target_modules: tuple[str, ...]
    architecture_family: str
    architecture_mapping_version: str
    use_rslora: bool
    scaling_rule: str

    @property
    def scaling(self) -> float:
        """PEFT's scaling: alpha / r. `use_rslora` would make it alpha / sqrt(r)."""
        if self.use_rslora:
            raise ScalingError(
                "use_rslora changes the induced update; the frozen rule is ALPHA_OVER_RANK"
            )
        if self.scaling_rule != "ALPHA_OVER_RANK":
            raise ScalingError(f"unsupported LoRA scaling rule {self.scaling_rule!r}")
        return float(self.lora_alpha) / float(self.rank)

    def peft_config_kwargs(self) -> dict[str, JSONValue]:
        """The keyword arguments a real `peft.LoraConfig` is constructed with."""
        return {
            "r": self.rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "target_modules": list(self.target_modules),
            "use_rslora": self.use_rslora,
        }

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": ADAPTER_SCHEMA,
            "peft_type": PEFT_TYPE,
            **self.peft_config_kwargs(),
            "lora_target_policy": self.target_policy,
            "architecture_family": self.architecture_family,
            "architecture_mapping_version": self.architecture_mapping_version,
            "lora_scaling_rule": self.scaling_rule,
            "scaling": self.scaling,
        }

    def identity(self) -> str:
        return sha256_canonical(self.as_dict())


def resolve_lora_specification(
    root: Path, *, architecture_family: str, target_modules: Sequence[str]
) -> LoraSpecification:
    """Build the production LoRA specification from version-controlled config only."""
    from src.backend.settings import backend_settings
    from src.materials import material, material_integer, material_number, material_text
    from src.training.settings import lora_settings

    lora = lora_settings(root)
    backend = backend_settings(root)
    policy = material_text(lora, "target_policy")
    if policy != "LANGUAGE_TRUNK_MLP_ONLY":
        raise AdapterError(
            f"target policy {policy!r} is not the frozen primary policy; the 01 §8B.1 fallback"
            " is gated on the P0-A measurement-power condition and is not activated here"
        )
    rank = material_integer(lora, "adapter_rank")
    if rank != 32:
        raise AdapterError(f"the frozen LoRA rank is 32, not {rank}")
    #: Resolved through the same fail-closed accessor as every other scaling-affecting value.
    #: Raw document access would let a missing or REQUIRED_NOT_CALIBRATED entry become False,
    #: and `use_rslora` changes the scaling semantics outright.
    use_rslora = material(backend, "use_rslora")
    if not isinstance(use_rslora, bool):
        raise AdapterError(
            f"use_rslora resolved to {use_rslora!r}, which is not a boolean; it changes the"
            " LoRA scaling rule, so an unresolved value fails closed [AUTH: 01 §17]"
        )
    return LoraSpecification(
        rank=rank,
        lora_alpha=material_number(lora, "adapter_scaling", allow_provisional=True),
        lora_dropout=material_number(lora, "adapter_dropout", allow_provisional=True),
        bias=material_text(backend, "peft_bias_policy"),
        task_type=material_text(backend, "peft_task_type"),
        target_policy=policy,
        target_modules=tuple(target_modules),
        architecture_family=architecture_family,
        architecture_mapping_version=material_text(backend, "architecture_mapping_version"),
        use_rslora=use_rslora,
        scaling_rule=material_text(backend, "lora_scaling_rule"),
    )


# ----------------------------------------------------------------------------------------
# persistence in the real PEFT layout
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PersistedAdapter:
    """One saved PEFT adapter directory, read back as scientific objects."""

    directory: Path
    specification: LoraSpecification
    base_model_id: str
    base_revision: str
    factors: Mapping[str, tuple[Matrix, Matrix]]
    config_sha256: str
    weights_sha256: str

    @property
    def target_parameters(self) -> tuple[str, ...]:
        return tuple(sorted(self.factors))

    def induced_update(self, name: str) -> Matrix:
        """ΔW = scaling · B · A, in float64. The scaling is applied here, exactly once."""
        a, b = self.factors[name]
        return float(self.specification.scaling) * (b @ a)

    def induced_updates(self) -> dict[str, Matrix]:
        return {name: self.induced_update(name) for name in self.target_parameters}

    def update_identity(self) -> str:
        """Identity over the induced updates, matching the accepted S06 convention."""
        import hashlib

        digest = hashlib.sha256()
        for name, delta in self.induced_updates().items():
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(delta, dtype="<f8").tobytes(order="C"))
        return digest.hexdigest()

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": ADAPTER_SCHEMA,
            "base_model_id": self.base_model_id,
            "base_revision": self.base_revision,
            "adapter_config_sha256": self.config_sha256,
            "adapter_weights_sha256": self.weights_sha256,
            "target_parameters": list(self.target_parameters),
            "induced_update_sha256": self.update_identity(),
            "specification": self.specification.as_dict(),
        }


def _peft_tensor_name(prefix: str, module: str, factor: str) -> str:
    return f"{prefix}{module}.lora_{factor}.weight"


def save_adapter_directory(
    directory: Path,
    *,
    specification: LoraSpecification,
    factors: Mapping[str, tuple[Matrix, Matrix]],
    base_model_id: str,
    base_revision: str,
    expected_selection: Sequence[str],
    module_prefix: str,
    weight_filename: str,
    config_filename: str,
) -> PersistedAdapter:
    """Write the real PEFT save layout: `adapter_config.json` + safetensors weights.

    The base model id and revision are written into the adapter config, which is what makes
    an adapter re-attachable to the wrong base detectable later. `expected_selection` is
    required rather than defaulted from the factors: a default derived from what is being
    written would make the target-surface check circular, and the point of the check is that
    the model decides the surface, not the adapter.

    The written adapter is read straight back through the validating reader, so anything the
    writer produced that the reader would refuse is refused here rather than downstream.
    """
    if not factors:
        raise AdapterError("an adapter covering no target parameter is not an adapter")
    directory.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, Matrix] = {}
    for module, (a, b) in sorted(factors.items()):
        _check_factor_pair(module, a, b, rank=specification.rank)
        tensors[_peft_tensor_name(module_prefix, module, "A")] = np.ascontiguousarray(
            a, dtype=np.float32
        )
        tensors[_peft_tensor_name(module_prefix, module, "B")] = np.ascontiguousarray(
            b, dtype=np.float32
        )

    from safetensors.numpy import save_file

    weights_path = directory / weight_filename
    save_file(tensors, str(weights_path))

    document: dict[str, JSONValue] = {
        "peft_type": PEFT_TYPE,
        "base_model_name_or_path": base_model_id,
        "revision": base_revision,
        **specification.peft_config_kwargs(),
        "lora_target_policy": specification.target_policy,
        "architecture_family": specification.architecture_family,
        "architecture_mapping_version": specification.architecture_mapping_version,
        "lora_scaling_rule": specification.scaling_rule,
    }
    config_path = directory / config_filename
    config_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return read_adapter_directory(
        directory,
        frozen_specification=specification,
        expected_selection=expected_selection,
        expected_base_model_id=base_model_id,
        expected_base_revision=base_revision,
        module_prefix=module_prefix,
        weight_filename=weight_filename,
        config_filename=config_filename,
    )


def _check_factor_pair(module: str, a: Matrix, b: Matrix, *, rank: int) -> None:
    """Shape, rank and finiteness, before anything is written or multiplied."""
    left = np.asarray(a)
    right = np.asarray(b)
    if left.ndim != 2 or right.ndim != 2:
        raise AdapterError(f"{module}: LoRA factors must be 2-D")
    if left.shape[0] != rank or right.shape[1] != rank:
        raise AdapterError(
            f"{module}: factor shapes {right.shape} x {left.shape} do not carry rank {rank};"
            " PEFT stores A as (r, in) and B as (out, r)"
        )
    if right.shape[1] != left.shape[0]:  # pragma: no cover - implied by the rank check
        raise AdapterError(f"{module}: B and A do not compose")
    for label, array in (("A", left), ("B", right)):
        if not np.all(np.isfinite(array)):
            raise AdapterError(
                f"{module}: LoRA factor {label} carries NaN or Inf; a non-finite adapter is"
                " not a parameter object [AUTH: 00 §22]"
            )


#: Every material LoRA field the adapter declares and the frozen specification owns. An
#: adapter may not differ on any of them [AUTH: 01 §8B, §17].
VALIDATED_SPECIFICATION_FIELDS: Final[tuple[str, ...]] = (
    "r",
    "lora_alpha",
    "lora_dropout",
    "bias",
    "task_type",
    "lora_target_policy",
    "lora_scaling_rule",
    "use_rslora",
    "architecture_mapping_version",
)


def _validate_declared_specification(
    document: Mapping[str, Any], frozen: LoraSpecification
) -> list[str]:
    """Every field on which the adapter's declaration differs from the frozen one."""
    declared: dict[str, Any] = {
        "r": document.get("r"),
        "lora_alpha": document.get("lora_alpha"),
        "lora_dropout": document.get("lora_dropout"),
        "bias": document.get("bias"),
        "task_type": document.get("task_type"),
        "lora_target_policy": document.get("lora_target_policy"),
        "lora_scaling_rule": document.get("lora_scaling_rule"),
        "use_rslora": document.get("use_rslora"),
        "architecture_mapping_version": document.get("architecture_mapping_version"),
    }
    expected: dict[str, Any] = {
        "r": frozen.rank,
        "lora_alpha": frozen.lora_alpha,
        "lora_dropout": frozen.lora_dropout,
        "bias": frozen.bias,
        "task_type": frozen.task_type,
        "lora_target_policy": frozen.target_policy,
        "lora_scaling_rule": frozen.scaling_rule,
        "use_rslora": frozen.use_rslora,
        "architecture_mapping_version": frozen.architecture_mapping_version,
    }
    problems: list[str] = []
    for field in VALIDATED_SPECIFICATION_FIELDS:
        left, right = declared[field], expected[field]
        if isinstance(right, float) and isinstance(left, int | float):
            matches = float(left) == float(right)
        else:
            matches = left == right
        if not matches:
            problems.append(f"{field}: adapter declares {left!r}, the frozen value is {right!r}")
    return problems


#: A LoRA-adapted module has exactly one weight parameter, so the module path the PEFT tensor
#: keys carry corresponds to `<module>.weight` in the model's parameter names.
ADAPTED_PARAMETER_SUFFIX: Final = ".weight"


def adapted_parameter_name(module: str) -> str:
    """The base parameter a LoRA-adapted module wraps."""
    return f"{module}{ADAPTED_PARAMETER_SUFFIX}"


def _validate_target_surface(adapted: Sequence[str], expected: Sequence[str]) -> list[str]:
    """Exact correspondence between the adapted parameters and the canonical selection.

    Compared as parameter names, because that is what the model-side selection speaks and what
    actually receives ΔW.
    """
    actual = [adapted_parameter_name(module) for module in adapted]
    duplicates = sorted({name for name in actual if actual.count(name) > 1})
    problems: list[str] = []
    if duplicates:
        problems.append(f"duplicate target parameter(s): {duplicates[:3]}")
    extra = sorted(set(actual) - set(expected))
    if extra:
        problems.append(
            f"the adapter adapts parameter(s) outside the language-trunk MLP surface: {extra[:3]}"
        )
    missing = sorted(set(expected) - set(actual))
    if missing:
        problems.append(f"the adapter omits expected target parameter(s): {missing[:3]}")
    return problems


def read_adapter_directory(
    directory: Path,
    *,
    frozen_specification: LoraSpecification,
    expected_selection: Sequence[str],
    expected_base_model_id: str | None = None,
    expected_base_revision: str | None = None,
    module_prefix: str,
    weight_filename: str,
    config_filename: str,
) -> PersistedAdapter:
    """Read a PEFT adapter directory back into scientific objects, or refuse it.

    `frozen_specification` is resolved from version-controlled config by the caller and is the
    authority: the adapter's own declaration is compared against it and may not override it.
    `expected_selection` is the canonical target-parameter set derived from the BASE MODEL's
    parameter names, so what the adapter claims about its targets is irrelevant — the tensor
    keys must match the model-side selection exactly.

    The declared base model and revision are checked against what the caller is attaching to,
    so an adapter trained on one checkpoint cannot be silently read against another.
    """
    config_path = directory / config_filename
    weights_path = directory / weight_filename
    for path in (config_path, weights_path):
        if not path.is_file():
            raise AdapterError(f"the adapter directory has no {path.name}")

    document = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AdapterError(f"{config_filename} is not a JSON object")
    if str(document.get("peft_type")) != PEFT_TYPE:
        raise AdapterError(f"{config_filename} declares peft_type {document.get('peft_type')!r}")

    declared_base = str(document.get("base_model_name_or_path", ""))
    declared_revision = str(document.get("revision", ""))
    if expected_base_model_id is not None and declared_base != expected_base_model_id:
        raise AdapterError(
            f"the adapter declares base {declared_base!r} but is being read against"
            f" {expected_base_model_id!r}; an adapter is only meaningful on the base it was"
            " trained on [AUTH: 00 §22; 01 §8G]"
        )
    if expected_base_revision is not None and declared_revision != expected_base_revision:
        raise AdapterError(
            f"the adapter declares base revision {declared_revision!r} but is being read"
            f" against {expected_base_revision!r} [AUTH: 01 §8G]"
        )

    drift = _validate_declared_specification(document, frozen_specification)
    if drift:
        raise AdapterError(
            "the adapter's declared LoRA configuration differs from the frozen specification,"
            " and an adapter may not redefine scientific semantics: "
            + "; ".join(drift)
            + " [AUTH: 01 §8B, §17; 00 §22]"
        )

    from safetensors.numpy import load_file

    tensors = load_file(str(weights_path))
    factors = _pair_factors(tensors, module_prefix=module_prefix, rank=frozen_specification.rank)
    surface = _validate_target_surface(sorted(factors), expected_selection)
    if surface:
        raise AdapterError(
            "the adapter's target surface is not the canonical language-trunk MLP selection: "
            + "; ".join(surface)
            + ". The persisted tensor keys decide what enters ΔW, not the adapter's own"
            " target_modules [AUTH: 01 §8B; 00 §22]"
        )
    return PersistedAdapter(
        directory=directory,
        specification=frozen_specification,
        base_model_id=declared_base,
        base_revision=declared_revision,
        factors=factors,
        config_sha256=sha256_file(config_path),
        weights_sha256=sha256_file(weights_path),
    )


def _pair_factors(
    tensors: Mapping[str, Matrix], *, module_prefix: str, rank: int
) -> dict[str, tuple[Matrix, Matrix]]:
    """Pair each module's A and B. Missing, duplicated or unexpected tensors are refused."""
    collected: dict[str, dict[str, Matrix]] = {}
    for name, array in tensors.items():
        if not name.startswith(module_prefix):
            raise AdapterError(
                f"{name!r} does not carry the PEFT module prefix {module_prefix!r}; the"
                " adapter layout is not the one this backend writes"
            )
        stripped = name[len(module_prefix) :]
        match = _LORA_SUFFIX_RE.match(stripped)
        if match is None:
            raise AdapterError(
                f"{name!r} is not a LoRA factor tensor; an adapter carrying an unexpected"
                " tensor is refused rather than partially read"
            )
        module = match.group("module")
        factor = match.group("factor")
        slot = collected.setdefault(module, {})
        if factor in slot:  # pragma: no cover - safetensors keys are unique
            raise AdapterError(f"{module}: duplicate lora_{factor} tensor")
        slot[factor] = np.asarray(array, dtype=np.float64)

    factors: dict[str, tuple[Matrix, Matrix]] = {}
    for module, slot in sorted(collected.items()):
        missing = sorted({"A", "B"} - set(slot))
        if missing:
            raise AdapterError(
                f"{module}: the adapter is missing lora_{missing[0]}; ΔW = BA needs exactly one"
                " A and one B per target"
            )
        a, b = slot["A"], slot["B"]
        _check_factor_pair(module, a, b, rank=rank)
        factors[module] = (a, b)
    if not factors:
        raise AdapterError("the adapter carries no LoRA factors")
    return factors


def induced_update_from_factors(a: Matrix, b: Matrix, *, scaling: float) -> Matrix:
    """ΔW = scaling · B · A. The one place the scaling is applied [AUTH: 00 §22]."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if right.shape[1] != left.shape[0]:
        raise AdapterError(f"B {right.shape} and A {left.shape} do not compose")
    delta = float(scaling) * (right @ left)
    if not np.all(np.isfinite(delta)):
        raise AdapterError("the induced update carries NaN or Inf [AUTH: 00 §22]")
    return delta
