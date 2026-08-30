"""The one canonical production HF loader [AUTH: 01 §8C, §10, §12; 02 §C6].

Loading is split into a **resolved load plan** and the call that executes it. The plan is
plain data: the model id, the pinned revision, the dtype, the attention implementation and
every refusal the policy makes. Every scientific decision is therefore inspectable and
testable without a GPU, a download or even torch installed, and the executing call adds no
policy of its own — it cannot, because there is nothing left for it to decide.

What the policy refuses, and why:

* **a floating or unresolved revision** — a result that cannot be re-derived is not evidence
  [AUTH: 01 §8G];
* **any quantized load** — 8-bit, 4-bit, GPTQ, AWQ or a `quantization_config`. The scientific
  object is the induced update on full-precision weights; a quantized checkpoint measures a
  different model. A checkpoint that *requires* quantization fails compatibility and is never
  silently replaced [AUTH: 01 §8C];
* **an unpermitted dtype** — BF16 or FP32 only, chosen explicitly rather than by an autocast
  default [AUTH: 01 §10];
* **a silent device or dtype fallback** — an unavailable device raises. Degrading quietly to
  CPU or to another dtype would change the numbers while the manifest kept claiming the
  requested profile [AUTH: 01 §12].

torch and transformers are absent from the CPU/dev lane by design, so they are imported
lazily, inside the executing call only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from src.backend.revision import ResolvedModelIdentity, require_evidentiary_identity
from src.provenance.hashing import JSONValue, sha256_canonical

LOADER_SCHEMA: Final = "backend.load-plan.v1"

#: The status a compatibility check reports when a checkpoint cannot be loaded as specified.
QUANTIZATION_REQUIRED: Final = "QUANTIZATION_REQUIRED_NOT_ELIGIBLE"
BACKEND_NOT_INSTALLED: Final = "BACKEND_NOT_INSTALLED"


class LoaderError(RuntimeError):
    """A model cannot be loaded under the frozen production policy."""


class QuantizationRefused(LoaderError):
    """A quantized load was requested or is required by the checkpoint [AUTH: 01 §8C]."""


class BackendUnavailable(LoaderError):
    """transformers/peft are not installed in this environment [AUTH: 02 §C6]."""


def backend_available() -> bool:
    """True when the CUDA-coupled science set is importable [AUTH: 02 §C6]."""
    import importlib.util

    return all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


def require_backend() -> None:
    if not backend_available():
        raise BackendUnavailable(
            f"{BACKEND_NOT_INSTALLED}: torch/transformers are resolved on the H100 image and"
            " are deliberately absent from the CPU/dev lane [AUTH: 01 §12; 02 §C6]"
        )


@dataclass(frozen=True)
class LoadPlan:
    """Exactly what the loader will do. No decision is left to the executing call."""

    model_id: str
    revision: str
    dtype: str
    attn_implementation: str
    trust_remote_code: bool
    local_files_only: bool
    device: str
    architecture_family: str
    refusals: tuple[str, ...] = field(default_factory=tuple)

    @property
    def loadable(self) -> bool:
        return not self.refusals

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": LOADER_SCHEMA,
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
            "device": self.device,
            "architecture_family": self.architecture_family,
            "refusals": list(self.refusals),
        }

    def identity(self) -> str:
        return sha256_canonical(self.as_dict())

    def from_pretrained_kwargs(self) -> dict[str, JSONValue]:
        """The keyword arguments the production call is made with. No quantization key."""
        if not self.loadable:
            raise LoaderError("; ".join(self.refusals))
        return {
            "revision": self.revision,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }


def quantization_refusals(
    model_config: Mapping[str, Any], *, forbidden: tuple[str, ...]
) -> list[str]:
    """Every reason this checkpoint's own config makes it ineligible [AUTH: 01 §8C]."""
    problems: list[str] = []
    for key in forbidden:
        value = model_config.get(key)
        if value in (None, False):
            continue
        problems.append(
            f"{QUANTIZATION_REQUIRED}: the checkpoint config sets {key}={value!r}. The"
            " scientific object is the induced update on full-precision weights, so a"
            " quantized checkpoint measures a different model and is not silently replaced"
            " [AUTH: 01 §8C]"
        )
    return problems


def build_load_plan(
    root: Path,
    identity: ResolvedModelIdentity,
    *,
    model_config: Mapping[str, Any] | None = None,
    device: str = "cuda",
    evidentiary: bool = True,
    precision: str | None = None,
) -> LoadPlan:
    """Resolve the production load policy for one pinned model.

    `model_config` is the checkpoint's own `config.json` when it is available; a checkpoint
    that declares a quantization mode is refused here, before any weight is touched.
    """
    from src.backend.settings import backend_settings
    from src.materials import material, material_text

    settings = backend_settings(root)
    if evidentiary:
        require_evidentiary_identity(identity)

    dtype = precision or material_text(settings, "forward_precision")
    permitted = material(settings, "permitted_precisions")
    forbidden_raw = material(settings, "forbidden_load_options")
    forbidden = tuple(str(key) for key in forbidden_raw) if isinstance(forbidden_raw, list) else ()

    refusals: list[str] = []
    if not isinstance(permitted, list) or dtype not in [str(p) for p in permitted]:
        refusals.append(
            f"dtype {dtype!r} is not a permitted production precision {permitted!r};"
            " precision is resolved explicitly, never left to an autocast default"
            " [AUTH: 01 §10]"
        )
    refusals.extend(quantization_refusals(model_config or {}, forbidden=forbidden))

    return LoadPlan(
        model_id=identity.model_id,
        revision=identity.revision,
        dtype=dtype,
        attn_implementation=material_text(settings, "attn_implementation"),
        trust_remote_code=bool(material(settings, "trust_remote_code")),
        local_files_only=bool(material(settings, "local_files_only")),
        device=device,
        architecture_family=identity.architecture_family,
        refusals=tuple(refusals),
    )


def load_causal_lm(plan: LoadPlan) -> Any:
    """Execute the plan. Adds no policy: everything was decided in `build_load_plan`.

    A requested accelerator that is unavailable raises rather than falling back, so a run
    can never record the H100 profile while having executed on CPU [AUTH: 01 §12].
    """
    require_backend()
    if not plan.loadable:
        raise LoaderError("; ".join(plan.refusals))

    import torch  # noqa: PLC0415
    from transformers import (  # type: ignore[import-not-found] # noqa: PLC0415
        AutoModelForCausalLM,
    )

    if plan.device.startswith("cuda") and not torch.cuda.is_available():
        raise LoaderError(
            f"device {plan.device!r} was requested and no CUDA device is available; refusing"
            " to fall back silently, because the recorded profile would then describe a"
            " computation that did not happen [AUTH: 01 §12]"
        )
    kwargs = plan.from_pretrained_kwargs()
    kwargs["dtype"] = getattr(torch, plan.dtype)
    model = AutoModelForCausalLM.from_pretrained(plan.model_id, **kwargs)
    resolved = str(next(model.parameters()).dtype).removeprefix("torch.")
    if resolved != plan.dtype:
        raise LoaderError(
            f"the loaded model is {resolved}, not the requested {plan.dtype}; a dtype the"
            " loader did not choose is a silent fallback [AUTH: 01 §10, §12]"
        )
    return model.to(plan.device)


def parameter_names(model: Any) -> tuple[str, ...]:
    """The model's parameter names, in declaration order."""
    return tuple(name for name, _ in model.named_parameters())
