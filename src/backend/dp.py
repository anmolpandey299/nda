"""Production DP plumbing [AUTH: 00 §8.2; 01 §8E; 02 §C6].

The DP *methodology* is already accepted: `src.dp.mechanism` owns the adjacency, the RDP
accountant, the clipping rule, the noise draw and the inferential-set contract. Nothing here
re-implements any of it. This module is the compatibility layer that lets an Opacus
`GradSampleModule` execute that already-frozen contract on a real model, and it fails closed
rather than substituting a second DP story.

Two refusals matter:

* **no second accountant.** The epsilon a run reports comes from `src.dp.mechanism.account`,
  never from Opacus's own accountant. Two accountants on one run means the recorded epsilon
  and the executed mechanism can disagree, and the manifest would still look complete.
* **no adjacency change.** 00 §8.2 fixes sample-level add/remove-one-record adjacency. Opacus
  defaults are not permitted to redefine it.

No DP training runs here. The hook is wired and tested against tiny local modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from src.dp.mechanism import (
    ACCOUNTANT_VERSION,
    SAMPLE_LEVEL_ADJACENCY,
    UNSUPPORTED_ADJACENCY,
    DPAccounting,
    DPMechanism,
    account,
)
from src.provenance.hashing import JSONValue, sha256_canonical

DP_HOOK_SCHEMA: Final = "backend.dp-hook.v1"

#: What the hook reports until a real Opacus run has executed [AUTH: 02 §C6].
DP_BACKEND_NOT_RUN: Final = "NOT_RUN_DEPENDENCY(OPACUS_BACKEND)"


class DPBackendError(RuntimeError):
    """The production DP path cannot execute the frozen contract."""


def opacus_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("opacus") is not None


@dataclass(frozen=True)
class DPBackendPlan:
    """What the DP engine will be configured with, derived from the frozen mechanism.

    Every field is read off the accepted `DPMechanism`; none is a second DP parameter that a
    caller could set independently of the accountant.
    """

    model_alias: str
    engine: str
    adjacency: str
    max_grad_norm: float
    noise_multiplier: float
    sample_rate: float
    steps: int
    dp_seed: int
    accountant_version: str
    backend_status: str

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": DP_HOOK_SCHEMA,
            "model_alias": self.model_alias,
            "engine": self.engine,
            "adjacency": self.adjacency,
            "max_grad_norm": self.max_grad_norm,
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
            "steps": self.steps,
            "dp_seed": self.dp_seed,
            "accountant_version": self.accountant_version,
            "dp_backend_status": self.backend_status,
        }

    def identity(self) -> str:
        return sha256_canonical(self.as_dict())


def build_dp_plan(root: Path, mechanism: DPMechanism, *, model_alias: str) -> DPBackendPlan:
    """Configure the production DP engine from the already-accepted mechanism.

    The mechanism is the single source: the clipping norm, noise multiplier, sample rate and
    step count are read from it, so the engine cannot be pointed at different values than the
    accountant charged for.
    """
    from src.backend.settings import backend_settings
    from src.materials import material_text

    settings = backend_settings(root)
    if mechanism.adjacency in UNSUPPORTED_ADJACENCY:
        raise DPBackendError(
            f"adjacency {mechanism.adjacency!r} is not supported; 00 §8.2 fixes"
            f" {SAMPLE_LEVEL_ADJACENCY} and the backend does not redefine it"
        )
    if mechanism.adjacency != SAMPLE_LEVEL_ADJACENCY:
        raise DPBackendError(
            f"adjacency {mechanism.adjacency!r} is not the frozen {SAMPLE_LEVEL_ADJACENCY}"
        )
    permitted = {
        material_text(settings, "dp_primary_alias"),
        material_text(settings, "dp_secondary_alias"),
    }
    if model_alias not in permitted:
        raise DPBackendError(
            f"{model_alias!r} carries no DP role; 01 §8E names"
            f" {sorted(permitted)} and the secondary is conditional on its own smoke"
        )
    return DPBackendPlan(
        model_alias=model_alias,
        engine=material_text(settings, "dp_engine"),
        adjacency=mechanism.adjacency,
        max_grad_norm=float(mechanism.clipping_norm),
        noise_multiplier=float(mechanism.noise_multiplier),
        sample_rate=float(mechanism.sample_rate),
        steps=int(mechanism.steps),
        dp_seed=int(mechanism.dp_seed),
        accountant_version=ACCOUNTANT_VERSION,
        backend_status=DP_BACKEND_NOT_RUN,
    )


def accounting_for(mechanism: DPMechanism) -> DPAccounting:
    """The one accountant. Opacus's own accountant is never consulted [AUTH: 00 §8.2]."""
    return account(mechanism)


def attach_grad_sample_module(model: Any, plan: DPBackendPlan) -> Any:
    """Wrap a real model for per-sample gradients. Runs no training step.

    Opacus is imported lazily: it belongs to the CUDA-coupled science set and is absent from
    the CPU/dev lane by design [AUTH: 02 §C6].
    """
    if not opacus_available():
        raise DPBackendError(
            f"{DP_BACKEND_NOT_RUN}: opacus is resolved on the H100 image and is absent here"
        )
    from opacus import GradSampleModule  # noqa: PLC0415

    if plan.engine != "opacus.GradSampleModule":
        raise DPBackendError(f"the resolved DP engine is {plan.engine!r}")
    return GradSampleModule(model)


def dp_role_for(root: Path, alias: str) -> str | None:
    """`DP_PRIMARY`, `DP_SECONDARY` or None, from the frozen panel [AUTH: 01 §8E]."""
    from src.backend.settings import backend_settings
    from src.materials import material_text

    settings = backend_settings(root)
    if alias == material_text(settings, "dp_primary_alias"):
        return "DP_PRIMARY"
    if alias == material_text(settings, "dp_secondary_alias"):
        return "DP_SECONDARY"
    return None
