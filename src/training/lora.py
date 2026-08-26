"""LoRA target policy and the trainable-parameter audit [AUTH: 01 §8B; 00 §8.3].

01 §8B fixes LANGUAGE_TRUNK_MLP_ONLY at rank 32: only each architecture's decoder MLP gate,
up and down projections may be adapted. Embeddings, the LM head, vision and audio towers,
multimodal projectors, routers and PLE modules are excluded unless a later robustness
experiment authorises them.

The audit is the enforcement. Selecting the right modules is not enough — the run must also
prove that nothing else became trainable, because an optimiser that touches a base weight
turns "the induced update is B·A" into a false statement, and every merge and recovery result
downstream depends on it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from src.provenance.hashing import JSONValue, sha256_canonical

TARGET_POLICY: Final = "LANGUAGE_TRUNK_MLP_ONLY"
FALLBACK_POLICY: Final = "LANGUAGE_TRUNK_ALL_COMMON_LINEAR"

#: The three projections the primary policy adapts [AUTH: 01 §8B].
PRIMARY_TARGET_ROLES: Final[tuple[str, ...]] = ("gate", "up", "down")

#: Module roles that may never be adapted under the primary policy [AUTH: 01 §8B].
FORBIDDEN_ROLES: Final[tuple[str, ...]] = (
    "embedding",
    "lm_head",
    "vision_tower",
    "audio_tower",
    "multimodal_projector",
    "router",
    "expert_gating",
    "ple_embedding",
)


class LoraPolicyError(ValueError):
    """A target mapping or a trainable surface violates the frozen policy."""


@dataclass(frozen=True)
class TargetMapping:
    """One architecture's mapping from the policy roles to its own module names."""

    architecture_family: str
    policy: str
    role_to_module: Mapping[str, str]
    forbidden_modules: tuple[str, ...]

    def identity(self) -> str:
        return sha256_canonical(
            {
                "architecture_family": self.architecture_family,
                "policy": self.policy,
                "role_to_module": dict(self.role_to_module),
            }
        )

    def module_names(self) -> tuple[str, ...]:
        return tuple(self.role_to_module[role] for role in PRIMARY_TARGET_ROLES)


def load_target_mapping(
    architecture_family: str, document: Mapping[str, JSONValue]
) -> TargetMapping:
    """Build and validate one family's mapping from resolved config [AUTH: 01 §17]."""
    families = document.get("target_mappings")
    if not isinstance(families, Mapping) or architecture_family not in families:
        raise LoraPolicyError(f"no LoRA target mapping for {architecture_family!r}")
    entry = families[architecture_family]
    if not isinstance(entry, Mapping):
        raise LoraPolicyError(f"{architecture_family}: mapping is not an object")
    policy = str(document.get("policy", ""))
    if policy != TARGET_POLICY:
        raise LoraPolicyError(
            f"policy {policy!r} is not the frozen primary policy {TARGET_POLICY!r};"
            f" the only permitted alternative is the pre-registered {FALLBACK_POLICY!r}"
            " fallback [AUTH: 01 §8B, §8B.1]"
        )
    roles = entry.get("roles")
    if not isinstance(roles, Mapping):
        raise LoraPolicyError(f"{architecture_family}: mapping declares no roles")
    missing = [role for role in PRIMARY_TARGET_ROLES if role not in roles]
    if missing:
        raise LoraPolicyError(
            f"{architecture_family}: mapping is missing the {', '.join(missing)} projection(s);"
            " a partial MLP mapping is not the common object [AUTH: 01 §8B]"
        )
    extra = sorted(set(roles) - set(PRIMARY_TARGET_ROLES))
    if extra:
        raise LoraPolicyError(
            f"{architecture_family}: mapping adds non-MLP role(s) {', '.join(extra)}"
        )
    forbidden = entry.get("forbidden_modules")
    return TargetMapping(
        architecture_family=architecture_family,
        policy=policy,
        role_to_module={role: str(roles[role]) for role in PRIMARY_TARGET_ROLES},
        forbidden_modules=tuple(str(name) for name in forbidden)
        if isinstance(forbidden, list)
        else (),
    )


def select_target_parameters(parameter_names: Sequence[str], mapping: TargetMapping) -> list[str]:
    """Every parameter the policy authorises for adaptation, in stable order.

    A forbidden module that happens to contain a target substring is excluded explicitly, so
    e.g. an `lm_head.down_proj` cannot be captured by matching on `down_proj` alone.
    """
    modules = mapping.module_names()
    selected: list[str] = []
    for name in parameter_names:
        if any(marker in name for marker in mapping.forbidden_modules):
            continue
        if any(marker in name for marker in FORBIDDEN_ROLES):
            continue
        if any(f".{module}." in name for module in modules):
            selected.append(name)
    if not selected:
        raise LoraPolicyError(
            f"{mapping.architecture_family}: no parameter matched the target mapping"
            f" {modules}; the module naming does not match the declared mapping"
        )
    return selected


@dataclass(frozen=True)
class TrainableAudit:
    """The pre-training record of exactly what may move [AUTH: 00 §8.3; 01 §8B]."""

    policy: str
    architecture_family: str
    target_modules: tuple[str, ...]
    adapter_parameters: tuple[str, ...]
    n_trainable: int
    n_frozen: int
    rank: int
    scaling: float
    dropout: float
    mapping_identity: str

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "policy": self.policy,
            "architecture_family": self.architecture_family,
            "target_modules": list(self.target_modules),
            "adapter_parameters": list(self.adapter_parameters),
            "n_trainable": self.n_trainable,
            "n_frozen": self.n_frozen,
            "rank": self.rank,
            "scaling": self.scaling,
            "dropout": self.dropout,
            "mapping_identity": self.mapping_identity,
        }


def audit_trainable_surface(
    *,
    base_parameter_names: Sequence[str],
    adapter_parameter_names: Sequence[str],
    trainable_parameter_names: Sequence[str],
    mapping: TargetMapping,
    rank: int,
    scaling: float,
    dropout: float,
) -> TrainableAudit:
    """Refuse the run unless every trainable parameter is an authorised adapter parameter.

    Three failures are distinguished because they mean different things: a base weight that
    is trainable breaks the induced-update identity; an adapter attached to a module the
    policy forbids breaks the cross-family common object; and a frozen authorised adapter
    means the surface that was declared is not the surface that trained.

    The declared adapter surface is checked against the mapping itself rather than trusted.
    A caller that declares `lm_head.weight.lora_A` as an adapter and then trains it would
    otherwise pass every consistency check while adapting a forbidden module.
    """
    authorised = set(adapter_parameter_names)
    base = set(base_parameter_names)
    permitted = set(select_target_parameters(list(base_parameter_names), mapping))
    misattached = sorted(
        name
        for name in authorised
        if not any(name.startswith(f"{target}.") for target in permitted)
    )
    if misattached:
        raise LoraPolicyError(
            f"{len(misattached)} adapter parameter(s) attach to a module the"
            f" {mapping.policy} policy does not authorise, first {misattached[0]!r}"
            " [AUTH: 01 §8B]"
        )
    intruders = sorted(set(trainable_parameter_names) & base)
    if intruders:
        raise LoraPolicyError(
            f"{len(intruders)} base parameter(s) are trainable, first {intruders[0]!r};"
            " base weights must stay frozen [AUTH: 01 §8B]"
        )
    unauthorised = sorted(set(trainable_parameter_names) - authorised)
    if unauthorised:
        raise LoraPolicyError(
            f"{len(unauthorised)} trainable parameter(s) are outside the authorised adapter"
            f" surface, first {unauthorised[0]!r} [AUTH: 01 §8B]"
        )
    absent = sorted(authorised - set(trainable_parameter_names))
    if absent:
        raise LoraPolicyError(
            f"{len(absent)} authorised adapter parameter(s) are frozen, first {absent[0]!r};"
            " the adapter surface is incomplete"
        )
    return TrainableAudit(
        policy=mapping.policy,
        architecture_family=mapping.architecture_family,
        target_modules=mapping.module_names(),
        adapter_parameters=tuple(sorted(authorised)),
        n_trainable=len(authorised),
        n_frozen=len(base),
        rank=rank,
        scaling=scaling,
        dropout=dropout,
        mapping_identity=mapping.identity(),
    )
