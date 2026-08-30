"""Language-trunk selection and the common LoRA target mapping [AUTH: 01 §8B; 00 §8.3].

The scientific subject is the **language trunk only**. Qwen, Gemma and Ministral variants
expose vision or audio towers, multimodal projectors and per-layer embeddings; none of those
may enter the LoRA surface, the induced update, or any merge or recovery surface. A parameter
that is adapted becomes part of ΔW = BA, so a multimodal weight leaking into the target set
would silently redefine the object every downstream stage measures.

Selection is **exact, not regex**. Each family declares `mlp.gate_proj`, `mlp.up_proj` and
`mlp.down_proj` in version-controlled config, and a parameter qualifies only when its dotted
name ends with `<module>.weight` under a decoder layer of the language trunk. Substring
matching would let `lm_head.down_proj` or `vision_tower...mlp.up_proj` in; suffix matching on
the declared module path does not.

If an expected projection cannot be uniquely identified the family fails compatibility. The
01 §8B.1 fallback policy `LANGUAGE_TRUNK_ALL_COMMON_LINEAR` is NOT activated here: it stays
gated on the later P0-A measurement-power condition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.provenance.hashing import JSONValue, sha256_canonical
from src.training.lora import (
    FORBIDDEN_ROLES,
    PRIMARY_TARGET_ROLES,
    TARGET_POLICY,
    LoraPolicyError,
    TargetMapping,
    load_target_mapping,
)

MAPPING_SCHEMA: Final = "backend.architecture-mapping.v1"

#: Module-name fragments that mark a non-language tower or projector. A parameter whose dotted
#: path contains any of these is outside the scientific subject [AUTH: 01 §8B].
MULTIMODAL_MARKERS: Final[tuple[str, ...]] = (
    "vision_tower",
    "vision_model",
    "vision_encoder",
    "visual",
    "audio_tower",
    "audio_encoder",
    "multi_modal_projector",
    "multimodal_projector",
    "mm_projector",
    "per_layer_embeddings",
    "ple_embedding",
    "image_newline",
)

#: Where the language trunk lives, per family. The first prefix that matches a parameter set
#: is the trunk; nothing outside it is ever adapted.
LANGUAGE_TRUNK_PREFIXES: Final[tuple[str, ...]] = (
    "model.language_model.",
    "language_model.model.",
    "language_model.",
    "model.",
)

#: The four locked core families plus the tiny CPU fixture [AUTH: 01 §8.1-§8.4].
CORE_FAMILIES: Final[tuple[str, ...]] = (
    "qwen3_5",
    "gemma_4",
    "ministral_3",
    "llama_3_2",
)


class ArchitectureError(LoraPolicyError):
    """A model's module layout cannot support the frozen common target policy."""


@dataclass(frozen=True)
class TrunkSelection:
    """One model's language trunk, the excluded towers, and the resolved LoRA surface."""

    architecture_family: str
    mapping_version: str
    trunk_prefix: str
    target_parameters: tuple[str, ...]
    excluded_multimodal: tuple[str, ...]
    excluded_forbidden: tuple[str, ...]
    role_to_module: Mapping[str, str]

    @property
    def target_modules(self) -> tuple[str, ...]:
        """The PEFT `target_modules` entries: leaf module names, in frozen role order."""
        return tuple(self.role_to_module[role].split(".")[-1] for role in PRIMARY_TARGET_ROLES)

    def identity(self) -> str:
        return sha256_canonical(
            {
                "schema": MAPPING_SCHEMA,
                "architecture_family": self.architecture_family,
                "mapping_version": self.mapping_version,
                "policy": TARGET_POLICY,
                "trunk_prefix": self.trunk_prefix,
                "role_to_module": dict(self.role_to_module),
                "target_parameters": list(self.target_parameters),
            }
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": MAPPING_SCHEMA,
            "architecture_family": self.architecture_family,
            "mapping_version": self.mapping_version,
            "policy": TARGET_POLICY,
            "trunk_prefix": self.trunk_prefix,
            "role_to_module": dict(self.role_to_module),
            "target_modules": list(self.target_modules),
            "n_target_parameters": len(self.target_parameters),
            "n_excluded_multimodal": len(self.excluded_multimodal),
            "mapping_sha256": self.identity(),
        }


def is_multimodal(name: str) -> bool:
    """True when a parameter belongs to a non-language tower or projector."""
    return any(marker in name for marker in MULTIMODAL_MARKERS)


def language_trunk_prefix(parameter_names: Sequence[str]) -> str:
    """The prefix under which the decoder layers live.

    Multimodal parameters are removed first, so a vision tower nested under `model.` cannot
    drag the trunk prefix up to something that would then include it.
    """
    language = [name for name in parameter_names if not is_multimodal(name)]
    if not language:
        raise ArchitectureError("every parameter belongs to a non-language tower")
    for prefix in LANGUAGE_TRUNK_PREFIXES:
        if any(
            f"{prefix}layers." in name or name.startswith(f"{prefix}layers.") for name in language
        ):
            return prefix
    raise ArchitectureError(
        "no decoder-layer stack was found under any declared language-trunk prefix"
        f" {LANGUAGE_TRUNK_PREFIXES}; the module layout is not the one 01 §8B maps"
    )


def select_language_trunk(
    parameter_names: Sequence[str], mapping: TargetMapping, *, mapping_version: str
) -> TrunkSelection:
    """Resolve the LoRA surface for one model's parameter names [AUTH: 01 §8B].

    Every exclusion is recorded rather than merely applied, so the audit can show what was
    kept out and why.
    """
    names = list(parameter_names)
    if not names:
        raise ArchitectureError("a model with no parameters has no language trunk")
    trunk = language_trunk_prefix(names)

    multimodal = tuple(sorted(name for name in names if is_multimodal(name)))
    forbidden_markers = (*mapping.forbidden_modules, *FORBIDDEN_ROLES)
    forbidden = tuple(
        sorted(
            name
            for name in names
            if name not in multimodal and any(marker in name for marker in forbidden_markers)
        )
    )

    selected: list[str] = []
    for role in PRIMARY_TARGET_ROLES:
        module = mapping.role_to_module[role]
        matches = [
            name
            for name in names
            if name not in multimodal
            and name not in forbidden
            and name.startswith(trunk)
            and name.endswith(f".{module}.weight")
        ]
        if not matches:
            raise ArchitectureError(
                f"{mapping.architecture_family}: the {role} projection {module!r} matched no"
                " language-trunk parameter; the module naming does not match the declared"
                " mapping, so this family fails compatibility [AUTH: 01 §8B]"
            )
        selected.extend(matches)

    duplicates = sorted({name for name in selected if selected.count(name) > 1})
    if duplicates:
        raise ArchitectureError(
            f"{mapping.architecture_family}: parameter(s) {duplicates[:3]} matched more than"
            " one declared role; the projection cannot be uniquely identified"
        )
    leaked = sorted(name for name in selected if is_multimodal(name))
    if leaked:  # pragma: no cover - the multimodal filter above already removed these
        raise ArchitectureError(f"multimodal parameter(s) {leaked[:3]} entered the LoRA surface")

    return TrunkSelection(
        architecture_family=mapping.architecture_family,
        mapping_version=mapping_version,
        trunk_prefix=trunk,
        target_parameters=tuple(sorted(selected)),
        excluded_multimodal=multimodal,
        excluded_forbidden=forbidden,
        role_to_module=dict(mapping.role_to_module),
    )


def resolve_trunk_selection(
    root: Path, *, architecture_family: str, parameter_names: Sequence[str]
) -> TrunkSelection:
    """Load the family's version-controlled mapping and resolve its LoRA surface."""
    from src.backend.settings import backend_settings
    from src.materials import material_text
    from src.provenance.config import resolve_config

    configs = root / "configs"
    panel = resolve_config(configs / "models" / "panel.json", config_root=configs)
    mapping = load_target_mapping(architecture_family, panel)
    version = material_text(backend_settings(root), "architecture_mapping_version")
    return select_language_trunk(parameter_names, mapping, mapping_version=version)
