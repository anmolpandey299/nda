"""Research-model contract for the frozen core panel [AUTH: 01 §8, §8B, §8C, §8E, §8G].

No model is downloaded here. This declares what each panel member must eventually prove, and
refuses the two ways a model can enter an evidentiary run without provenance:

* a floating revision — `main`, a branch, or a tag — which moves [AUTH: 01 §8G];
* a fabricated one. An unacquired model's revision is the explicit sentinel
  `UNRESOLVED_NOT_DOWNLOADED`, which is a valid *declaration* and an invalid *evidentiary*
  value, so it can be recorded now and cannot be run on later.

The Block A `src.provenance.model_manifest` contract owns manifest validation; this module
supplies the panel and the LoRA/DP roles that 01 §8B and §8E fix.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from src.provenance.hashing import JSONValue

#: An unacquired model states this rather than a guess [AUTH: 03 §8].
UNRESOLVED: Final = "UNRESOLVED_NOT_DOWNLOADED"

_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_FLOATING: Final = frozenset({"main", "master", "head", "latest"})

#: Fields 01 §8G requires for every research model.
RESEARCH_PROVENANCE_FIELDS: Final[tuple[str, ...]] = (
    "model_id",
    "revision",
    "base_or_instruct",
    "architecture_family",
    "parameter_count",
    "dtype_on_disk",
    "license",
    "training_data_cutoff",
    "config_sha256",
    "tokenizer_sha256",
    "weight_file_sha256",
    "lora_target_mapping",
    "frozen_modules",
)


class ModelContractError(ValueError):
    """A model declaration cannot support an evidentiary run."""


@dataclass(frozen=True)
class PanelMember:
    """One frozen core-panel checkpoint and its declared roles [AUTH: 01 §8, §8E]."""

    alias: str
    model_id: str
    architecture_family: str
    base_or_instruct: str
    role: str
    revision: str = UNRESOLVED
    dp_role: str | None = None
    #: 01 §8G identity fields. Every one is UNRESOLVED until the model is actually acquired;
    #: none of them may be guessed to make a declaration look complete [AUTH: 03 §8].
    parameter_count: str = UNRESOLVED
    dtype_on_disk: str = UNRESOLVED
    license: str = UNRESOLVED
    training_data_cutoff: str = UNRESOLVED
    config_sha256: str = UNRESOLVED
    tokenizer_sha256: str = UNRESOLVED
    weight_file_sha256: str = UNRESOLVED
    lora_target_mapping: str = UNRESOLVED
    frozen_modules: tuple[str, ...] = ()

    @property
    def acquired(self) -> bool:
        return self.revision != UNRESOLVED

    def unresolved_fields(self) -> tuple[str, ...]:
        """Which 01 §8G fields still have no value. Reported, never filled in."""
        document = self.as_dict()
        return tuple(
            name
            for name in RESEARCH_PROVENANCE_FIELDS
            if document.get(name) == UNRESOLVED
            or (name == "frozen_modules" and not self.frozen_modules)
        )

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "alias": self.alias,
            "model_id": self.model_id,
            "architecture_family": self.architecture_family,
            "base_or_instruct": self.base_or_instruct,
            "role": self.role,
            "revision": self.revision,
            "dp_role": self.dp_role,
            "parameter_count": self.parameter_count,
            "dtype_on_disk": self.dtype_on_disk,
            "license": self.license,
            "training_data_cutoff": self.training_data_cutoff,
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "weight_file_sha256": self.weight_file_sha256,
            "lora_target_mapping": self.lora_target_mapping,
            "frozen_modules": list(self.frozen_modules),
        }


def revision_problems(revision: str, *, evidentiary: bool) -> list[str]:
    """Every reason this revision may not pin a run.

    `UNRESOLVED_NOT_DOWNLOADED` is accepted in a declaration and refused for an evidentiary
    run, which is what keeps an unacquired model visible instead of guessed.
    """
    if not isinstance(revision, str) or not revision.strip():
        return ["revision is empty"]
    value = revision.strip()
    if value == UNRESOLVED:
        return (
            [f"revision is {UNRESOLVED}; the model has not been acquired [AUTH: 03 §8]"]
            if evidentiary
            else []
        )
    if value.lower() in _FLOATING or value.startswith("refs/"):
        return [f"revision {revision!r} is a floating branch or ref [AUTH: 01 §8G]"]
    if not _COMMIT_RE.match(value):
        return [f"revision {revision!r} is not an immutable 40-hex commit revision [01 §8G]"]
    return []


def validate_panel_member(member: PanelMember, *, evidentiary: bool) -> list[str]:
    """Every reason this declaration cannot pin a run of the requested kind."""
    problems = revision_problems(member.revision, evidentiary=evidentiary)
    if not member.model_id or "/" not in member.model_id:
        problems.append(f"{member.alias}: model_id {member.model_id!r} is not an org/name id")
    if member.base_or_instruct not in {"base", "instruct"}:
        problems.append(f"{member.alias}: base_or_instruct must be 'base' or 'instruct'")
    if evidentiary:
        unresolved = member.unresolved_fields()
        if unresolved:
            problems.append(
                f"{member.alias}: 01 §8G field(s) {', '.join(unresolved)} are still"
                f" {UNRESOLVED}; the model has not been acquired [AUTH: 01 §8G; 03 §8]"
            )
    return problems


def _string_tuple(value: JSONValue) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def load_panel(document: Mapping[str, JSONValue]) -> list[PanelMember]:
    """Build the panel from resolved config; nothing is hard-coded in source [AUTH: 01 §17]."""
    raw = document.get("panel")
    if not isinstance(raw, list) or not raw:
        raise ModelContractError("the model panel config defines no 'panel'")
    members: list[PanelMember] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ModelContractError("each panel entry must be an object")
        members.append(
            PanelMember(
                alias=str(entry["alias"]),
                model_id=str(entry["model_id"]),
                architecture_family=str(entry["architecture_family"]),
                base_or_instruct=str(entry["base_or_instruct"]),
                role=str(entry["role"]),
                revision=str(entry.get("revision", UNRESOLVED)),
                dp_role=None if entry.get("dp_role") is None else str(entry["dp_role"]),
                parameter_count=str(entry.get("parameter_count", UNRESOLVED)),
                dtype_on_disk=str(entry.get("dtype_on_disk", UNRESOLVED)),
                license=str(entry.get("license", UNRESOLVED)),
                training_data_cutoff=str(entry.get("training_data_cutoff", UNRESOLVED)),
                config_sha256=str(entry.get("config_sha256", UNRESOLVED)),
                tokenizer_sha256=str(entry.get("tokenizer_sha256", UNRESOLVED)),
                weight_file_sha256=str(entry.get("weight_file_sha256", UNRESOLVED)),
                lora_target_mapping=str(entry.get("lora_target_mapping", UNRESOLVED)),
                frozen_modules=_string_tuple(entry.get("frozen_modules")),
            )
        )
    return members


def dp_primary(members: Sequence[PanelMember]) -> PanelMember:
    """The single DP_PRIMARY member [AUTH: 01 §8E]."""
    primary = [member for member in members if member.dp_role == "DP_PRIMARY"]
    if len(primary) != 1:
        raise ModelContractError(
            f"exactly one DP_PRIMARY is required, found {len(primary)} [AUTH: 01 §8E]"
        )
    return primary[0]


@dataclass(frozen=True)
class BackendRuntime:
    """What a training/scoring backend must declare about itself.

    Block B's scorer validates its resolved `precision` and `max_sequence_length` against a
    declared runtime. Exposing them here is the handoff: the backend states what it actually
    is, and a silent mismatch has nowhere to hide [AUTH: 02 §C6, §C7].
    """

    precision: str
    max_sequence_length: int
    backend_identity: str
    supports_gradients: bool

    def mismatch(self, *, precision: str, max_sequence_length: int) -> list[str]:
        problems: list[str] = []
        if self.precision != precision:
            problems.append(
                f"runtime precision {self.precision!r} differs from the configured {precision!r}"
            )
        if self.max_sequence_length != max_sequence_length:
            problems.append(
                f"runtime max_sequence_length {self.max_sequence_length} differs from the"
                f" configured {max_sequence_length}"
            )
        return problems


def require_scoring_agreement(
    runtime: BackendRuntime, resolved_scoring: Mapping[str, JSONValue]
) -> None:
    """Bind the backend's real runtime to Block B's resolved scoring config.

    Block B deferred production verification of `precision` and `max_sequence_length` to the
    H100 pass. This is the hook it deferred to: the backend states what it is, the config
    states what was asked for, and a difference raises here rather than becoming a silent
    scoring difference [AUTH: 02 §C6, §C7].
    """
    problems = runtime.mismatch(
        precision=str(resolved_scoring["precision"]),
        max_sequence_length=int(str(resolved_scoring["max_sequence_length"])),
    )
    if problems:
        raise ModelContractError("; ".join(problems))
