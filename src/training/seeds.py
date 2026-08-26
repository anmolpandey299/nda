"""Seed families derived from one master training seed [AUTH: 01 §30; 00 §9].

01 §30 requires each RNG family to be recorded separately and forbids one undocumented global
seed standing in for all of them. Both halves matter: the families must be *recorded*, and
they must actually *drive different components*, or the record is decoration.

The derivation is a keyed hash of (master seed, family name), so:

* every family is reproducible from the master seed alone;
* families are independent — changing the data order cannot move the LoRA initialisation;
* a different master seed moves every family at once, which is what 00 §9 means by "each seed
  changes parameter initialisation, mini-batch order, DP noise and canary inclusion".

The family vocabulary is Block A's `run_manifest.SEED_FIELDS`, bound by test, so a run
manifest and a trainer cannot disagree about what was recorded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from src.provenance.hashing import JSONValue

DERIVATION_SCHEME: Final = "s06.seed-family.sha256.v1"

#: The families this block drives, beyond the master seed itself [AUTH: 01 §30].
DERIVED_FAMILIES: Final[tuple[str, ...]] = (
    "python_rng",
    "numpy_rng",
    "torch_cpu_rng",
    "torch_cuda_rng",
    "data_order",
    "canary_inclusion",
    "lora_init",
    "dp_noise",
)

#: Families that only apply under differential privacy [AUTH: 00 §8.2].
DP_ONLY_FAMILIES: Final[tuple[str, ...]] = ("dp_noise",)

_MODULUS: Final = 1 << 31


class SeedError(ValueError):
    """A seed family cannot be derived or is inconsistent with its run."""


def derive(master_seed: int, family: str) -> int:
    """One family's value. Deterministic, independent, and reproducible anywhere."""
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise SeedError("the master training seed must be an integer")
    if family not in DERIVED_FAMILIES:
        raise SeedError(f"{family!r} is not a declared seed family [AUTH: 01 §30]")
    digest = hashlib.sha256(f"{DERIVATION_SCHEME}|{master_seed}|{family}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _MODULUS


@dataclass(frozen=True)
class SeedFamilies:
    """Every RNG stream one training run uses, recorded separately [AUTH: 01 §30]."""

    training_seed: int
    values: Mapping[str, int]
    scheme: str = DERIVATION_SCHEME

    def __getitem__(self, family: str) -> int:
        if family not in self.values:
            raise SeedError(f"{family!r} was not recorded for this run")
        return self.values[family]

    def as_manifest_seeds(self) -> dict[str, int]:
        """The mapping Block A's run manifest stores under `seeds` [AUTH: 01 §15, §30]."""
        return {"training_seed": self.training_seed, **dict(self.values)}

    def as_dict(self) -> dict[str, JSONValue]:
        return {"training_seed": self.training_seed, "scheme": self.scheme, **dict(self.values)}


def seed_families(master_seed: int, *, differentially_private: bool) -> SeedFamilies:
    """Derive the families a run of this kind actually uses.

    A non-DP run records no `dp_noise` family: recording an unused stream would claim the run
    drew noise it never drew.
    """
    families = [
        name for name in DERIVED_FAMILIES if differentially_private or name not in DP_ONLY_FAMILIES
    ]
    return SeedFamilies(
        training_seed=master_seed, values={name: derive(master_seed, name) for name in families}
    )


def families_are_independent(master_seed: int) -> bool:
    """No two families of one master seed may collide."""
    values = [derive(master_seed, family) for family in DERIVED_FAMILIES]
    return len(set(values)) == len(values)
