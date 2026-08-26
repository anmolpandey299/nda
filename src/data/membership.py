"""Natural-member subsets, canary inclusion, and the training data plan.

**Membership state is structural, not asserted.** Two things follow from that.

*Natural subsets* are drawn from a validated `NaturalCandidatePool`, not from a list of
strings a caller assembled. The pool knows the authoritative trained-natural ids AND the
canonical canary identities, and refuses to exist if they intersect, so a canary cannot be
presented as a natural candidate and land in NATURAL_MEMBER_EVAL.

*Canary inclusion* is derived inside `derive_canary_inclusion` from the canonical pool hash,
the training seed and that seed's own recorded `canary_inclusion` family, at the frozen
p from config. The training seed, the inclusion seed and the mask are therefore not three
independent caller assertions that can be mixed and matched; a stored mask can only be
replayed, and replay recomputes it and refuses disagreement.

Architect adjudication resolving 00 §6.3 against §27.1 [PRE-DATA]:

* `NATURAL_MEMBER_EVAL` is ONE fixed set reused across training seeds 101/202/303. Its
  selection RNG is derived from the fixed master split/study seed and the domain separator
  `natural_member_eval`, never from a post-hoc numeric seed.
* `NATURAL_MEMBER_CALIBRATION` may vary per training seed; its selection seed is derived
  from that training seed and the domain separator `natural_member_calibration`.
* Canary inclusion stays independently redrawn Bernoulli(p) per canary per training seed.

The matched no-canary control is a distinct plan type rather than a flag, so a plan with
`canaries_permitted = False` has no code path that can carry one [AUTH: 00 §7.5].
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from src.data.canaries import Canary, inclusion_map
from src.provenance.hashing import JSONValue, sha256_canonical
from src.training.seeds import SeedFamilies

NATURAL_MEMBER_CALIBRATION: Final = "NATURAL_MEMBER_CALIBRATION"
NATURAL_MEMBER_EVAL: Final = "NATURAL_MEMBER_EVAL"

#: Domain separators for the two selection RNGs. They keep the fixed evaluation draw and the
#: per-seed calibration draw from ever colliding, and they name what each stream is for.
EVAL_DOMAIN: Final = "natural_member_eval"
CALIBRATION_DOMAIN: Final = "natural_member_calibration"

SELECTION_SCHEME: Final = "s05.member-selection.sha256.v1"
INCLUSION_SCHEME: Final = "s05.canary-inclusion.derived.v1"


class MembershipError(ValueError):
    """A membership subset or training plan cannot be built as specified."""


def _order_key(record_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{record_id}".encode()).hexdigest()


def sample_without_replacement(
    candidates: Sequence[str], *, count: int, seed: int
) -> tuple[str, ...]:
    """Uniform draw without replacement, deterministic from `seed` [AUTH: 00 §6.3].

    Implemented as a keyed sort rather than an RNG shuffle so the draw is reproducible from
    the seed alone, in any language, without depending on a library's shuffle algorithm.
    """
    unique = list(dict.fromkeys(candidates))
    if len(unique) != len(candidates):
        raise MembershipError("candidate ids contain duplicates")
    if count > len(unique):
        raise MembershipError(
            f"cannot draw {count} records without replacement from {len(unique)} candidates"
        )
    ordered = sorted(unique, key=lambda record_id: _order_key(record_id, seed))
    return tuple(ordered[:count])


def selection_seed(master_seed: int, domain: str) -> int:
    """One selection stream, keyed on (scheme, master seed, domain separator).

    The evaluation draw passes the fixed study/split seed; the calibration draw passes the
    training seed. Neither is an arbitrary numeric constant chosen after the fact.
    """
    digest = hashlib.sha256(f"{SELECTION_SCHEME}|{master_seed}|{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (1 << 31)


@dataclass(frozen=True)
class NaturalCandidatePool:
    """The authoritative natural candidates for one study, with canaries excluded by force.

    Built from the master corpus partition and the canonical canary pool. Construction fails
    if any canary identity appears among the natural candidates, so no downstream draw has to
    remember to filter them out [AUTH: 00 §6.3, §7.1].
    """

    trained_natural_ids: tuple[str, ...]
    canary_ids: frozenset[str]
    canary_pool_sha256: str
    study_seed: int

    def __post_init__(self) -> None:
        if len(set(self.trained_natural_ids)) != len(self.trained_natural_ids):
            raise MembershipError("the natural candidate pool repeats a record id")
        intruders = sorted(set(self.trained_natural_ids) & set(self.canary_ids))
        if intruders:
            raise MembershipError(
                f"{len(intruders)} synthetic canary/canaries were presented as natural"
                f" candidates, first {intruders[0]!r}; canaries are excluded from natural"
                " member selection [AUTH: 00 §6.3, §7.1]"
            )

    def identity(self) -> str:
        return sha256_canonical(
            {
                "trained_natural_ids": list(self.trained_natural_ids),
                "canary_pool_sha256": self.canary_pool_sha256,
                "study_seed": self.study_seed,
            }
        )


def canary_pool_identity(pool: Sequence[Canary]) -> str:
    """Canonical identity of the canary pool: ids and normalised-text hashes, in order."""
    return sha256_canonical(
        [
            {"canary_id": canary.canary_id, "text_sha256": canary.record.text_sha256}
            for canary in pool
        ]
    )


def natural_candidate_pool(
    trained_natural_ids: Sequence[str], pool: Sequence[Canary], *, study_seed: int
) -> NaturalCandidatePool:
    """Build the validated pool. This is the only entry point for natural member selection."""
    return NaturalCandidatePool(
        trained_natural_ids=tuple(trained_natural_ids),
        canary_ids=frozenset(canary.canary_id for canary in pool),
        canary_pool_sha256=canary_pool_identity(pool),
        study_seed=study_seed,
    )


@dataclass(frozen=True)
class NaturalMemberSubsets:
    """The two disjoint natural-member subsets for one training seed [AUTH: 00 §6.3]."""

    calibration: tuple[str, ...]
    evaluation: tuple[str, ...]
    seed: int
    evaluation_seed: int
    calibration_seed: int
    candidate_pool_identity: str

    def __post_init__(self) -> None:
        overlap = sorted(set(self.calibration) & set(self.evaluation))
        if overlap:
            raise MembershipError(
                f"{len(overlap)} record(s) appear in both natural-member subsets, first"
                f" {overlap[0]!r} [AUTH: 00 §6.3; 01 §23]"
            )

    def evaluation_identity(self) -> str:
        """The persisted identity of the FIXED evaluation subset [architect adjudication].

        It carries no training seed, because the set does not depend on one: the same hash
        must appear for seeds 101, 202 and 303.
        """
        return sha256_canonical(
            {
                "subset": NATURAL_MEMBER_EVAL,
                "scheme": SELECTION_SCHEME,
                "domain": EVAL_DOMAIN,
                "selection_seed": self.evaluation_seed,
                "candidate_pool_identity": self.candidate_pool_identity,
                "record_ids": list(self.evaluation),
            }
        )

    def calibration_identity(self) -> str:
        return sha256_canonical(
            {
                "subset": NATURAL_MEMBER_CALIBRATION,
                "scheme": SELECTION_SCHEME,
                "domain": CALIBRATION_DOMAIN,
                "selection_seed": self.calibration_seed,
                "candidate_pool_identity": self.candidate_pool_identity,
                "record_ids": list(self.calibration),
            }
        )


def natural_member_subsets(
    candidates: NaturalCandidatePool,
    *,
    calibration_size: int,
    evaluation_size: int,
    training_seed: int,
) -> NaturalMemberSubsets:
    """Draw both subsets from the validated pool [architect adjudication; AUTH: 00 §6.3].

    Order matters and is fixed: the evaluation subset is drawn FIRST from the whole candidate
    pool using the study-seed stream, so it is identical for every training seed. Calibration
    is then drawn from what remains, using the training-seed stream, so it may vary per seed
    and can never intersect the evaluation set.
    """
    if not isinstance(candidates, NaturalCandidatePool):  # pragma: no cover - typing guard
        raise MembershipError(
            "natural member selection requires a validated NaturalCandidatePool, not raw"
            " candidate strings [AUTH: 00 §6.3, §7.1]"
        )
    if calibration_size < 0 or evaluation_size < 0:
        raise MembershipError("subset sizes must be non-negative")
    available = len(candidates.trained_natural_ids)
    if calibration_size + evaluation_size > available:
        raise MembershipError(
            f"cannot draw {calibration_size + evaluation_size} natural members from"
            f" {available} trained natural records"
        )

    eval_seed = selection_seed(candidates.study_seed, EVAL_DOMAIN)
    calibration_seed = selection_seed(training_seed, CALIBRATION_DOMAIN)
    evaluation = sample_without_replacement(
        candidates.trained_natural_ids, count=evaluation_size, seed=eval_seed
    )
    remaining = [
        record_id
        for record_id in candidates.trained_natural_ids
        if record_id not in set(evaluation)
    ]
    calibration = sample_without_replacement(
        remaining, count=calibration_size, seed=calibration_seed
    )
    return NaturalMemberSubsets(
        calibration=calibration,
        evaluation=evaluation,
        seed=training_seed,
        evaluation_seed=eval_seed,
        calibration_seed=calibration_seed,
        candidate_pool_identity=candidates.identity(),
    )


@dataclass(frozen=True)
class CanaryInclusion:
    """One training seed's canary membership, DERIVED and self-describing [AUTH: 00 §7.2].

    The training seed, the inclusion seed and the mask travel as one object precisely so they
    cannot be asserted independently. `derive_canary_inclusion` is the only constructor a
    caller should use; `replay_canary_inclusion` re-derives a persisted mask and refuses any
    disagreement rather than trusting the stored bytes.
    """

    canary_pool_sha256: str
    training_seed: int
    inclusion_seed: int
    probability: float
    mask: Mapping[str, bool]
    scheme: str = INCLUSION_SCHEME

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise MembershipError("the inclusion probability must lie in [0, 1]")
        if not self.mask:
            raise MembershipError("an inclusion mask covering no canary is not a draw")

    def mask_sha256(self) -> str:
        return sha256_canonical(
            {
                "scheme": self.scheme,
                "canary_pool_sha256": self.canary_pool_sha256,
                "inclusion_seed": self.inclusion_seed,
                "probability": self.probability,
                "mask": {key: bool(value) for key, value in sorted(self.mask.items())},
            }
        )

    def included_ids(self) -> tuple[str, ...]:
        return tuple(sorted(key for key, value in self.mask.items() if value))

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "scheme": self.scheme,
            "canary_pool_sha256": self.canary_pool_sha256,
            "training_seed": self.training_seed,
            "inclusion_seed": self.inclusion_seed,
            "inclusion_probability": self.probability,
            "inclusion_mask_sha256": self.mask_sha256(),
            "n_included": len(self.included_ids()),
        }


def derive_canary_inclusion(
    pool: Sequence[Canary], *, seeds: SeedFamilies, probability: float
) -> CanaryInclusion:
    """The only way to obtain an inclusion mask [AUTH: 00 §7.2; 01 §30].

    The inclusion seed is read off the run's own recorded `canary_inclusion` family, so a
    caller cannot pair one training seed with another seed's draw. `probability` comes from
    resolved config, not from a source literal [AUTH: 01 §17].
    """
    inclusion_seed = seeds["canary_inclusion"]
    return CanaryInclusion(
        canary_pool_sha256=canary_pool_identity(pool),
        training_seed=seeds.training_seed,
        inclusion_seed=inclusion_seed,
        probability=probability,
        mask=inclusion_map(pool, inclusion_seed=inclusion_seed, probability=probability),
    )


def replay_canary_inclusion(
    pool: Sequence[Canary],
    *,
    seeds: SeedFamilies,
    probability: float,
    stored: CanaryInclusion,
) -> CanaryInclusion:
    """Re-derive a persisted mask and refuse it if anything disagrees.

    This is the guard against a mask from seed 111 being replayed as seed 202's draw, and
    against a stored mask that was edited after the fact.
    """
    derived = derive_canary_inclusion(pool, seeds=seeds, probability=probability)
    problems: list[str] = []
    if stored.training_seed != derived.training_seed:
        problems.append(
            f"stored mask claims training seed {stored.training_seed}, this run is"
            f" {derived.training_seed}"
        )
    if stored.inclusion_seed != derived.inclusion_seed:
        problems.append(
            f"stored inclusion seed {stored.inclusion_seed} is not the {derived.training_seed}"
            f" seed family value {derived.inclusion_seed} [AUTH: 01 §30]"
        )
    if stored.canary_pool_sha256 != derived.canary_pool_sha256:
        problems.append("stored mask was drawn over a different canary pool")
    if stored.probability != derived.probability:
        problems.append(
            f"stored inclusion probability {stored.probability} is not the configured"
            f" {derived.probability}"
        )
    if stored.mask_sha256() != derived.mask_sha256():
        problems.append(
            "the stored mask does not match the mask its recorded seed derives; a canary"
            " membership vector may not be edited or substituted [AUTH: 00 §7.2]"
        )
    if problems:
        raise MembershipError("; ".join(problems))
    return derived


@dataclass(frozen=True)
class TrainingPlan:
    """Exactly what one training run consumes [AUTH: 00 §7.2, §7.5, §9].

    `canaries_permitted = False` is the matched no-canary control. Construction refuses a plan
    that carries canaries while forbidding them, so the control cannot acquire one by an
    ordering mistake later.
    """

    label: str
    training_seed: int
    natural_ids: tuple[str, ...]
    canaries: tuple[Canary, ...]
    canaries_permitted: bool
    inclusion: CanaryInclusion | None
    candidate_pool_identity: str | None = None

    def __post_init__(self) -> None:
        if len(set(self.natural_ids)) != len(self.natural_ids):
            raise MembershipError(f"{self.label}: duplicate natural record id")
        if not self.canaries_permitted:
            if self.canaries:
                raise MembershipError(
                    f"{self.label}: a no-canary plan may not carry {len(self.canaries)}"
                    " synthetic canary/canaries [AUTH: 00 §7.5]"
                )
            if self.inclusion is not None:
                raise MembershipError(
                    f"{self.label}: a no-canary plan has no inclusion draw to record"
                )
        else:
            if self.inclusion is None:
                raise MembershipError(
                    f"{self.label}: a canary plan must carry its derived CanaryInclusion"
                )
            if self.inclusion.training_seed != self.training_seed:
                raise MembershipError(
                    f"{self.label}: the inclusion draw is for training seed"
                    f" {self.inclusion.training_seed}, not {self.training_seed}"
                    " [AUTH: 00 §7.2]"
                )
            carried = {canary.canary_id for canary in self.canaries}
            if carried != set(self.inclusion.included_ids()):
                raise MembershipError(
                    f"{self.label}: the carried canaries are not the ones the recorded"
                    " inclusion mask selects [AUTH: 00 §7.2]"
                )
        ids = [canary.canary_id for canary in self.canaries]
        if len(set(ids)) != len(ids):
            raise MembershipError(f"{self.label}: a canary appears more than once [AUTH: 00 §7.3]")
        intruders = sorted(set(self.natural_ids) & set(ids))
        if intruders:
            raise MembershipError(
                f"{self.label}: {intruders[0]!r} is both a natural record and a canary"
            )

    @property
    def inclusion_seed(self) -> int | None:
        return None if self.inclusion is None else self.inclusion.inclusion_seed

    @property
    def trained_natural_ids(self) -> tuple[str, ...]:
        """Natural records only. Canaries are never eligible as natural members."""
        return self.natural_ids

    @property
    def n_training_records(self) -> int:
        return len(self.natural_ids) + len(self.canaries)

    def contains_canary(self, canary_id: str) -> bool:
        return any(canary.canary_id == canary_id for canary in self.canaries)

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "label": self.label,
            "training_seed": self.training_seed,
            "n_natural": len(self.natural_ids),
            "n_canaries": len(self.canaries),
            "canaries_permitted": self.canaries_permitted,
            "candidate_pool_identity": self.candidate_pool_identity,
            "canary_pool_sha256": None
            if self.inclusion is None
            else self.inclusion.canary_pool_sha256,
            "inclusion_seed": self.inclusion_seed,
            "inclusion_mask_sha256": (
                None if self.inclusion is None else self.inclusion.mask_sha256()
            ),
        }


def canary_plan(
    *,
    label: str,
    candidates: NaturalCandidatePool,
    natural_ids: Sequence[str],
    pool: Sequence[Canary],
    inclusion: CanaryInclusion,
) -> TrainingPlan:
    """The treated arm. The canaries are read off the derived mask, never handed in."""
    unknown = sorted(set(natural_ids) - set(candidates.trained_natural_ids))
    if unknown:
        raise MembershipError(
            f"{len(unknown)} training id(s) are not in the validated natural candidate pool,"
            f" first {unknown[0]!r}"
        )
    if inclusion.canary_pool_sha256 != canary_pool_identity(pool):
        raise MembershipError("the inclusion mask was drawn over a different canary pool")
    selected = set(inclusion.included_ids())
    return TrainingPlan(
        label=label,
        training_seed=inclusion.training_seed,
        natural_ids=tuple(natural_ids),
        canaries=tuple(canary for canary in pool if canary.canary_id in selected),
        canaries_permitted=True,
        inclusion=inclusion,
        candidate_pool_identity=candidates.identity(),
    )


def no_canary_plan(
    *,
    label: str,
    training_seed: int,
    candidates: NaturalCandidatePool,
    natural_ids: Sequence[str],
) -> TrainingPlan:
    """The matched control: identical natural corpus, zero synthetic canaries [00 §7.5]."""
    unknown = sorted(set(natural_ids) - set(candidates.trained_natural_ids))
    if unknown:
        raise MembershipError(
            f"{len(unknown)} training id(s) are not in the validated natural candidate pool,"
            f" first {unknown[0]!r}"
        )
    return TrainingPlan(
        label=label,
        training_seed=training_seed,
        natural_ids=tuple(natural_ids),
        canaries=(),
        canaries_permitted=False,
        inclusion=None,
        candidate_pool_identity=candidates.identity(),
    )


def matched_control_problems(canary: TrainingPlan, control: TrainingPlan) -> list[str]:
    """Every way the pair fails to be a matched comparison [AUTH: 00 §7.5].

    Plan-level matching only. The execution-level match — hyperparameters, seed families and
    the optimiser-update budget — is checked by
    `src.training.execution.matched_execution_problems`, because two matched plans can still
    be executed differently.
    """
    problems: list[str] = []
    if canary.training_seed != control.training_seed:
        problems.append("the control uses a different training seed")
    if canary.natural_ids != control.natural_ids:
        problems.append("the control uses a different natural training corpus")
    if canary.candidate_pool_identity != control.candidate_pool_identity:
        problems.append("the two arms were drawn from different natural candidate pools")
    if control.canaries or control.inclusion is not None:
        problems.append("the control carries synthetic canaries")
    if not canary.canaries:
        problems.append("the canary-containing plan carries no canaries")
    return problems
