"""The pre-data calibration rule for the 11 unfrozen training constants.

00 §8.3 requires the LoRA and DP constants to be *fixed across seeds and written into the
experiment configuration*, but fixes no numeric value for any of them. They are therefore
`REQUIRED_NOT_CALIBRATED` until one bounded calibration run measures them on the real
checkpoint.

This module is the **rule**, declared before any measurement exists. Writing the rule now and
the values later is what keeps the choice from being result-driven: every tie-break below is
mechanical, so the same measurements always select the same winner, and no one can prefer a
configuration after seeing what it does to a privacy outcome [AUTH: 00 §34B.1A; 01 §3.2].

What the rule is allowed to look at is deliberately narrow — training stability, training
loss, held-out language-modelling loss, peak memory and wall-clock. No membership label,
score, threshold or privacy quantity is formed from any of it [AUTH: 00 §18].

Two data-source rules are science-critical, and are constants here rather than prose:

* **the gradient-norm percentile is estimated on public non-member data only.** A clipping
  norm fitted to the gradient norms of protected training examples is a statistic *of the
  private set*, and it then travels into the DP mechanism as a public hyperparameter — the
  released sigma would be calibrated to the very records DP is meant to protect. The
  percentile is therefore taken on `CALIBRATION_NONMEMBERS` and never on member records.
* **held-out utility is measured on `CALIBRATION_NONMEMBERS`, never `EVAL_NONMEMBERS`.**
  Selecting a configuration on the evaluation split would put evaluation IDs into a
  calibration decision [AUTH: 01 §23; 00 §34B.1].

The DP constants are derived, not chosen: epsilon is already frozen at the 00 §8.2 target, and
`src.dp.mechanism.calibrate_noise_multiplier` solves for the smallest sigma meeting it at the
sample rate and step count the selected LoRA constants imply.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from src.provenance.hashing import JSONValue, sha256_canonical

CALIBRATION_RULE: Final = "s10.pre-data-training-calibration.v1"

#: The bounded grid. Small on purpose: this is a feasibility-and-stability sweep, not a
#: hyperparameter search, and a wide grid would invite selecting on an outcome.
LEARNING_RATES: Final[tuple[float, ...]] = (5e-5, 1e-4, 2e-4)
BATCH_SIZES: Final[tuple[int, ...]] = (4, 8, 16)
EPOCHS: Final[tuple[int, ...]] = (1, 2, 3)

#: The optimiser is not swept. 00 §8.3 requires one fixed optimiser and 00 §2772 forbids
#: introducing a new one; AdamW is the PEFT/HF default the reference trainer already assumes.
OPTIMIZER: Final = "adamw"

#: Feasibility limits for one H100 SXM 80GB. A configuration that does not fit, or that cannot
#: finish the arm inside the budget, is rejected before any loss is compared.
MEMORY_HEADROOM_FRACTION: Final = 0.90
MAX_WALL_CLOCK_SECONDS_PER_RUN: Final = 6 * 60 * 60

#: Percentile rules, fixed here so the measurement cannot suggest them afterwards.
GRADIENT_CLIP_PERCENTILE: Final = 95.0
SEQUENCE_LENGTH_PERCENTILE: Final = 99.0

#: The ONLY partition the gradient-norm percentile may be estimated on. Public non-member
#: data: the clipping norm becomes a public DP hyperparameter, so fitting it to protected
#: records would leak a statistic of the private set into the released mechanism.
GRADIENT_NORM_SOURCE: Final = "CALIBRATION_NONMEMBERS"

#: The ONLY partition held-out utility may be measured on. `EVAL_NONMEMBERS` is reserved for
#: the realised-FPR measurement and must not enter a calibration decision [AUTH: 01 §23].
HELDOUT_UTILITY_SOURCE: Final = "CALIBRATION_NONMEMBERS"

#: Partitions that carry protected training records. Never a calibration input.
PROTECTED_PARTITIONS: Final[tuple[str, ...]] = ("TRAIN_CANDIDATES", "NATURAL_MEMBER_EVAL")

#: delta = 1 / N^1.1, rounded down to one significant figure.
#:
#: **N is the PROTECTED TRAINING-SET SIZE** — the number of records the DP mechanism actually
#: protects (the natural records plus included canaries that the arm trains on), not the
#: candidate-pool size, not the corpus size, and not the number of optimizer steps. 00 §8.2
#: fixes epsilon and is silent on delta; this is the standard "comfortably sub-1/N"
#: convention, declared before N is known so it cannot be tuned to a result.
DELTA_EXPONENT: Final = 1.1
DELTA_N_DEFINITION: Final = "PROTECTED_TRAINING_SET_SIZE"

CALIBRATION_SEED: Final = 101


class CalibrationError(ValueError):
    """A calibration cannot be planned or resolved as specified."""


@dataclass(frozen=True)
class GridPoint:
    """One configuration the sweep will train."""

    learning_rate: float
    batch_size: int
    epochs: int

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

    @property
    def sort_key(self) -> tuple[float, int, int]:
        """The final, total tie-break: lexicographic over the grid axes."""
        return (self.learning_rate, self.batch_size, self.epochs)


@dataclass(frozen=True)
class GridMeasurement:
    """What one trained grid point produced. Utility and cost only — never a privacy view."""

    point: GridPoint
    stable: bool
    final_train_loss: float
    heldout_loss: float
    peak_memory_bytes: int
    wall_clock_seconds: float
    gradient_norm_percentile: float

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "point": self.point.as_dict(),
            "stable": self.stable,
            "final_train_loss": self.final_train_loss,
            "heldout_loss": self.heldout_loss,
            "peak_memory_bytes": self.peak_memory_bytes,
            "wall_clock_seconds": self.wall_clock_seconds,
            "gradient_norm_percentile": self.gradient_norm_percentile,
        }


def calibration_grid() -> tuple[GridPoint, ...]:
    """The whole sweep, in a fixed order. 27 points, declared before any of them runs."""
    return tuple(
        GridPoint(learning_rate=lr, batch_size=bs, epochs=ep)
        for lr in LEARNING_RATES
        for bs in BATCH_SIZES
        for ep in EPOCHS
    )


def feasible(measurement: GridMeasurement, *, device_memory_bytes: int) -> bool:
    """Whether a point may be considered at all.

    Applied before any loss comparison, so an unstable or infeasible configuration is never
    rescued by having produced an attractive number.
    """
    if not measurement.stable:
        return False
    for value in (measurement.final_train_loss, measurement.heldout_loss):
        if value != value or value in (float("inf"), float("-inf")):  # NaN or Inf
            return False
    if measurement.peak_memory_bytes > device_memory_bytes * MEMORY_HEADROOM_FRACTION:
        return False
    return measurement.wall_clock_seconds <= MAX_WALL_CLOCK_SECONDS_PER_RUN


def select_winner(
    measurements: Sequence[GridMeasurement], *, device_memory_bytes: int
) -> GridMeasurement:
    """The single deterministic winner [PRE-DECLARED].

    1. keep only stable, memory-feasible, time-feasible points;
    2. lowest held-out loss wins;
    3. ties: lower peak memory, then lower wall clock, then lexicographic (lr, batch, epochs).

    Every step is total and mechanical, so the same measurements always yield the same winner
    regardless of who runs the selection or in what order the points completed.
    """
    survivors = [m for m in measurements if feasible(m, device_memory_bytes=device_memory_bytes)]
    if not survivors:
        raise CalibrationError(
            "no grid point was stable and feasible; the calibration has failed rather than"
            " produced a winner, and the constants stay REQUIRED_NOT_CALIBRATED"
        )
    return min(
        survivors,
        key=lambda m: (
            m.heldout_loss,
            m.peak_memory_bytes,
            m.wall_clock_seconds,
            m.point.sort_key,
        ),
    )


def resolve_delta(protected_training_set_size: int) -> float:
    """delta = 1 / N^1.1, floored to one significant figure [PRE-DECLARED].

    `protected_training_set_size` is N: the number of records the DP mechanism protects, i.e.
    what the arm actually trains on. Passing the candidate pool or the corpus size would make
    delta smaller than the guarantee justifies.
    """
    import math

    if protected_training_set_size <= 1:
        raise CalibrationError(
            "delta needs the protected training-set size, which must exceed one record"
        )
    raw = 1.0 / (float(protected_training_set_size) ** DELTA_EXPONENT)
    exponent = math.floor(math.log10(raw))
    leading = math.floor(raw / (10.0**exponent))
    return float(leading * (10.0**exponent))


def resolve_sequence_length(token_lengths: Sequence[int], *, model_max_positions: int) -> int:
    """The smallest power of two covering the 99th percentile, capped by the model."""
    import math

    import numpy as np

    if not token_lengths:
        raise CalibrationError("the tokenized corpus is empty, so no length can be resolved")
    percentile = float(np.percentile(np.asarray(token_lengths), SEQUENCE_LENGTH_PERCENTILE))
    power = 1 << max(0, math.ceil(math.log2(max(percentile, 1.0))))
    return int(min(power, model_max_positions))


def require_public_gradient_source(partition: str, *, member_ids: Sequence[str] = ()) -> None:
    """Refuse a gradient-norm estimate taken anywhere but the public non-member split.

    This is a guard, not a comment: the clipping norm is published as part of the DP
    mechanism, so an estimate contaminated by protected records would leak a statistic of the
    private set no matter how the surrounding code is written.
    """
    if partition != GRADIENT_NORM_SOURCE:
        raise CalibrationError(
            f"the gradient-norm percentile may only be estimated on {GRADIENT_NORM_SOURCE};"
            f" {partition!r} was supplied. A clipping norm fitted to protected records"
            " becomes a public statistic of the private training set"
        )
    if member_ids:
        raise CalibrationError(
            f"{len(member_ids)} protected member id(s) reached the gradient-norm estimate,"
            f" first {member_ids[0]!r}; the estimate must contain non-members only"
        )


def resolve_dp_constants(
    *,
    target_epsilon: float,
    protected_training_set_size: int,
    batch_size: int,
    epochs: int,
    clipping_norm: float,
    gradient_norm_partition: str = GRADIENT_NORM_SOURCE,
    dp_seed: int = CALIBRATION_SEED,
) -> dict[str, JSONValue]:
    """Derive the three DP constants from the already-frozen epsilon row [AUTH: 00 §8.2].

    Nothing here invents a privacy target: epsilon is read from the frozen config and the
    accountant solves for the smallest sigma that meets it at the sample rate and step count
    the selected LoRA constants imply.
    """
    from src.dp.mechanism import calibrate_noise_multiplier, poisson_sample_rate, steps_for

    require_public_gradient_source(gradient_norm_partition)
    size = protected_training_set_size
    delta = resolve_delta(size)
    mechanism = calibrate_noise_multiplier(
        target_epsilon=target_epsilon,
        delta=delta,
        sample_rate=poisson_sample_rate(batch_size=batch_size, dataset_size=size),
        steps=steps_for(epochs=epochs, dataset_size=size, batch_size=batch_size),
        clipping_norm=clipping_norm,
        dp_seed=dp_seed,
    )
    return {
        "delta": delta,
        "clipping_norm": clipping_norm,
        "noise_multiplier": mechanism.noise_multiplier,
        "delta_n_definition": DELTA_N_DEFINITION,
        "protected_training_set_size": size,
        "gradient_norm_source": gradient_norm_partition,
    }


@dataclass(frozen=True)
class CalibrationOutcome:
    """The 11 values, bound to the measurements and the rule that produced them."""

    winner: GridMeasurement
    constants: Mapping[str, JSONValue]
    measurements: tuple[GridMeasurement, ...]
    rule: str = CALIBRATION_RULE

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "rule": self.rule,
            "winner": self.winner.as_dict(),
            "constants": dict(self.constants),
            "measurements": [m.as_dict() for m in self.measurements],
        }

    def identity(self) -> str:
        return sha256_canonical(self.as_dict())
