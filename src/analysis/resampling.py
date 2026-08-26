"""ONE bootstrap, used by every metric. Pairing is preserved by construction.

00 §32.1 requires record-level resampling within seed with at least 2,000 replicates;
§32.3 requires paired resampling whenever views derive from the same trained A_s; 00 §34B.1
requires DRY-G uncertainty to resample by registry-cell cluster, retaining all seed
observations of a sampled cell.

All three are the same operation over different *units*: draw units with replacement, then
recompute the statistic on the drawn units. Because a unit carries every view's value for
the same record (or every seed's row for the same cell), pairing cannot be lost by accident
— there is no code path that resamples one side independently.

Randomness is `numpy.random.Generator(PCG64(seed)).random()`, whose bit stream is a stability
guarantee, mapped to indices by `floor(u * n)`. The construction is written down here so a
replicate is reproducible from the seed alone [AUTH: 01 §30].
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class ResamplingError(ValueError):
    """A bootstrap cannot be run as specified."""


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate, percentile interval, and the replicate distribution behind them."""

    point: float
    low: float
    high: float
    standard_error: float
    replicates: NDArray[np.float64]
    n_units: int
    n_replicates: int

    @property
    def interval(self) -> tuple[float, float]:
        return (self.low, self.high)

    def excludes_zero_positive(self) -> bool:
        """The 00 §18 condition: the interval lies strictly above zero."""
        return self.low > 0.0

    def contains_zero(self) -> bool:
        return self.low <= 0.0 <= self.high


def _indices(rng: np.random.Generator, n: int, size: int) -> NDArray[np.int_]:
    return np.floor(rng.random(size) * n).astype(np.int_).clip(0, n - 1)


def bootstrap[Unit](
    statistic: Callable[[Sequence[Unit]], float],
    units: Sequence[Unit],
    *,
    n_replicates: int,
    seed: int,
    alpha: float,
) -> BootstrapResult:
    """Resample `units` with replacement `n_replicates` times.

    A unit is whatever must move together: one record's values across every view for a
    paired comparison, or one registry cell's whole seed block for a clustered null.
    """
    if not units:
        raise ResamplingError("no resampling units")
    if n_replicates < 1:
        raise ResamplingError("n_replicates must be positive")
    if not 0.0 < alpha < 1.0:
        raise ResamplingError(f"alpha {alpha!r} is not in (0, 1)")

    point = float(statistic(units))
    rng = np.random.Generator(np.random.PCG64(seed))
    n = len(units)
    draws = np.empty(n_replicates, dtype=np.float64)
    for replicate in range(n_replicates):
        picks = _indices(rng, n, n)
        draws[replicate] = statistic([units[int(i)] for i in picks])

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        raise ResamplingError("every bootstrap replicate was non-finite")
    low = float(np.percentile(finite, 100.0 * alpha / 2.0))
    high = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
    return BootstrapResult(
        point=point,
        low=low,
        high=high,
        standard_error=float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        replicates=draws,
        n_units=n,
        n_replicates=n_replicates,
    )


@dataclass(frozen=True)
class PairedUnit:
    """One resampling unit carrying every view's value for the same underlying record.

    Both sides of a paired difference travel in the same object, so a replicate cannot
    contain view A's record i with view B's record j.
    """

    key: str
    values: tuple[float, ...]


def paired_difference(index_left: int, index_right: int) -> Callable[[Sequence[PairedUnit]], float]:
    """Statistic factory: mean(values[left] - values[right]) over the drawn units."""

    def statistic(units: Sequence[PairedUnit]) -> float:
        if not units:
            raise ResamplingError("no units in replicate")
        return float(
            np.mean([unit.values[index_left] - unit.values[index_right] for unit in units])
        )

    return statistic


@dataclass(frozen=True)
class ClusterUnit:
    """One registry cell and every seed observation belonging to it [AUTH: 00 §34B.1]."""

    cell: str
    rows: tuple[tuple[float, ...], ...]


def cluster_rows(units: Sequence[ClusterUnit]) -> list[tuple[float, ...]]:
    """Flatten sampled clusters back into rows, retaining every seed of every sampled cell."""
    return [row for unit in units for row in unit.rows]
