"""P — the shared paired bootstrap [AUTH: 00 §32.1, §32.3, §34B.1]."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.resampling import (
    ClusterUnit,
    PairedUnit,
    ResamplingError,
    bootstrap,
    cluster_rows,
    paired_difference,
)


def _units(left: list[float], right: list[float]) -> list[PairedUnit]:
    return [
        PairedUnit(key=f"r{i}", values=(a, b))
        for i, (a, b) in enumerate(zip(left, right, strict=True))
    ]


def test_the_point_estimate_is_the_observed_paired_mean() -> None:
    units = _units([1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5])
    result = bootstrap(paired_difference(0, 1), units, n_replicates=200, seed=1, alpha=0.05)
    assert result.point == pytest.approx(0.5)


def test_pairing_survives_resampling() -> None:
    """Each unit's two values move together, so a constant paired difference stays constant
    in every replicate. Independent resampling of the two sides would not [AUTH: 00 §32.3]."""
    left = list(np.arange(50, dtype=float))
    right = [value - 2.0 for value in left]
    result = bootstrap(
        paired_difference(0, 1), _units(left, right), n_replicates=300, seed=2, alpha=0.05
    )
    assert np.allclose(result.replicates, 2.0)
    assert result.interval == pytest.approx((2.0, 2.0))
    assert result.standard_error == pytest.approx(0.0, abs=1e-12)


def test_unpaired_resampling_would_not_be_degenerate() -> None:
    """The mutation this kills: resample the two sides independently and the constant
    difference acquires spurious variance."""
    rng = np.random.Generator(np.random.PCG64(3))
    left = np.arange(50, dtype=float)
    right = left - 2.0
    unpaired = np.array(
        [
            float(np.mean(left[rng.integers(0, 50, 50)]) - np.mean(right[rng.integers(0, 50, 50)]))
            for _ in range(300)
        ]
    )
    assert unpaired.std() > 0.5, "the unpaired control must be visibly wrong"


def test_replicates_are_reproducible_from_the_seed() -> None:
    units = _units([1.0, 4.0, 9.0, 16.0], [0.0, 1.0, 2.0, 3.0])
    first = bootstrap(paired_difference(0, 1), units, n_replicates=100, seed=7, alpha=0.05)
    second = bootstrap(paired_difference(0, 1), units, n_replicates=100, seed=7, alpha=0.05)
    third = bootstrap(paired_difference(0, 1), units, n_replicates=100, seed=8, alpha=0.05)
    assert np.array_equal(first.replicates, second.replicates)
    assert not np.array_equal(first.replicates, third.replicates)


def test_every_replicate_keeps_the_sample_size() -> None:
    seen: list[int] = []

    def statistic(units):  # type: ignore[no-untyped-def]
        seen.append(len(units))
        return float(len(units))

    bootstrap(statistic, _units([1.0] * 20, [0.0] * 20), n_replicates=50, seed=4, alpha=0.05)
    assert set(seen) == {20}, "a replicate changed the sample size"


def test_interval_flags_are_consistent() -> None:
    positive = bootstrap(
        paired_difference(0, 1),
        _units([5.0] * 30, [1.0] * 30),
        n_replicates=200,
        seed=5,
        alpha=0.05,
    )
    assert positive.excludes_zero_positive() and not positive.contains_zero()
    null = bootstrap(
        paired_difference(0, 1),
        _units([1.0, -1.0] * 15, [0.0] * 30),
        n_replicates=400,
        seed=6,
        alpha=0.05,
    )
    assert null.contains_zero()


def test_a_degenerate_sample_still_produces_an_interval() -> None:
    result = bootstrap(
        paired_difference(0, 1), _units([2.0], [1.0]), n_replicates=25, seed=9, alpha=0.05
    )
    assert result.interval == pytest.approx((1.0, 1.0))


@pytest.mark.parametrize(
    ("units", "replicates", "alpha"),
    [
        ([], 10, 0.05),
        ([PairedUnit("a", (1.0, 0.0))], 0, 0.05),
        ([PairedUnit("a", (1.0, 0.0))], 10, 1.5),
    ],
)
def test_bad_bootstrap_arguments_fail_closed(units, replicates, alpha) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ResamplingError):
        bootstrap(paired_difference(0, 1), units, n_replicates=replicates, seed=1, alpha=alpha)


# ------------------------------------------------------------------ cluster resampling
def test_a_sampled_cluster_brings_all_its_seed_rows() -> None:
    """DRY-G resamples registry cells and retains every seed observation [AUTH: 00 §34B.1]."""
    clusters = [
        ClusterUnit(cell=f"cell-{i}", rows=((float(i), 1.0), (float(i), 2.0), (float(i), 3.0)))
        for i in range(5)
    ]

    def statistic(units):  # type: ignore[no-untyped-def]
        rows = cluster_rows(units)
        assert len(rows) == 3 * len(units), "a cluster lost seed rows"
        return float(np.mean([row[0] for row in rows]))

    result = bootstrap(statistic, clusters, n_replicates=100, seed=11, alpha=0.05)
    assert result.point == pytest.approx(2.0)
    assert result.contains_zero() is False
