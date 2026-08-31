"""The pre-data calibration rule is total, deterministic, and outcome-blind.

These run on the CPU/dev lane: the rule is pure, so it can be proved correct before the
measurements it will be applied to exist [AUTH: 00 §34B.1A; 01 §3.2].
"""

from __future__ import annotations

import pytest

from src.training.calibration import (
    BATCH_SIZES,
    EPOCHS,
    LEARNING_RATES,
    CalibrationError,
    GridMeasurement,
    GridPoint,
    calibration_grid,
    feasible,
    resolve_delta,
    resolve_dp_constants,
    resolve_sequence_length,
    select_winner,
)

H100_BYTES = 85_520_809_984


def _m(lr: float, bs: int, ep: int, **over: float | bool | int) -> GridMeasurement:
    fields: dict[str, float | bool | int] = {
        "stable": True,
        "final_train_loss": 2.0,
        "heldout_loss": 2.0,
        "peak_memory_bytes": 40_000_000_000,
        "wall_clock_seconds": 3600.0,
        "gradient_norm_percentile": 1.0,
    }
    fields.update(over)
    return GridMeasurement(point=GridPoint(lr, bs, ep), **fields)  # type: ignore[arg-type]


def test_the_grid_is_declared_and_complete() -> None:
    grid = calibration_grid()
    assert len(grid) == len(LEARNING_RATES) * len(BATCH_SIZES) * len(EPOCHS) == 27
    assert len(set(grid)) == len(grid)


def test_unstable_or_infeasible_points_are_rejected_before_any_loss_is_compared() -> None:
    assert not feasible(_m(1e-4, 8, 2, stable=False), device_memory_bytes=H100_BYTES)
    assert not feasible(_m(1e-4, 8, 2, heldout_loss=float("nan")), device_memory_bytes=H100_BYTES)
    assert not feasible(
        _m(1e-4, 8, 2, peak_memory_bytes=H100_BYTES), device_memory_bytes=H100_BYTES
    )
    assert not feasible(_m(1e-4, 8, 2, wall_clock_seconds=10**9), device_memory_bytes=H100_BYTES)
    assert feasible(_m(1e-4, 8, 2), device_memory_bytes=H100_BYTES)


def test_the_winner_is_lowest_heldout_loss() -> None:
    best = _m(2e-4, 16, 3, heldout_loss=1.5)
    chosen = select_winner(
        [_m(5e-5, 4, 1, heldout_loss=2.5), best, _m(1e-4, 8, 2, heldout_loss=1.9)],
        device_memory_bytes=H100_BYTES,
    )
    assert chosen is best


def test_ties_break_deterministically_and_totally() -> None:
    """Equal loss -> memory, then wall clock, then lexicographic. No residual ambiguity."""
    a = _m(1e-4, 8, 2, peak_memory_bytes=30_000_000_000)
    b = _m(5e-5, 4, 1, peak_memory_bytes=40_000_000_000)
    assert select_winner([b, a], device_memory_bytes=H100_BYTES) is a

    c = _m(1e-4, 8, 2, wall_clock_seconds=100.0)
    d = _m(5e-5, 4, 1, wall_clock_seconds=200.0)
    assert select_winner([d, c], device_memory_bytes=H100_BYTES) is c

    # fully identical measurements: the grid order decides, and it is total
    e, f = _m(5e-5, 4, 1), _m(2e-4, 16, 3)
    assert select_winner([f, e], device_memory_bytes=H100_BYTES) is e
    assert select_winner([e, f], device_memory_bytes=H100_BYTES) is e


def test_a_failed_sweep_produces_no_winner() -> None:
    """If nothing is stable the constants stay uncalibrated rather than defaulting."""
    with pytest.raises(CalibrationError):
        select_winner([_m(1e-4, 8, 2, stable=False)], device_memory_bytes=H100_BYTES)
    with pytest.raises(CalibrationError):
        select_winner([], device_memory_bytes=H100_BYTES)


def test_delta_is_comfortably_below_one_over_n() -> None:
    """The whole point of the delta rule: delta << 1/N for the real training-set size."""
    for size in (1_000, 30_000, 250_000):
        delta = resolve_delta(size)
        assert 0 < delta < 1.0 / size
    assert resolve_delta(30_000) == resolve_delta(30_000), "deterministic"
    with pytest.raises(CalibrationError):
        resolve_delta(1)


def test_sequence_length_covers_the_corpus_and_respects_the_model_cap() -> None:
    lengths = [100] * 99 + [900]
    assert resolve_sequence_length(lengths, model_max_positions=4096) == 128
    assert resolve_sequence_length(lengths, model_max_positions=64) == 64
    with pytest.raises(CalibrationError):
        resolve_sequence_length([], model_max_positions=4096)


def test_the_dp_constants_are_derived_from_the_frozen_epsilon_not_invented() -> None:
    """epsilon comes in frozen; sigma is solved for, and the achieved epsilon respects it."""
    from src.dp.mechanism import DPMechanism, account

    resolved = resolve_dp_constants(
        target_epsilon=8.0,
        training_set_size=30_000,
        batch_size=8,
        epochs=2,
        clipping_norm=1.0,
    )
    assert set(resolved) == {"delta", "clipping_norm", "noise_multiplier"}
    assert resolved["noise_multiplier"] > 0

    from src.dp.mechanism import poisson_sample_rate, steps_for

    achieved = account(
        DPMechanism(
            adjacency="SAMPLE_LEVEL_ADD_REMOVE_ONE_RECORD",
            delta=resolved["delta"],
            clipping_norm=resolved["clipping_norm"],
            noise_multiplier=resolved["noise_multiplier"],
            sample_rate=poisson_sample_rate(batch_size=8, dataset_size=30_000),
            steps=steps_for(epochs=2, dataset_size=30_000, batch_size=8),
            dp_seed=101,
        )
    )
    assert achieved.achieved_epsilon <= 8.0, "the accountant meets the frozen 00 §8.2 target"
