"""The production LoRA trainer's decidable surface [AUTH: 00 §8.3; 01 §12, §17, §30].

Everything here runs on the CPU/dev lane: the planning half is pure, and the executing half
is checked for the one behaviour that matters without a GPU — that it refuses rather than
emulating a backend it does not have.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.materials import UncalibratedConstantError
from src.training.production import (
    FINAL_ADAPTER_ONLY,
    SHUFFLE_ONCE_PER_EPOCH,
    ProductionTrainingError,
    epoch_order,
    peft_available,
    resolve_dp_constants,
    resolve_training_constants,
    train_production_lora,
)

ROOT = Path(__file__).resolve().parents[2]


def test_an_uncalibrated_constant_is_refused_not_defaulted() -> None:
    """A missing learning rate must stop the run, not become a plausible number.

    This is the whole point of 01 §17: a trainer that silently defaulted would produce an
    adapter indistinguishable from one trained on a frozen value.
    """
    with pytest.raises(UncalibratedConstantError):
        resolve_training_constants(ROOT)
    with pytest.raises(UncalibratedConstantError):
        resolve_dp_constants(ROOT)


def test_the_data_order_is_reproducible_and_seed_dependent() -> None:
    ids = tuple(f"r{index:03d}" for index in range(64))
    first = epoch_order(ids, data_order_seed=4242, epoch=0)
    assert first == epoch_order(ids, data_order_seed=4242, epoch=0), "same inputs, same order"
    assert sorted(first) == sorted(ids), "a permutation drops and invents nothing"
    assert first != epoch_order(ids, data_order_seed=4242, epoch=1), "epochs reshuffle"
    assert first != epoch_order(ids, data_order_seed=9999, epoch=0), "the seed moves the order"


def test_the_frozen_policies_this_executor_implements_are_named() -> None:
    """It implements exactly the 00 §8.3 policies, and refuses to approximate another."""
    assert SHUFFLE_ONCE_PER_EPOCH == "SHUFFLE_ONCE_PER_EPOCH_FROM_DATA_ORDER_SEED"
    assert FINAL_ADAPTER_ONLY == "FINAL_ADAPTER_ONLY_NO_INTERMEDIATE_SELECTION"


def test_training_refuses_without_the_backend(tmp_path: Path) -> None:
    """No emulation: an adapter is produced by the real backend or not at all."""
    if peft_available():  # pragma: no cover - the H100 image resolves peft
        pytest.skip("peft resolves here; the refusal path is the CPU/dev lane's")
    from src.backend.loader import BackendUnavailable

    with pytest.raises((BackendUnavailable, ProductionTrainingError)):
        train_production_lora(
            ROOT,
            plan=None,  # type: ignore[arg-type]
            corpus={},
            adapter_directory=tmp_path / "adapter",
            training_plan=None,  # type: ignore[arg-type]
        )


def test_the_trainer_never_imports_the_backend_at_module_scope() -> None:
    """torch/peft stay behind the availability guard, so this module imports on any lane."""
    import ast
    import inspect

    import src.training.production as module

    tree = ast.parse(inspect.getsource(module))
    top_level = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    names = {getattr(n, "module", "") or "" for n in top_level}
    for name in names:
        assert name.split(".")[0] not in {"torch", "peft", "transformers", "opacus"}, name
