"""Deterministic S08 recovery fixtures. No model, no data, no network.

Every matrix is drawn from a seeded generator with a fixed seed, so the same call gives the
same numbers. Everything built here is non-evidentiary, and every recovery context issued here
carries `FIXTURE_ONLY_NOT_SCIENTIFIC`, so nothing built from these fixtures can be mistaken for
the accepted scientific method.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.merge.context import MergeExecutionContext, resolve_merge_context
from src.merge.family import MergeResult, MergeSpec, ReleaseFamily, build_release_family
from src.merge.updates import FixtureTaskVector, TaskVector, fixture_induced_update
from src.recovery.context import RecoveryExecutionContext, fixture_recovery_context
from src.recovery.settings import recovery_settings

Matrix = NDArray[Any]
REPO_ROOT = Path(__file__).resolve().parents[1]

LINEAR = "O1_LINEAR_TASK_ARITHMETIC"
SVD_TRUNC = "O3_SVD_TRUNC_MERGE"
DARE = "O2_DARE"

#: The fixture surface: two non-square target matrices.
SURFACE: dict[str, tuple[int, int]] = {"w0": (16, 12), "w1": (12, 16)}

#: Descendant counts the release-family builder may issue in fixtures. `1` is here so the
#: k = 1 STRUCTURAL_NA path can be exercised; the S08 observation factory still refuses it.
FIXTURE_K: tuple[int, ...] = (1, 2, 3, 4)


def merge_context() -> MergeExecutionContext:
    return resolve_merge_context(REPO_ROOT)


def recovery_context(*, n_iters: int = 200, residual_rank: int = 3) -> RecoveryExecutionContext:
    """A FIXTURE recovery context: the frozen algorithm at a fixture iteration count.

    The scientific values live in `configs/recovery/methods.json` and are pinned by
    `load_recovery_execution_context`; a fixture solve does not need 1000 sweeps and a rank-32
    residual model over 16x12 matrices. Results issued under this context carry
    `NON_EVIDENTIARY_S08_RECOVERY_FIXTURE`.
    """
    return fixture_recovery_context(
        recovery_settings(REPO_ROOT).document,
        c2_n_iters=n_iters,
        linear_residual_rank=residual_rank,
    )


def rewritten_recovery_context(**overrides: Any) -> RecoveryExecutionContext:
    """A fixture context from a modified config snapshot, for fail-closed attacks."""
    return fixture_recovery_context(recovery_settings(REPO_ROOT).document, **overrides)


def generator(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def low_rank(rank: int, rng: np.random.Generator, *, scale: float = 1.0) -> dict[str, Matrix]:
    """One exactly rank-`rank` induced update per target matrix, as ΔW = BA is."""
    tensors: dict[str, Matrix] = {}
    for name, (rows, columns) in SURFACE.items():
        tensors[name] = scale * (rng.normal(size=(rows, rank)) @ rng.normal(size=(rank, columns)))
    return tensors


def protected_and_partners(
    *,
    k: int,
    seed: int = 4242,
    protected_rank: int = 3,
    partner_rank: int = 3,
    ctx: MergeExecutionContext | None = None,
) -> tuple[FixtureTaskVector, list[FixtureTaskVector]]:
    """One protected constituent A and k partners B_i, all exactly low rank."""
    context = ctx or merge_context()
    rng = generator(seed)
    protected = fixture_induced_update(low_rank(protected_rank, rng), context=context, origin="A")
    partners = [
        fixture_induced_update(low_rank(partner_rank, rng), context=context, origin=f"B{i + 1}")
        for i in range(k)
    ]
    return protected, partners


def _specs(
    alphas: Sequence[float], *, operator: str = LINEAR, retained_rank: int | None = None
) -> list[MergeSpec]:
    return [
        MergeSpec(
            descendant_id=f"C{index + 1}",
            operator=operator,
            alpha=float(alpha),
            partner_id=f"B{index + 1}",
            retained_rank=retained_rank,
        )
        for index, alpha in enumerate(alphas)
    ]


def release_family(
    protected: TaskVector,
    partners: Sequence[TaskVector],
    alphas: Sequence[float],
    *,
    operator: str = LINEAR,
    retained_rank: int | None = None,
    ctx: MergeExecutionContext | None = None,
) -> ReleaseFamily:
    """A genuine issued S07 ReleaseFamily — the only thing S08 will observe."""
    context = ctx or merge_context()
    return build_release_family(
        protected=protected,
        partners={f"B{index + 1}": partner for index, partner in enumerate(partners)},
        specs=_specs(alphas, operator=operator, retained_rank=retained_rank),
        permitted_k=FIXTURE_K,
        context=context,
    )


def o3_family(
    protected: TaskVector,
    partners: Sequence[TaskVector],
    alphas: Sequence[float],
    *,
    retained_rank: int,
    ctx: MergeExecutionContext | None = None,
) -> ReleaseFamily:
    """An issued S07 O3 family at one shared retained rank."""
    return release_family(
        protected,
        partners,
        alphas,
        operator=SVD_TRUNC,
        retained_rank=retained_rank,
        ctx=ctx,
    )


def linear_descendant(
    protected: TaskVector,
    partner: TaskVector,
    alpha: float,
    *,
    ctx: MergeExecutionContext | None = None,
    descendant_id: str = "C1",
) -> MergeResult:
    """One issued S07 LINEAR descendant, for the known-partner regime."""
    return release_family(protected, [partner], [alpha], ctx=ctx).by_id(descendant_id)


def recoverable_bank(
    *, k: int, seed: int = 90210, ctx: MergeExecutionContext | None = None
) -> tuple[FixtureTaskVector, list[FixtureTaskVector]]:
    """A planted fixture whose residual model is correctly specified.

    The protected constituent is full-ish rank while every partner residual is exactly rank 3,
    so a rank-3 residual model is correctly specified. Correct specification is not by itself
    a claim about recovery: on THIS fixture the k = 4 instance is recovered exactly and the
    k = 2 instance is not, and the k = 2 solve does not beat the naive shared mean. That is a
    measured property of this fixture, not a theorem about two-descendant families — S08 adds
    no such theorem. See `tests/integration/test_s08_recovery_pipeline.py` for the values.
    """
    context = ctx or merge_context()
    rng = generator(seed)
    protected = fixture_induced_update(
        {name: rng.normal(size=shape) for name, shape in SURFACE.items()},
        context=context,
        origin="A",
    )
    partners = [
        fixture_induced_update(low_rank(3, rng), context=context, origin=f"B{i + 1}")
        for i in range(k)
    ]
    return protected, partners


def number(report: Any, key: str) -> float:
    """One scalar out of a JSONValue-typed diagnostic report, narrowed for the type checker."""
    value = report[key]
    assert isinstance(value, int | float) and not isinstance(value, bool), (key, value)
    return float(value)


def layer_map(report: Any, key: str) -> dict[str, float]:
    """One per-layer mapping out of a JSONValue-typed diagnostic report."""
    value = report[key]
    assert isinstance(value, dict), (key, value)
    return {str(name): float(str(entry)) for name, entry in value.items()}


def sequence(report: Any, key: str) -> list[float]:
    """One numeric sequence out of a JSONValue-typed diagnostic report."""
    value = report[key]
    assert isinstance(value, list | tuple), (key, value)
    return [float(str(item)) for item in value]


#: Historical alias: the S08 tests build a genuine issued family, never a loose list.
linear_family = release_family


def bind(observation: Any, source: Any, protected: TaskVector) -> Any:
    """`bind_truth` under a short name, for the many evaluator call sites."""
    from src.recovery.truth import bind_truth

    return bind_truth(observation=observation, source=source, protected_truth=protected)
