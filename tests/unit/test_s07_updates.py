"""B1 — the canonical induced-update object [AUTH: 00 §3.2, §22]."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from s07_fixtures import SURFACE, context, matrix, shuffled, surface_vector

from src.merge.updates import (
    FIXTURE_INDUCED_UPDATE,
    FixtureTaskVector,
    SurfaceMismatchError,
    TaskVectorError,
    combine,
    fixture_induced_update,
    require_compatible,
)

CTX = context()


def task_vector(tensors: Mapping[str, Any], *, origin: str = "unlabelled") -> FixtureTaskVector:
    """Local alias: every task vector in this module is a fixture."""
    return fixture_induced_update(tensors, context=CTX, origin=origin)


# ------------------------------------------------------------------ canonical ordering
def test_dict_order_never_reaches_identity() -> None:
    """Same logical mapping, different insertion order -> identical bytes and hashes."""
    forward = surface_vector("A")
    backward = shuffled(forward)
    assert forward.names == backward.names
    assert forward.content_identity() == backward.content_identity()
    assert forward.surface_identity() == backward.surface_identity()


def test_the_canonical_order_is_sorted_names() -> None:
    vector = surface_vector("A")
    assert vector.names == tuple(sorted(SURFACE))
    assert list(vector.shapes) == list(vector.names)


def test_content_and_surface_identities_answer_different_questions() -> None:
    first, second = surface_vector("A"), surface_vector("B")
    assert first.surface_identity() == second.surface_identity()
    assert first.content_identity() != second.content_identity()


def test_the_origin_label_is_not_part_of_identity() -> None:
    first = task_vector({"w": np.ones((3, 3))}, origin="A")
    second = task_vector({"w": np.ones((3, 3))}, origin="something_else")
    assert first.content_identity() == second.content_identity()
    assert first.origin != second.origin


# ------------------------------------------------------------------ induced updates only
@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.gate_proj.weight.lora_A",
        "model.layers.0.mlp.gate_proj.weight.lora_B",
        "lora_A",
    ],
)
def test_raw_lora_factors_are_refused(name: str) -> None:
    """The mutation this kills: merging factors instead of ΔW = BA [AUTH: 00 §22]."""
    with pytest.raises(TaskVectorError, match="raw LoRA factor"):
        task_vector({name: np.ones((4, 4))})


def test_a_fixture_vector_declares_its_provenance_class() -> None:
    updates = {name: matrix(f"c/{name}", shape) for name, shape in SURFACE.items()}
    vector = fixture_induced_update(updates, context=CTX, origin="fixture_adapter")
    assert vector.names == tuple(sorted(SURFACE))
    assert vector.dtype == CTX.arithmetic_dtype
    assert vector.provenance_class == FIXTURE_INDUCED_UPDATE


def test_a_caller_mutation_after_construction_cannot_rewrite_the_vector() -> None:
    source = {name: matrix(f"m/{name}", shape) for name, shape in SURFACE.items()}
    vector = task_vector(source)
    before = vector.content_identity()
    for array in source.values():
        array += 1.0
    assert vector.content_identity() == before


# ------------------------------------------------------------------ fail-closed construction
def test_an_empty_surface_is_refused() -> None:
    with pytest.raises(TaskVectorError):
        task_vector({})


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_refused(value: float) -> None:
    """B17 item 10: NaN/Inf may not enter a merge input."""
    array = np.ones((3, 3))
    array[1, 1] = value
    with pytest.raises(TaskVectorError, match="NaN or Inf"):
        task_vector({"w": array})


def test_a_non_two_dimensional_array_is_refused() -> None:
    with pytest.raises(TaskVectorError, match="2-D"):
        task_vector({"w": np.ones(5)})


def test_the_context_dtype_governs_the_stored_bytes() -> None:
    """A float64 input is cast to the frozen float32 artifact dtype, not refused."""
    vector = task_vector({"w": np.ones((3, 3), dtype=np.float64)})
    assert vector["w"].dtype == np.float32
    assert vector.dtype == CTX.arithmetic_dtype == "float32"


# ------------------------------------------------------------------ surface compatibility
def test_compatible_surfaces_return_one_identity() -> None:
    identity = require_compatible([surface_vector("A"), surface_vector("B")])
    assert identity == surface_vector("A").surface_identity()


def test_a_missing_tensor_is_refused_never_zero_filled() -> None:
    full = surface_vector("A")
    partial = task_vector({name: full[name] for name in full.names[:-1]})
    with pytest.raises(SurfaceMismatchError, match="missing"):
        require_compatible([full, partial])


def test_an_extra_tensor_is_refused() -> None:
    full = surface_vector("A")
    extended = task_vector({**{n: full[n] for n in full.names}, "extra.weight": np.ones((2, 2))})
    with pytest.raises(SurfaceMismatchError, match="absent from"):
        require_compatible([full, extended])


def test_a_shape_mismatch_is_refused_never_broadcast() -> None:
    full = surface_vector("A")
    reshaped = dict.fromkeys(full.names)
    tensors = {name: full[name] for name in full.names}
    first = full.names[0]
    tensors[first] = np.ones((full.shapes[first][0] + 1, full.shapes[first][1]))
    with pytest.raises(SurfaceMismatchError, match="never broadcast"):
        require_compatible([full, task_vector(tensors)])
    assert reshaped is not None


def test_a_single_participant_is_not_a_merge() -> None:
    with pytest.raises(SurfaceMismatchError, match="at least two"):
        require_compatible([surface_vector("A")])


# ------------------------------------------------------------------ combine
def test_combine_is_the_one_place_operator_arithmetic_happens() -> None:
    first, second = surface_vector("A"), surface_vector("B")
    merged = combine([(0.25, first), (0.75, second)], context=CTX, origin="c")
    for name in merged.names:
        assert np.allclose(merged[name], 0.25 * first[name] + 0.75 * second[name], rtol=1e-6)


def test_combine_refuses_a_non_finite_coefficient() -> None:
    first, second = surface_vector("A"), surface_vector("B")
    with pytest.raises(TaskVectorError, match="not finite"):
        combine([(np.inf, first), (1.0, second)], context=CTX, origin="c")


def test_combine_refuses_incompatible_surfaces() -> None:
    full = surface_vector("A")
    partial = task_vector({name: full[name] for name in full.names[:-1]})
    with pytest.raises(SurfaceMismatchError):
        combine([(0.5, full), (0.5, partial)], context=CTX, origin="c")


def test_the_frobenius_norm_is_over_the_whole_surface() -> None:
    vector = surface_vector("A")
    expected = float(np.sqrt(sum(float(np.sum(vector[n] ** 2)) for n in vector.names)))
    assert vector.frobenius_norm() == pytest.approx(expected)
