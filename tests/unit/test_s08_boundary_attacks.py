"""R13-R17 — the S07->S08 handoff, truth-binding, storage, config and provenance attacks.

Each block runs the attack for real and requires a refusal. Nothing here asserts that an
attack is *hard*; every one of them is closed by construction — a genuine factory-issued
object is required, an identity is re-derived, or authoritative storage is immutable.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from s08_fixtures import (
    FIXTURE_K,
    SURFACE,
    bind,
    linear_family,
    merge_context,
    o3_family,
    protected_and_partners,
    recoverable_bank,
    recovery_context,
    rewritten_recovery_context,
)

from src.merge.family import MergeSpec, build_release_family
from src.merge.updates import fixture_induced_update
from src.recovery.context import (
    RecoveryContextError,
    load_recovery_execution_context,
    resolve_recovery_context,
)
from src.recovery.evaluation import EvaluationError, o3_error_report, parameter_recovery_metrics
from src.recovery.observations import (
    ObservationError,
    observe_incomplete_fixed,
    observe_incomplete_varying,
    observe_known_partner,
)
from src.recovery.settings import recovery_settings
from src.recovery.solvers import (
    NAIVE_UNRESCALED_FIXED_ALPHA,
    recover_c1,
    recover_c2_fixed_alpha,
    recover_c2b_varying_alpha,
    recover_naive_unrescaled_fixed_alpha,
)
from src.recovery.truth import TruthBindingError, bind_truth

REPO_ROOT = Path(__file__).resolve().parents[2]
MCTX = merge_context()
RCTX = recovery_context()


def forged(candidate: object) -> Any:
    """Route a forgery through `Any` so the RUNTIME attack executes.

    mypy now rejects these call sites outright — the factories are typed on the genuine S07
    classes — which is itself part of the fix. The runtime refusal is what this module proves.
    """
    return candidate


# ==================================================================== R13 lineage attacks
def test_13_1_a_simple_namespace_pretending_to_be_a_merge_result_is_rejected() -> None:
    """Duck typing is what made every other forgery in this block possible."""
    protected, partners = protected_and_partners(k=2)
    genuine = linear_family(protected, partners[:1], [0.5]).by_id("C1")
    impostor = SimpleNamespace(
        update=genuine.update,
        alpha=0.5,
        operator="O1_LINEAR_TASK_ARITHMETIC",
        descendant_id="C1",
        partner_identity=partners[0].content_identity(),
        retained_rank=None,
    )
    with pytest.raises(ObservationError, match="factory-issued S07 MergeResult"):
        observe_known_partner(descendant=forged(impostor), partner_update=partners[0], context=RCTX)


def test_13_1b_a_loose_list_of_genuine_descendants_is_not_a_family() -> None:
    """Only a ReleaseFamily carries the 00 §3.4 same-A guarantee."""
    protected, partners = protected_and_partners(k=2)
    family = linear_family(protected, partners, [0.5, 0.5])
    with pytest.raises(ObservationError, match="ReleaseFamily"):
        observe_incomplete_fixed(family=forged(list(family.descendants)), context=RCTX)


def test_13_2_a_relabelled_fake_alpha_cannot_reach_a_solver() -> None:
    """The coefficient is read off the issued result, so a relabel changes nothing real."""
    protected, partners = protected_and_partners(k=2)
    genuine = linear_family(protected, partners[:1], [0.5]).by_id("C1")
    relabelled = SimpleNamespace(
        update=genuine.update,
        alpha=0.9,
        operator=genuine.operator,
        descendant_id=genuine.descendant_id,
        partner_identity=genuine.partner_identity,
        retained_rank=None,
    )
    with pytest.raises(ObservationError, match="factory-issued S07 MergeResult"):
        observe_known_partner(
            descendant=forged(relabelled), partner_update=partners[0], context=RCTX
        )
    honest = observe_known_partner(descendant=genuine, partner_update=partners[0], context=RCTX)
    assert honest.alphas == (0.5,)


def test_13_3_a_relabelled_fake_retained_rank_cannot_reach_a_solver() -> None:
    protected, partners = protected_and_partners(k=2, protected_rank=6)
    family = o3_family(protected, partners, [0.5, 0.5], retained_rank=2)
    forged_rank = SimpleNamespace(
        update=family.descendants[0].update,
        alpha=0.5,
        operator="O3_SVD_TRUNC_MERGE",
        descendant_id="O1",
        partner_identity=family.descendants[0].partner_identity,
        retained_rank=31,
    )
    with pytest.raises(ObservationError, match="factory-issued S07 MergeResult"):
        observe_known_partner(
            descendant=forged(forged_rank), partner_update=partners[0], context=RCTX
        )
    assert observe_incomplete_fixed(family=family, context=RCTX).retained_rank == 2


def test_13_4_a_relabelled_fake_partner_identity_cannot_reach_a_solver() -> None:
    protected, partners = protected_and_partners(k=2)
    genuine = linear_family(protected, partners[:1], [0.5]).by_id("C1")
    forged_partner = SimpleNamespace(
        update=genuine.update,
        alpha=genuine.alpha,
        operator=genuine.operator,
        descendant_id=genuine.descendant_id,
        #: claims B2 built this descendant, when B1 did
        partner_identity=partners[1].content_identity(),
        retained_rank=None,
    )
    with pytest.raises(ObservationError, match="factory-issued S07 MergeResult"):
        observe_known_partner(
            descendant=forged(forged_partner), partner_update=partners[1], context=RCTX
        )
    with pytest.raises(ObservationError, match="not the partner this descendant"):
        observe_known_partner(descendant=genuine, partner_update=partners[1], context=RCTX)


def test_13_5_one_descendant_listed_twice_is_not_a_k_equals_two_family() -> None:
    """S07 refuses the duplicate id, so [d, d] never becomes a family at all."""
    from src.merge.family import MergeFamilyError

    protected, partners = protected_and_partners(k=2)
    spec = MergeSpec(
        descendant_id="C1", operator="O1_LINEAR_TASK_ARITHMETIC", alpha=0.5, partner_id="B1"
    )
    with pytest.raises(MergeFamilyError, match="duplicate descendant id"):
        build_release_family(
            protected=protected,
            partners={"B1": partners[0]},
            specs=[spec, spec],
            permitted_k=FIXTURE_K,
            context=MCTX,
        )


def test_13_5b_the_same_release_under_two_ids_is_not_two_observations() -> None:
    """Relabelling one release is not an additional observation of A [AUTH: 00 §11]."""
    protected, partners = protected_and_partners(k=2)
    doubled = build_release_family(
        protected=protected,
        partners={"B1": partners[0], "B2": partners[0]},
        specs=[
            MergeSpec(
                descendant_id="C1",
                operator="O1_LINEAR_TASK_ARITHMETIC",
                alpha=0.5,
                partner_id="B1",
            ),
            MergeSpec(
                descendant_id="C2",
                operator="O1_LINEAR_TASK_ARITHMETIC",
                alpha=0.5,
                partner_id="B2",
            ),
        ],
        permitted_k=FIXTURE_K,
        context=MCTX,
    )
    with pytest.raises(ObservationError, match="same descendant more than once"):
        observe_incomplete_fixed(family=doubled, context=RCTX)


def test_13_6_descendants_of_different_protected_constituents_are_not_one_family() -> None:
    """00 §3.4: there is no route to a family whose descendants come from two different A.

    The S07 factory takes ONE protected constituent for the whole family, and `ReleaseFamily`
    has no public constructor. So the mixed-A case cannot be assembled at all — and a loose
    list of descendants drawn from two families is refused by the S08 factory, which requires
    a genuine issued family rather than a sequence.
    """
    first, partners = protected_and_partners(k=2)
    second, other_partners = protected_and_partners(k=2, seed=999)
    left = linear_family(first, partners, [0.5, 0.5])
    right = linear_family(second, other_partners, [0.5, 0.5])
    assert left.protected_identity != right.protected_identity

    with pytest.raises(TypeError, match="factory-issued"):
        type(left)()
    mixed = [left.descendants[0], right.descendants[1]]
    with pytest.raises(ObservationError, match="ReleaseFamily"):
        observe_incomplete_fixed(family=forged(mixed), context=RCTX)

    #: and the S07 guard that would catch it is live, not decorative
    source = (REPO_ROOT / "src" / "merge" / "family.py").read_text(encoding="utf-8")
    assert "was built from a different protected constituent" in source


def test_13_7_descendants_from_two_merge_contexts_are_not_one_family() -> None:
    """One family runs under one execution context, and mixing them has no route either."""
    from src.merge.context import load_merge_execution_context
    from src.merge.settings import merge_settings

    document = json.loads(json.dumps(dict(merge_settings(REPO_ROOT).document)))
    document["mask_scheme"] = {"status": "FROZEN", "value": "s07.dare-mask.v1", "note": "x"}
    other_context = load_merge_execution_context(document)

    protected, partners = protected_and_partners(k=2)
    here = linear_family(protected, partners, [0.5, 0.5])
    there = linear_family(protected, partners, [0.5, 0.5], ctx=other_context)
    assert here.context.identity() != there.context.identity()

    mixed = [here.descendants[0], there.descendants[1]]
    with pytest.raises(ObservationError, match="ReleaseFamily"):
        observe_incomplete_fixed(family=forged(mixed), context=RCTX)
    source = (REPO_ROOT / "src" / "merge" / "family.py").read_text(encoding="utf-8")
    assert "ran under a different execution context" in source


def test_13_8_a_family_mixing_operators_is_refused() -> None:
    protected, partners = protected_and_partners(k=2, protected_rank=6)
    mixed = build_release_family(
        protected=protected,
        partners={"B1": partners[0], "B2": partners[1]},
        specs=[
            MergeSpec(
                descendant_id="C1",
                operator="O1_LINEAR_TASK_ARITHMETIC",
                alpha=0.5,
                partner_id="B1",
            ),
            MergeSpec(
                descendant_id="C2",
                operator="O3_SVD_TRUNC_MERGE",
                alpha=0.5,
                partner_id="B2",
                retained_rank=2,
            ),
        ],
        permitted_k=FIXTURE_K,
        context=MCTX,
    )
    with pytest.raises(ObservationError, match="mix operators"):
        observe_incomplete_fixed(family=mixed, context=RCTX)


def test_13_9_a_permuted_family_issues_the_identical_canonical_observation() -> None:
    """R12: caller order is not scientific information, so it cannot move the bytes."""
    protected, partners = recoverable_bank(k=4)
    specs = [
        MergeSpec(
            descendant_id=f"C{index + 1}",
            operator="O1_LINEAR_TASK_ARITHMETIC",
            alpha=0.5,
            partner_id=f"B{index + 1}",
        )
        for index in range(4)
    ]
    catalog = {f"B{index + 1}": partner for index, partner in enumerate(partners)}
    forward = build_release_family(
        protected=protected,
        partners=catalog,
        specs=specs,
        permitted_k=FIXTURE_K,
        context=MCTX,
    )
    shuffled = build_release_family(
        protected=protected,
        partners=catalog,
        specs=[specs[2], specs[0], specs[3], specs[1]],
        permitted_k=FIXTURE_K,
        context=MCTX,
    )
    first = observe_incomplete_fixed(family=forward, context=RCTX)
    second = observe_incomplete_fixed(family=shuffled, context=RCTX)
    assert first.identity() == second.identity()
    assert first.descendant_ids == second.descendant_ids == ("C1", "C2", "C3", "C4")
    assert (
        recover_c2_fixed_alpha(first).content_identity()
        == recover_c2_fixed_alpha(second).content_identity()
    )


# ==================================================================== R14 binding attacks
def _case(seed: int = 90210, k: int = 4) -> tuple[Any, Any, Any, Any]:
    protected, partners = recoverable_bank(k=k, seed=seed)
    family = linear_family(protected, partners, [0.5] * k)
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    return protected, family, observation, bind(observation, family, protected)


def test_14_1_the_right_truth_for_the_right_observation_is_accepted() -> None:
    protected, _, observation, binding = _case()
    result = recover_c2_fixed_alpha(observation)
    assert binding.observation_identity == observation.identity()
    assert binding.protected_identity == protected.content_identity()
    metrics = parameter_recovery_metrics(result, binding)
    assert metrics["truth_binding_sha256"] == binding.identity()


def test_14_2_a_loose_task_vector_has_no_public_evaluation_api() -> None:
    protected, _, observation, _ = _case()
    result = recover_c2_fixed_alpha(observation)
    with pytest.raises(EvaluationError, match="EvaluationTruthBinding"):
        parameter_recovery_metrics(result, forged(protected))


def test_14_3_a_family_for_a1_cannot_be_bound_to_the_truth_of_a2() -> None:
    _, family, observation, _ = _case()
    other, _ = recoverable_bank(k=4, seed=31337)
    with pytest.raises(TruthBindingError, match="not the one this lineage was built from"):
        bind_truth(observation=observation, source=family, protected_truth=other)


def test_14_4_a_result_from_r1_cannot_be_scored_against_a_binding_for_r2() -> None:
    _, _, first, _ = _case(seed=90210)
    _, _, second, second_binding = _case(seed=31337)
    result = recover_c2_fixed_alpha(first)
    assert first.identity() != second.identity()
    with pytest.raises(TruthBindingError, match="issued for a different observation"):
        parameter_recovery_metrics(result, second_binding)


def test_14_5_a_binding_cannot_be_issued_from_an_unrelated_source_family() -> None:
    _, _, observation, _ = _case(seed=90210)
    other_protected, other_family, _, _ = _case(seed=31337)
    with pytest.raises(TruthBindingError, match="does not reproduce this observation"):
        bind_truth(observation=observation, source=other_family, protected_truth=other_protected)


def test_14_6_an_o3_result_cannot_be_scored_against_an_unrelated_a2() -> None:
    protected, partners = protected_and_partners(k=4, protected_rank=8, partner_rank=3)
    family = o3_family(protected, partners, [0.5] * 4, retained_rank=3)
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    other, _ = protected_and_partners(k=4, seed=31337, protected_rank=8, partner_rank=3)
    with pytest.raises(TruthBindingError, match="not the one this lineage was built from"):
        bind_truth(observation=observation, source=family, protected_truth=other)


def test_14_7_an_o3_result_at_rank_s1_cannot_use_a_binding_from_rank_s2() -> None:
    protected, partners = protected_and_partners(k=4, protected_rank=8, partner_rank=3)
    coarse = o3_family(protected, partners, [0.5] * 4, retained_rank=2)
    fine = o3_family(protected, partners, [0.5] * 4, retained_rank=5)
    coarse_observation = observe_incomplete_fixed(family=coarse, context=RCTX)
    fine_observation = observe_incomplete_fixed(family=fine, context=RCTX)
    fine_binding = bind(fine_observation, fine, protected)
    result = recover_c2_fixed_alpha(coarse_observation)
    assert result.retained_rank == 2 and fine_binding.retained_rank == 5
    with pytest.raises(TruthBindingError):
        o3_error_report(result, fine_binding)


def test_14_8_an_o3_floor_cannot_come_from_an_unrelated_release_family() -> None:
    protected, partners = protected_and_partners(k=4, protected_rank=8, partner_rank=3)
    family = o3_family(protected, partners, [0.5] * 4, retained_rank=3)
    other, other_partners = protected_and_partners(
        k=4, seed=31337, protected_rank=8, partner_rank=3
    )
    unrelated = o3_family(other, other_partners, [0.5] * 4, retained_rank=3)
    observation = observe_incomplete_fixed(family=family, context=RCTX)
    binding = bind(observation, family, protected)
    result = recover_c2_fixed_alpha(observation)
    with pytest.raises(TruthBindingError, match="not the one the binding was issued for"):
        o3_error_report(result, binding, o3_source=unrelated)


def test_14_9_the_binding_never_leaks_into_the_attacker_view() -> None:
    protected, _, observation, binding = _case()
    rendered = str(observation.as_dict()) + str(recover_c2_fixed_alpha(observation).as_dict())
    assert protected.content_identity() not in rendered
    assert binding.identity() not in rendered
    assert not hasattr(observation, "protected_truth")


# ==================================================================== R15 storage attacks
def test_15_1_observation_authoritative_bytes_cannot_be_written_back() -> None:
    _, _, observation, _ = _case()
    update = observation.updates[0]
    recorded = (update.content_identity(), observation.identity())
    replacement = np.ones(SURFACE["w0"], dtype=np.float32).tobytes()

    with pytest.raises(AttributeError):
        update.__setattr__("_entries", ())
    #: the authoritative container is a tuple, so there is no item assignment at all
    with pytest.raises(TypeError):
        forged(update._entries)[0] = ("w0", SURFACE["w0"], "float32", replacement)
    #: and the mapping a property builds is a fresh object
    shapes = update.shapes
    shapes["w0"] = (1, 1)
    assert update.shapes["w0"] == SURFACE["w0"]
    assert (update.content_identity(), observation.identity()) == recorded


def test_15_2_recovery_result_authoritative_bytes_cannot_be_written_back() -> None:
    _, _, observation, _ = _case()
    result = recover_c2_fixed_alpha(observation)
    recorded = (result.content_identity(), result.identity())
    name = result.names[0]

    with pytest.raises(AttributeError):
        result.__setattr__("_entries", ())
    with pytest.raises(TypeError):
        forged(result._entries)[0] = ("x", (1, 1), "float32", b"")
    shapes = result.shapes
    shapes[name] = (1, 1)
    parameters = result.method_parameters
    parameters["assumed_fixed_alpha"] = 0.99
    assert result.shapes[name] == SURFACE[name]
    assert result.method_parameters == {}
    assert (result.content_identity(), result.identity()) == recorded


def test_15_3_a_read_only_view_refuses_writes_and_the_write_flag() -> None:
    _, _, observation, _ = _case()
    result = recover_c2_fixed_alpha(observation)
    name = result.names[0]
    recorded = result.content_identity()
    with pytest.raises(ValueError, match="read-only|assignment destination"):
        result[name][0, 0] = 7.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        result[name].setflags(write=True)
    with pytest.raises(ValueError, match="read-only|assignment destination"):
        observation.updates[0][name][0, 0] = 7.0
    assert result.content_identity() == recorded


def test_15_4_a_detached_copy_is_writable_and_reaches_nothing() -> None:
    _, _, observation, _ = _case()
    result = recover_c2_fixed_alpha(observation)
    name = result.names[0]
    recorded = result.content_identity()
    scratch = result.copy_of(name)
    scratch[:] = 99.0
    assert scratch.flags.owndata and scratch.flags.writeable
    assert result.content_identity() == recorded
    assert not np.allclose(result[name], scratch)

    observed = observation.updates[0].copy_of(name)
    observed[:] = -1.0
    assert observation.updates[0].content_identity() != ""
    assert not np.allclose(observation.updates[0][name], observed)


# ==================================================================== R16 config attacks
def test_16_1_a_switching_mapping_is_canonicalised_into_one_snapshot() -> None:
    from collections.abc import Iterator, Mapping

    base = dict(recovery_settings(REPO_ROOT).document)

    class Switching(Mapping[str, object]):
        def __init__(self) -> None:
            self.reads = 0

        def __iter__(self) -> Iterator[str]:
            return iter(base)

        def __len__(self) -> int:
            return len(base)

        def __getitem__(self, key: str) -> object:
            if key == "linear_residual_rank":
                self.reads += 1
                return {
                    "status": "FROZEN",
                    "value": 32 if self.reads % 2 else 5,
                    "note": "switching",
                }
            return base[key]

    switching = Switching()
    context = load_recovery_execution_context(forged(switching))
    assert switching.reads == 1
    assert context.linear_residual_rank == 32


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("c2_n_iters", 7),
        ("linear_residual_rank", 5),
        ("conditioning_numerical_rank_tolerance", 0.25),
        ("arithmetic_dtype", "float64"),
    ],
)
def test_16_2_a_scientific_context_refuses_a_changed_pinned_value(key: str, value: object) -> None:
    document = json.loads(json.dumps(dict(recovery_settings(REPO_ROOT).document)))
    document[key] = {"status": "FROZEN", "value": value, "note": "attack"}
    with pytest.raises(RecoveryContextError):
        load_recovery_execution_context(document)


def test_16_6_a_config_sha_cannot_be_injected() -> None:
    """The hash is computed from the snapshot; a document field claiming one is ignored."""
    document = json.loads(json.dumps(dict(recovery_settings(REPO_ROOT).document)))
    honest = load_recovery_execution_context(document)
    document["config_sha256"] = {"status": "FROZEN", "value": "0" * 64, "note": "attack"}
    injected = load_recovery_execution_context(document)
    assert injected.config_sha256 != "0" * 64
    assert injected.config_sha256 != honest.config_sha256, "the extra key is part of the snapshot"


def test_16_7_an_unsupported_method_version_fails_closed() -> None:
    with pytest.raises(RecoveryContextError, match="C2 method version"):
        rewritten_recovery_context(c2_method_version="s08.c2-core.v99")


def test_16_8_the_repository_context_is_the_pinned_scientific_one() -> None:
    context = resolve_recovery_context(REPO_ROOT)
    assert context.is_scientific
    assert (context.c2_n_iters, context.linear_residual_rank) == (1000, 32)
    assert context.conditioning_rank_tolerance == 1e-06
    assert context.arithmetic_dtype == "float32"


# ==================================================================== R17 comparator alpha
def test_17_the_assumed_comparator_alpha_is_bound_into_the_provenance() -> None:
    """R8: two runs of one method under different predeclared alphas are distinguishable."""
    protected, partners = recoverable_bank(k=4)
    family = linear_family(protected, partners, [0.35, 0.45, 0.55, 0.65])
    observation = observe_incomplete_varying(family=family, context=RCTX)

    low = recover_naive_unrescaled_fixed_alpha(observation, assumed_fixed_alpha=0.4)
    high = recover_naive_unrescaled_fixed_alpha(observation, assumed_fixed_alpha=0.6)

    assert low.method == high.method == NAIVE_UNRESCALED_FIXED_ALPHA
    assert low.observation_identity == high.observation_identity
    assert low.method_parameters == {"assumed_fixed_alpha": 0.4}
    assert high.method_parameters == {"assumed_fixed_alpha": 0.6}
    assert low.as_dict()["method_parameters"] == {"assumed_fixed_alpha": 0.4}
    assert high.as_dict()["method_parameters"] == {"assumed_fixed_alpha": 0.6}
    assert low.identity() != high.identity()


def test_17b_the_parameter_moves_the_identity_even_at_identical_bytes() -> None:
    """The provenance is distinguishable BEFORE the recovered bytes are compared."""
    protected, partners = recoverable_bank(k=4)
    family = linear_family(protected, partners, [0.35, 0.45, 0.55, 0.65])
    observation = observe_incomplete_varying(family=family, context=RCTX)
    result = recover_naive_unrescaled_fixed_alpha(observation, assumed_fixed_alpha=0.4)
    spectral = recover_c2b_varying_alpha(observation)

    assert "assumed_fixed_alpha" not in spectral.method_parameters
    assert spectral.as_dict()["method_parameters"] == {}
    assert result.identity() != spectral.identity()


def test_17c_the_comparator_alpha_has_no_default_and_is_keyword_only() -> None:
    import inspect

    signature = inspect.signature(recover_naive_unrescaled_fixed_alpha)
    parameter = signature.parameters["assumed_fixed_alpha"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_17d_the_configured_comparator_alpha_remains_uncalibrated() -> None:
    from src.materials import UncalibratedConstantError, material

    with pytest.raises(UncalibratedConstantError):
        material(recovery_settings(REPO_ROOT), "naive_unrescaled_assumed_alpha")


# ==================================================================== C1 binding
def test_a_c1_binding_proves_the_known_partner_regime_too() -> None:
    protected, partners = protected_and_partners(k=2)
    descendant = linear_family(protected, partners[:1], [0.5]).by_id("C1")
    observation = observe_known_partner(
        descendant=descendant, partner_update=partners[0], context=RCTX
    )
    binding = bind(observation, descendant, protected)
    assert binding.regime == "Z_HIGH"
    metrics = parameter_recovery_metrics(recover_c1(observation), binding)
    assert isinstance(metrics["e_f"], float)

    other, _ = protected_and_partners(k=2, seed=31337)
    with pytest.raises(TruthBindingError):
        bind_truth(observation=observation, source=descendant, protected_truth=other)


def test_a_family_cannot_found_a_known_partner_binding() -> None:
    protected, partners = protected_and_partners(k=2)
    descendant = linear_family(protected, partners[:1], [0.5]).by_id("C1")
    observation = observe_known_partner(
        descendant=descendant, partner_update=partners[0], context=RCTX
    )
    family = linear_family(protected, partners, [0.5, 0.5])
    with pytest.raises(TruthBindingError, match="incomplete-lineage observation"):
        bind_truth(observation=observation, source=family, protected_truth=protected)


def test_a_non_task_vector_truth_is_refused() -> None:
    _, _, observation, _ = _case()
    raw = {name: np.zeros(shape) for name, shape in SURFACE.items()}
    with pytest.raises(TruthBindingError, match="factory-issued S07 task vector"):
        bind_truth(observation=observation, source=forged(None), protected_truth=forged(raw))
    zero = fixture_induced_update(raw, context=MCTX, origin="zero")
    with pytest.raises(TruthBindingError, match="MergeResult or ReleaseFamily"):
        bind_truth(observation=observation, source=forged(None), protected_truth=zero)
