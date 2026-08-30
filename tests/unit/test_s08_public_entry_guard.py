"""F1 — every public recovery entry point requires a factory-issued ObservedLineage.

The observation factories already refuse a duck-typed S07 descendant. That closes the front
door but not the side one: an object exposing exactly the attributes a solver reads could be
handed straight to `recover_*` and skip issuance entirely, carrying raw matrices — the true
protected constituent among them — under a borrowed observation identity.

Each fake below declares the CORRECT regime for the function it attacks, so a refusal here
proves the type guard and not a regime mismatch. Every fake also returns a genuine
observation's SHA from `identity()`, so a refusal is not merely an identity check either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from s08_fixtures import (
    linear_family,
    o3_family,
    protected_and_partners,
    recoverable_bank,
    recovery_context,
)

from src.recovery.observations import (
    Z_HIGH,
    Z_INCOMPLETE_FIXED,
    Z_INCOMPLETE_VARYING_KNOWN,
    ObservedLineage,
    observe_incomplete_fixed,
    observe_incomplete_varying,
    observe_known_partner,
)
from src.recovery.solvers import (
    RecoveryInputError,
    recover_c1,
    recover_c2_fixed_alpha,
    recover_c2_naive_shared_mean,
    recover_c2b_varying_alpha,
    recover_naive_unrescaled_fixed_alpha,
    rescale_observations,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RCTX = recovery_context()


class FakeLineage:
    """Every attribute a solver reads, and nothing that was ever stripped.

    Deliberately NOT an `ObservedLineage`: this is the object the guard exists for. Its
    updates are raw task vectors — including the true protected constituent — because that is
    the whole point of skipping issuance.
    """

    def __init__(
        self,
        *,
        regime: str,
        operator: str,
        updates: tuple[Any, ...],
        alphas: tuple[float, ...],
        identity: str,
        retained_rank: int | None = None,
        partner: Any = None,
    ) -> None:
        self.regime = regime
        self.operator = operator
        self.updates = updates
        self.alphas = alphas
        self.retained_rank = retained_rank
        self.context = RCTX
        self._identity = identity
        self._partner = partner

    @property
    def k(self) -> int:
        return len(self.updates)

    @property
    def fixed_alpha(self) -> float:
        return self.alphas[0]

    @property
    def has_partner(self) -> bool:
        return self._partner is not None

    @property
    def partner(self) -> Any:
        return self._partner

    @property
    def surface_identity(self) -> str:
        return str(self.updates[0].surface_identity())

    def identity(self) -> str:
        """A GENUINE observation's SHA, so no identity check can be what refuses this."""
        return str(self._identity)

    def as_dict(self) -> dict[str, Any]:
        return {"regime": self.regime, "observation_sha256": self._identity}


def _genuine_fixed() -> tuple[Any, Any, ObservedLineage]:
    protected, partners = recoverable_bank(k=4)
    family = linear_family(protected, partners, [0.5] * 4)
    return protected, partners, observe_incomplete_fixed(family=family, context=RCTX)


def _genuine_varying() -> tuple[Any, Any, ObservedLineage]:
    protected, partners = recoverable_bank(k=4)
    family = linear_family(protected, partners, [0.35, 0.45, 0.55, 0.65])
    return protected, partners, observe_incomplete_varying(family=family, context=RCTX)


def _forged(candidate: FakeLineage) -> Any:
    """Route through `Any`: mypy rejects the call, and the runtime must too."""
    return candidate


# ==================================================================== the six attacks
def test_a_fake_fixed_alpha_observation_cannot_reach_the_spectral_solver() -> None:
    protected, _, genuine = _genuine_fixed()
    fake = FakeLineage(
        regime=Z_INCOMPLETE_FIXED,
        operator="O1_LINEAR_TASK_ARITHMETIC",
        #: raw true A, four times over — exactly what issuance would never carry
        updates=(protected, protected, protected, protected),
        alphas=(0.5,) * 4,
        identity=genuine.identity(),
    )
    assert fake.regime == genuine.regime
    with pytest.raises(RecoveryInputError, match="factory-issued ObservedLineage"):
        recover_c2_fixed_alpha(_forged(fake))


def test_a_fake_fixed_alpha_observation_cannot_reach_the_naive_comparator() -> None:
    protected, _, genuine = _genuine_fixed()
    fake = FakeLineage(
        regime=Z_INCOMPLETE_FIXED,
        operator="O1_LINEAR_TASK_ARITHMETIC",
        updates=(protected, protected),
        alphas=(0.5, 0.5),
        identity=genuine.identity(),
    )
    with pytest.raises(RecoveryInputError, match="factory-issued ObservedLineage"):
        recover_c2_naive_shared_mean(_forged(fake))


def test_a_fake_varying_alpha_observation_cannot_reach_c2b() -> None:
    protected, _, genuine = _genuine_varying()
    fake = FakeLineage(
        regime=Z_INCOMPLETE_VARYING_KNOWN,
        operator="O1_LINEAR_TASK_ARITHMETIC",
        updates=(protected,) * 4,
        alphas=(0.35, 0.45, 0.55, 0.65),
        identity=genuine.identity(),
    )
    assert fake.regime == genuine.regime
    with pytest.raises(RecoveryInputError, match="factory-issued ObservedLineage"):
        recover_c2b_varying_alpha(_forged(fake))
    with pytest.raises(RecoveryInputError, match="factory-issued ObservedLineage"):
        rescale_observations(_forged(fake))


def test_a_fake_comparator_observation_cannot_reach_the_unrescaled_naive_method() -> None:
    protected, _, genuine = _genuine_varying()
    fake = FakeLineage(
        regime=Z_INCOMPLETE_VARYING_KNOWN,
        operator="O1_LINEAR_TASK_ARITHMETIC",
        updates=(protected,) * 4,
        alphas=(0.35, 0.45, 0.55, 0.65),
        identity=genuine.identity(),
    )
    with pytest.raises(RecoveryInputError, match="factory-issued ObservedLineage"):
        recover_naive_unrescaled_fixed_alpha(_forged(fake), assumed_fixed_alpha=0.5)


def test_a_fake_known_partner_observation_cannot_reach_c1() -> None:
    protected, partners = protected_and_partners(k=2)
    descendant = linear_family(protected, partners[:1], [0.5]).by_id("C1")
    genuine = observe_known_partner(descendant=descendant, partner_update=partners[0], context=RCTX)
    fake = FakeLineage(
        regime=Z_HIGH,
        operator="O1_LINEAR_TASK_ARITHMETIC",
        updates=(descendant.update,),
        alphas=(0.5,),
        identity=genuine.identity(),
        partner=partners[0],
    )
    assert fake.regime == genuine.regime == Z_HIGH and fake.has_partner
    with pytest.raises(RecoveryInputError, match="factory-issued ObservedLineage"):
        recover_c1(_forged(fake))


def test_a_fake_o3_observation_cannot_reach_either_o3_recovery_path() -> None:
    protected, partners = protected_and_partners(k=4, protected_rank=8, partner_rank=3)
    fixed_family = o3_family(protected, partners, [0.5] * 4, retained_rank=3)
    varying_family = o3_family(protected, partners, [0.35, 0.45, 0.55, 0.65], retained_rank=3)
    fixed_genuine = observe_incomplete_fixed(family=fixed_family, context=RCTX)
    varying_genuine = observe_incomplete_varying(family=varying_family, context=RCTX)

    fixed_fake = FakeLineage(
        regime=Z_INCOMPLETE_FIXED,
        operator="O3_SVD_TRUNC_MERGE",
        updates=(protected,) * 4,
        alphas=(0.5,) * 4,
        identity=fixed_genuine.identity(),
        retained_rank=3,
    )
    varying_fake = FakeLineage(
        regime=Z_INCOMPLETE_VARYING_KNOWN,
        operator="O3_SVD_TRUNC_MERGE",
        updates=(protected,) * 4,
        alphas=(0.35, 0.45, 0.55, 0.65),
        identity=varying_genuine.identity(),
        retained_rank=3,
    )
    with pytest.raises(RecoveryInputError, match="factory-issued ObservedLineage"):
        recover_c2_fixed_alpha(_forged(fixed_fake))
    with pytest.raises(RecoveryInputError, match="factory-issued ObservedLineage"):
        recover_c2b_varying_alpha(_forged(varying_fake))


# ==================================================================== completeness
def test_every_public_entry_point_that_issues_a_result_is_guarded() -> None:
    """Enumerated from source, so a new public solver cannot quietly skip the guard."""
    import ast
    import inspect

    import src.recovery.solvers as solvers

    tree = ast.parse(Path(inspect.getfile(solvers)).read_text(encoding="utf-8"))
    unguarded: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        body = ast.dump(node)
        if "_issue_result" not in body:
            continue
        if "_require_observed_lineage" not in body:
            unguarded.append(node.name)
    assert not unguarded, unguarded


def test_the_guard_runs_before_any_scientific_field_is_read() -> None:
    """A fake whose every attribute raises still gets a clean refusal, not an AttributeError."""

    class Exploding:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"{name} was read before the type guard ran")

    for solver in (
        recover_c1,
        recover_c2_fixed_alpha,
        recover_c2_naive_shared_mean,
        recover_c2b_varying_alpha,
        rescale_observations,
    ):
        with pytest.raises(RecoveryInputError):
            solver(_forged(Exploding()))  # type: ignore[arg-type]
    with pytest.raises(RecoveryInputError):
        recover_naive_unrescaled_fixed_alpha(
            _forged(Exploding()),  # type: ignore[arg-type]
            assumed_fixed_alpha=0.5,
        )


def test_result_issuance_itself_refuses_a_non_observation() -> None:
    """Defence in depth: `_issue_result` re-checks rather than trusting its caller."""
    import numpy as np

    from src.recovery.solvers import _issue_result

    _, _, genuine = _genuine_fixed()
    fake = FakeLineage(
        regime=Z_INCOMPLETE_FIXED,
        operator="O1_LINEAR_TASK_ARITHMETIC",
        updates=genuine.updates,
        alphas=genuine.alphas,
        identity=genuine.identity(),
    )
    with pytest.raises(RecoveryInputError):
        _issue_result(
            method="C2_SPECTRAL_FIXED",
            target_class="ORIGINAL_PROTECTED_CONSTITUENT",
            tensors={"w0": np.zeros((16, 12), dtype=np.float32)},
            observation=_forged(fake),
            context=RCTX,
            residual_rank=3,
            iterations=10,
        )


# ==================================================================== genuine still works
def test_a_genuine_issued_observation_still_runs_every_public_method() -> None:
    protected, partners = recoverable_bank(k=4)
    fixed = observe_incomplete_fixed(
        family=linear_family(protected, partners, [0.5] * 4), context=RCTX
    )
    varying = observe_incomplete_varying(
        family=linear_family(protected, partners, [0.35, 0.45, 0.55, 0.65]), context=RCTX
    )
    descendant = linear_family(protected, partners[:1], [0.5]).by_id("C1")
    high = observe_known_partner(descendant=descendant, partner_update=partners[0], context=RCTX)

    assert recover_c1(high).method == "C1_EXACT_LINEAR_ORACLE"
    assert recover_c2_fixed_alpha(fixed).method == "C2_SPECTRAL_FIXED"
    assert recover_c2_naive_shared_mean(fixed).method == "C2_NAIVE_SHARED_MEAN"
    assert recover_c2b_varying_alpha(varying).method == "C2B_RESCALED_SPECTRAL_REPRODUCTION"
    assert (
        recover_naive_unrescaled_fixed_alpha(varying, assumed_fixed_alpha=0.5).method
        == "NAIVE_UNRESCALED_FIXED_ALPHA"
    )
    assert set(rescale_observations(varying)) == {"w0", "w1"}


def test_compared_methods_still_share_one_issued_observation() -> None:
    _, _, observation = _genuine_fixed()
    spectral = recover_c2_fixed_alpha(observation)
    naive = recover_c2_naive_shared_mean(observation)
    assert spectral.observation_identity == naive.observation_identity == observation.identity()
