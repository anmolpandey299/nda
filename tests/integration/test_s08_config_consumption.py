"""Every S08 recovery constant is consumed, and mutating it changes execution [AUTH: 01 §17].

Same contract Block B and Block C carry: a constant that nothing reads is decoration, and a
constant a source default can shadow is worse. Each key gets a probe that resolves it through
the real accessor and reaches the real consumer; the test rewrites the key on a copied config
tree and requires the probe to move or to fail closed.

One key is deliberately unprobeable: `naive_unrescaled_assumed_alpha` is
REQUIRED_NOT_CALIBRATED, because 00 §25 names the unrescaled comparator without freezing any
coefficient-estimation rule. It must stay unreadable, which is asserted rather than probed.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import numpy as np
import pytest
from s08_fixtures import (
    SURFACE,
    linear_family,
    merge_context,
    o3_family,
    protected_and_partners,
    recoverable_bank,
)

from src.materials import (
    PROVISIONAL_FIXTURE_ONLY,
    REQUIRED_NOT_CALIBRATED,
    UncalibratedConstantError,
    material,
    material_status,
    uncalibrated_keys,
)
from src.provenance.config import ConfigError
from src.recovery.context import (
    FIXTURE_PROFILE,
    SCIENTIFIC_PROFILE,
    RecoveryContextError,
    fixture_recovery_context,
    load_recovery_execution_context,
    resolve_recovery_context,
)
from src.recovery.settings import coefficient_schedule, descendant_counts, recovery_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
RELATIVE = "recovery/methods.json"

#: Keys that carry no material value. Documentation only [AUTH: 01 §17].
INERT_CONFIG_FIELDS: tuple[str, ...] = ("authority", "description", "reference_method")


def _copy_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "configs", root / "configs", dirs_exist_ok=True)
    return root


def _mutate(root: Path, key: str, value: object) -> None:
    path = root / "configs" / RELATIVE
    document = json.loads(path.read_text(encoding="utf-8"))
    document[key]["value"] = value
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ structure
def test_every_key_declares_a_calibration_status() -> None:
    document = json.loads((REPO_ROOT / "configs" / RELATIVE).read_text(encoding="utf-8"))
    for key, entry in document.items():
        if key in ("authority", "description"):
            continue
        assert isinstance(entry, dict) and "status" in entry, key


def test_a_key_with_a_value_but_no_status_is_refused(tmp_path: Path) -> None:
    root = _copy_repo(tmp_path)
    path = root / "configs" / RELATIVE
    document = json.loads(path.read_text(encoding="utf-8"))
    document["c2_n_iters"] = 1000
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="calibration status"):
        material(recovery_settings(root), "c2_n_iters")


# ------------------------------------------------------------------ fail-closed states
def test_the_only_uncalibrated_key_is_the_unrescaled_comparator_alpha() -> None:
    """00 §25 names the comparator; no spec freezes a coefficient-estimation rule."""
    settings = recovery_settings(REPO_ROOT)
    assert uncalibrated_keys(settings) == ("naive_unrescaled_assumed_alpha",)
    assert material_status(settings, "naive_unrescaled_assumed_alpha") == REQUIRED_NOT_CALIBRATED
    with pytest.raises(UncalibratedConstantError, match="REQUIRED_NOT_CALIBRATED"):
        material(settings, "naive_unrescaled_assumed_alpha")
    with pytest.raises(UncalibratedConstantError):
        material(settings, "naive_unrescaled_assumed_alpha", allow_provisional=True)


def test_the_fixture_iteration_and_rank_values_need_a_waiver() -> None:
    settings = recovery_settings(REPO_ROOT)
    for key in ("fixture_n_iters", "fixture_residual_rank"):
        assert material_status(settings, key) == PROVISIONAL_FIXTURE_ONLY
        with pytest.raises(UncalibratedConstantError, match="fixture code only"):
            material(settings, key)
        assert material(settings, key, allow_provisional=True)


def test_the_production_context_never_reads_a_fixture_value() -> None:
    context = resolve_recovery_context(REPO_ROOT)
    settings = recovery_settings(REPO_ROOT)
    assert context.c2_n_iters == material(settings, "c2_n_iters") == 1000
    assert context.linear_residual_rank == material(settings, "linear_residual_rank") == 32
    assert context.c2_n_iters != material(settings, "fixture_n_iters", allow_provisional=True)
    assert context.linear_residual_rank != material(
        settings, "fixture_residual_rank", allow_provisional=True
    )
    assert context.profile == SCIENTIFIC_PROFILE and context.is_scientific


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("c2_n_iters", 7),
        ("linear_residual_rank", 5),
        ("conditioning_numerical_rank_tolerance", 0.25),
        ("arithmetic_dtype", "float64"),
        ("c1_pass_criterion_ef", 0.5),
        ("partner_norm_match_limit", 4.0),
        ("c2_rank_scheduler", "ADAPTIVE_RANK_SCHEDULE"),
    ],
)
def test_a_scientific_context_cannot_move_a_pinned_value(
    key: str, value: object, tmp_path: Path
) -> None:
    """R6: `s08.recovery.v1` IS its frozen values, not merely a version string."""
    root = _copy_repo(tmp_path)
    _mutate(root, key, value)
    with pytest.raises(RecoveryContextError):
        resolve_recovery_context(root)


def test_a_fixture_context_is_labelled_and_never_scientific() -> None:
    """R6: the planted-truth escape hatch is unmistakable, not a second scientific regime."""
    fixture = fixture_recovery_context(
        recovery_settings(REPO_ROOT).document, c2_n_iters=3, linear_residual_rank=2
    )
    assert fixture.profile == FIXTURE_PROFILE and not fixture.is_scientific
    assert fixture.identity() != resolve_recovery_context(REPO_ROOT).identity()


def test_a_switching_mapping_cannot_split_the_settings_from_the_hash() -> None:
    """R5: one deep snapshot, so values and config_sha256 describe the same document."""
    base = dict(recovery_settings(REPO_ROOT).document)

    class Switching(Mapping[str, object]):
        """Repeated reads of c2_n_iters alternate between the frozen value and 7."""

        def __init__(self) -> None:
            self.reads = 0

        def __iter__(self) -> Iterator[str]:
            return iter(base)

        def __len__(self) -> int:
            return len(base)

        def __getitem__(self, key: str) -> object:
            if key == "c2_n_iters":
                self.reads += 1
                value = 1000 if self.reads % 2 else 7
                return {"status": "FROZEN", "value": value, "note": "switching"}
            return base[key]

    switching = Switching()
    context = load_recovery_execution_context(switching)  # type: ignore[arg-type]
    assert switching.reads == 1, "the document must be read exactly once, into one snapshot"
    assert context.c2_n_iters == 1000

    #: The hash describes the document the settings came from: rebuilding that exact static
    #: document reproduces it, so no read can have landed on the other value.
    static = json.loads(json.dumps(base))
    static["c2_n_iters"] = {"status": "FROZEN", "value": 1000, "note": "switching"}
    assert context.config_sha256 == load_recovery_execution_context(static).config_sha256


# ------------------------------------------------------------------ consumption probes
def _context_probe(root: Path) -> object:
    context = resolve_recovery_context(root)
    return (
        context.recovery_version,
        context.arithmetic_dtype,
        context.diagnostic_dtype,
        context.svd_workspace_dtype,
        context.svd_backend,
        context.c2_method,
        context.c2_method_version,
        context.c2_rank_scheduler,
        context.conditioning_rank_tolerance,
        context.identity(),
    )


def _c2_iteration_probe(root: Path) -> object:
    """The frozen sweep count is what the scientific context carries into every solve."""
    context = resolve_recovery_context(root)
    return (context.c2_n_iters, context.identity())


def _residual_rank_probe(root: Path) -> object:
    """The configured residual rank reaches the C2 solve through the observation regime."""
    from src.recovery.observations import observe_incomplete_fixed
    from src.recovery.solvers import recover_c2_fixed_alpha

    context = resolve_recovery_context(root)
    protected, partners = recoverable_bank(k=4)
    observation = observe_incomplete_fixed(
        family=linear_family(protected, partners, [0.5] * 4), context=context
    )
    result = recover_c2_fixed_alpha(observation)
    return (result.residual_rank, result.iterations, result.content_identity())


def _conditioning_probe(root: Path) -> object:
    from src.recovery.conditioning import conditioning_report

    context = resolve_recovery_context(root)
    report = conditioning_report(np.diag([1.0, 1e-6 * 1.5, 0.0]), context=context)
    return (report["effective_numerical_rank"], report["kappa"])


def _c1_criterion_probe(root: Path) -> object:
    """The frozen C1 criterion travels on the context; no evaluator argument can move it."""
    from src.recovery.evaluation import c1_pass
    from src.recovery.observations import observe_known_partner
    from src.recovery.solvers import recover_c1
    from src.recovery.truth import bind_truth

    context = resolve_recovery_context(root)
    protected, partners = protected_and_partners(k=1)
    descendant = linear_family(protected, partners, [0.5]).by_id("C1")
    observation = observe_known_partner(
        descendant=descendant, partner_update=partners[0], context=context
    )
    result = recover_c1(observation)
    binding = bind_truth(observation=observation, source=descendant, protected_truth=protected)
    return (context.c1_pass_criterion_ef, c1_pass(result, binding))


def _partner_norm_probe(root: Path) -> object:
    """A 1.3x norm spread passes at the frozen 1.25 limit only if the limit is not read."""
    from src.merge.updates import fixture_induced_update
    from src.recovery.evaluation import partner_norm_guard

    context = resolve_recovery_context(root)
    _, partners = protected_and_partners(k=2)
    scaled = fixture_induced_update(
        {name: np.asarray(partners[0][name], dtype=np.float64) * 1.3 for name in SURFACE},
        context=merge_context(),
        origin="B_scaled",
    )
    return (
        context.partner_norm_match_limit,
        partner_norm_guard([partners[0], scaled], context=context),
    )


def _schedule_probe(root: Path) -> object:
    settings = recovery_settings(root)
    return tuple(
        coefficient_schedule(settings, key)
        for key in (
            "c2b_k2_reference_schedule",
            "c2b_k4_spread_matched_schedule",
            "c2b_k4_wide_schedule",
        )
    )


def _descendant_count_probe(root: Path) -> object:
    return descendant_counts(recovery_settings(root))


def _authorization_probe(root: Path) -> object:
    from src.recovery.solvers import recovery_status

    return tuple(sorted(recovery_status(resolve_recovery_context(root)).items()))


def _o3_probe(root: Path) -> object:
    """The O3 authorisation reaches both the reported status and a truncated-source recovery."""
    from src.recovery.observations import observe_incomplete_fixed
    from src.recovery.solvers import recover_c2_fixed_alpha, recovery_status

    context = resolve_recovery_context(root)
    protected, partners = recoverable_bank(k=4)
    observation = observe_incomplete_fixed(
        family=o3_family(protected, partners, [0.5] * 4, retained_rank=4), context=context
    )
    result = recover_c2_fixed_alpha(observation)
    return (
        recovery_status(context)["O3_RECOVERY_AUTHORIZATION"],
        result.target_class,
        result.retained_rank,
    )


def _profile_probe(root: Path) -> object:
    """A scientific context is pinned; only the fixture factory may hold other values."""
    context = resolve_recovery_context(root)
    return (context.profile, context.is_scientific, context.identity())


def _fixture_probe(root: Path) -> object:
    settings = recovery_settings(root)
    return (
        int(str(material(settings, "fixture_n_iters", allow_provisional=True))),
        int(str(material(settings, "fixture_residual_rank", allow_provisional=True))),
    )


#: key -> (probe, mutated value). Every non-inert calibrated key appears exactly once.
PROBES: dict[str, tuple[Callable[[Path], object], object]] = {
    "arithmetic_dtype": (_context_probe, "float64"),
    "diagnostic_dtype": (_context_probe, "float32"),
    "svd_workspace_dtype": (_context_probe, "float32"),
    "svd_backend": (_context_probe, "TORCH_LINALG_SVD"),
    "c2_method": (_context_probe, "SOMETHING_ELSE"),
    "c2_method_version": (_context_probe, "s08.c2-core.v2"),
    "c2_rank_scheduler": (_context_probe, "ADAPTIVE_RANK_SCHEDULE"),
    "c2_n_iters": (_c2_iteration_probe, 3),
    "linear_residual_rank": (_residual_rank_probe, 5),
    "conditioning_numerical_rank_tolerance": (_conditioning_probe, 1e-3),
    "c1_pass_criterion_ef": (_c1_criterion_probe, 1e-30),
    "partner_norm_match_limit": (_partner_norm_probe, 1.05),
    "c2b_k2_reference_schedule": (_schedule_probe, [0.2, 0.8]),
    "c2b_k4_spread_matched_schedule": (_schedule_probe, [0.3, 0.4, 0.6, 0.7]),
    "c2b_k4_wide_schedule": (_schedule_probe, [0.1, 0.3, 0.7, 0.9]),
    "descendant_counts": (_descendant_count_probe, [2, 3, 4]),
    "o3_recovery_authorization": (_o3_probe, "NOT_AUTHORIZED"),
    "recovery_version": (_profile_probe, "s08.recovery.v2"),
    "dare_incomplete_recovery_status": (_authorization_probe, "AUTHORIZED"),
    "hidden_alpha_recovery_status": (_authorization_probe, "AUTHORIZED"),
    "fixture_n_iters": (_fixture_probe, 11),
    "fixture_residual_rank": (_fixture_probe, 7),
}


def test_every_configured_key_has_a_probe_or_is_declared_uncalibrated() -> None:
    covered = set(PROBES) | set(INERT_CONFIG_FIELDS)
    document = json.loads((REPO_ROOT / "configs" / RELATIVE).read_text(encoding="utf-8"))
    for key, entry in document.items():
        if isinstance(entry, dict) and entry.get("status") == REQUIRED_NOT_CALIBRATED:
            continue
        assert key in covered, f"{RELATIVE}:{key} is consumed by nothing"


def test_the_inert_list_holds_only_documentation_keys() -> None:
    """`reference_method` is traceability prose, not a value any solver branches on."""
    document = json.loads((REPO_ROOT / "configs" / RELATIVE).read_text(encoding="utf-8"))
    entry = document["reference_method"]
    assert isinstance(entry["value"], str) and "Spectral" in entry["value"]
    source = (REPO_ROOT / "src" / "recovery").rglob("*.py")
    for path in source:
        assert "reference_method" not in path.read_text(encoding="utf-8"), path.name


@pytest.mark.parametrize("key", sorted(PROBES))
def test_mutating_a_material_constant_changes_execution(key: str, tmp_path: Path) -> None:
    probe, mutated = PROBES[key]
    baseline = probe(REPO_ROOT)
    root = _copy_repo(tmp_path)
    _mutate(root, key, mutated)
    try:
        after = probe(root)
    except (ConfigError, RecoveryContextError, ValueError) as exc:
        assert str(exc)  # failing closed is a valid consumption
        return
    assert after != baseline, f"{RELATIVE}:{key} does not reach execution"
