"""C27, C31, C32 and R8/R9/R15-R17/R20 — the compatibility contract and the real CLI.

The command is exercised for real: `main()` is invoked with the arguments an operator would
pass, against a deterministic local root. It is NOT run against a research model — none has
been acquired, so no compatibility claim exists and none is made [AUTH: 03 §8; 02 §C6].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from backend_fixtures import (
    FAMILY_ALIAS,
    FAMILY_LAYOUTS,
    FIXTURE_MODEL_ID,
    build_cli_fixture_root,
    fixture_panel_entry,
    lora_factors,
    read_fixture_adapter,
    trunk_selection,
    write_checkpoint_snapshot,
    write_fixture_adapter,
)

from src.backend.adapters import resolve_lora_specification
from src.backend.compatibility import (
    CONTRACT_CHECKS,
    ContractWeights,
    check_checkpoint_quantization,
    check_dp_requirement,
    check_exact_inversion,
    check_h100_profile,
    check_scorer_determinism,
)
from src.backend.dp import DPBackendError, build_dp_plan, dp_role_for
from src.backend.evidence import (
    BACKEND_NOT_INSTALLED,
    CHECKPOINT_NOT_ACQUIRED,
    FAIL,
    NO_H100,
    NOT_RUN,
    NOT_RUN_REASONS,
    PASS,
    RunBinding,
    structural_problems,
    verify_contract_evidence,
)
from src.backend.settings import backend_settings
from src.dp.mechanism import ACCOUNTANT_VERSION, DPMechanism, account
from src.materials import material, material_text

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from model_compatibility_contract import main, run_contract_for  # noqa: E402

SELECTION = list(trunk_selection("tiny_fixture"))
MODULES = tuple(name.removesuffix(".weight") for name in SELECTION)


def _forbidden_load_options() -> tuple[str, ...]:
    value = material(backend_settings(REPO_ROOT), "forbidden_load_options")
    assert isinstance(value, list)
    return tuple(str(key) for key in value)


FORBIDDEN = _forbidden_load_options()


def _weights(tmp_path: Path) -> ContractWeights:
    """A complete weight-dependent bundle from the tiny local fixture."""
    spec = resolve_lora_specification(
        REPO_ROOT,
        architecture_family="tiny_fixture",
        target_modules=("gate_proj", "up_proj", "down_proj"),
    )
    factors = lora_factors(MODULES, rank=spec.rank, seed=13)
    saved = write_fixture_adapter(
        tmp_path / "peft", specification=spec, factors=factors, expected_selection=SELECTION
    )
    reloaded = read_fixture_adapter(
        tmp_path / "peft", specification=spec, expected_selection=SELECTION
    )
    extracted = reloaded.induced_updates()

    alpha = 0.5
    rng = np.random.default_rng(17)
    partner = {name: rng.normal(size=value.shape) for name, value in extracted.items()}
    merged = {n: alpha * extracted[n] + (1.0 - alpha) * partner[n] for n in extracted}

    def score() -> list[float]:
        """Execute the fixture scoring path; called twice by the determinism check."""
        from backend_fixtures import DeterministicLogitsSource, token_batches, tokenizer_identity

        from src.backend.scoring_backend import HuggingFaceScoringBackend
        from src.scoring.reference import negative_log_likelihood

        records = ("rec-1", "rec-2", "rec-3")
        backend = HuggingFaceScoringBackend(
            root=REPO_ROOT,
            model_revision="f" * 40,
            tokenizer=tokenizer_identity(),
            precision="float32",
            source=DeterministicLogitsSource(),
            batches=token_batches(records),
        )
        return [
            negative_log_likelihood(backend.token_evidence(artifact_id="art-1", record_id=record))
            for record in records
        ]

    return ContractWeights(
        factors=reloaded.factors,
        scaling=reloaded.specification.scaling,
        extracted=extracted,
        reloaded_update_sha256=reloaded.update_identity(),
        saved_update_sha256=saved.update_identity(),
        merged=merged,
        score=score,
    )


def _fixture_outcome(tmp_path: Path, **overrides: object) -> object:
    snapshot, digest = write_checkpoint_snapshot(tmp_path / "snapshot")
    arguments: dict[str, object] = {
        "parameter_names": FAMILY_LAYOUTS["tiny_fixture"],
        "weights": _weights(tmp_path),
        "entry": fixture_panel_entry(config_sha256=digest),
        "checkpoint_config": snapshot,
    }
    arguments.update(overrides)
    return run_contract_for(REPO_ROOT, "tiny_fixture", **arguments)  # type: ignore[arg-type]


# ==================================================================== C27 measured state
def test_c27_every_cpu_decidable_check_passes_against_the_local_fixture(
    tmp_path: Path,
) -> None:
    """Ten of the thirteen checks are decidable from the fixture alone, and all ten pass.

    The other three are stated so they hold on both lanes. `model_loads` and
    `bf16_text_only_forward` are NOT_RUN either because transformers is absent (CPU/dev) or
    because a fixture checkpoint is never acquired (H100 image) — the contract downloads
    nothing to close a row. `h100_profile_collectable` is NOT_RUN(NO_H100) without an
    accelerator and PASSes on a real H100. Either way a NOT_RUN row can never contribute to a
    PASS, so the contract is NOT_RUN against a fixture on every lane [AUTH: 02 §C6].
    """
    outcome = _fixture_outcome(tmp_path)
    assert [r.name for r in outcome.results] == list(CONTRACT_CHECKS)  # type: ignore[attr-defined]
    by_name = {r.name: r for r in outcome.results}  # type: ignore[attr-defined]

    unavailable = ("model_loads", "bf16_text_only_forward", "h100_profile_collectable")
    decidable = [name for name in CONTRACT_CHECKS if name not in unavailable]
    assert len(decidable) == 10
    failing = [(n, by_name[n].detail) for n in decidable if by_name[n].status != PASS]
    assert not failing, failing

    for name in ("model_loads", "bf16_text_only_forward"):
        assert by_name[name].status == NOT_RUN, "a fixture is never loaded, on any lane"
        assert by_name[name].reason in {BACKEND_NOT_INSTALLED, CHECKPOINT_NOT_ACQUIRED}

    profile = by_name["h100_profile_collectable"]
    assert profile.status in {NOT_RUN, PASS}
    if profile.status == NOT_RUN:
        assert profile.reason == NO_H100

    assert outcome.status == NOT_RUN  # type: ignore[attr-defined]
    assert outcome.exit_code == 1  # type: ignore[attr-defined]
    assert outcome.model_id == FIXTURE_MODEL_ID  # type: ignore[attr-defined]


def test_the_weight_dependent_checks_really_ran(tmp_path: Path) -> None:
    outcome = _fixture_outcome(tmp_path)
    by_name = {r.name: r for r in outcome.results}  # type: ignore[attr-defined]
    for name in (
        "adapter_save_load_stable",
        "induced_update_ba_exact",
        "linear_merge_constructs",
        "p0_c1_exact_inversion",
        "scorer_deterministic",
    ):
        assert by_name[name].status == PASS, (name, by_name[name].detail)
    assert by_name["induced_update_ba_exact"].measurement["scaling"] == 2.0
    assert float(str(by_name["p0_c1_exact_inversion"].measurement["e_f"])) <= 1e-5


# ==================================================================== R15 accepted C1 path
def test_r15_the_inversion_check_runs_the_accepted_s08_c1_path(tmp_path: Path) -> None:
    """Not a hand-inverted formula: a genuine S07 family, S08 observation and evaluator."""
    import inspect

    from src.backend.compatibility import _run_c1

    source = inspect.getsource(check_exact_inversion) + inspect.getsource(_run_c1)
    assert "recover_c1" in source and "bind_truth" in source
    assert "parameter_recovery_metrics" in source
    assert "build_release_family" in source and "observe_known_partner" in source
    assert "(merged[n] - (1.0" not in source, "no local re-derivation of the inversion"

    weights = _weights(tmp_path)
    result = check_exact_inversion(REPO_ROOT, weights.extracted)
    assert result.status == PASS
    assert "recovery_result_sha256" in result.measurement
    assert "truth_binding_sha256" in result.measurement
    assert float(str(result.measurement["e_f"])) <= 1e-5


def test_r15_a_corrupted_surface_fails_the_inversion_check() -> None:
    zeros = {"w": np.zeros((4, 3))}
    assert check_exact_inversion(REPO_ROOT, zeros).status == FAIL
    assert check_exact_inversion(REPO_ROOT, {}).status == FAIL


# ==================================================================== R16 two executions
def test_r16_the_determinism_check_executes_the_scorer_twice(tmp_path: Path) -> None:
    calls: list[int] = []

    def score() -> list[float]:
        calls.append(1)
        return [0.25, -1.5, 3.0]

    result = check_scorer_determinism(score, tolerance=0.0)
    assert result.status == PASS
    assert len(calls) == 2, "a copy of one result is tautological"
    assert result.measurement["n_scores"] == 3


def test_r16_a_non_deterministic_scorer_fails() -> None:
    values = iter([[1.0, 2.0], [1.0, 2.5]])

    def score() -> list[float]:
        return next(values)

    assert check_scorer_determinism(score, tolerance=0.0).status == FAIL


def test_r16_the_fixture_scorer_is_genuinely_reproducible(tmp_path: Path) -> None:
    weights = _weights(tmp_path)
    first, second = weights.score(), weights.score()
    assert first == second and len(first) == 3
    assert all(np.isfinite(v) for v in first)


# ==================================================================== R17 H100
def test_r17_a_fixture_accelerator_cannot_pass_the_h100_row() -> None:
    result = check_h100_profile(
        {"device_name": "fixture-accelerator", "total_memory_bytes": 1 << 30}
    )
    assert result.status == NOT_RUN and result.reason == NO_H100
    assert check_h100_profile(None).status == NOT_RUN
    assert check_h100_profile({"device_name": "x"}).status == FAIL


def test_r17_only_a_real_h100_profile_closes_the_row() -> None:
    result = check_h100_profile(
        {"device_name": "NVIDIA H100 80GB HBM3", "total_memory_bytes": 85_899_345_920}
    )
    assert result.status == PASS


# ==================================================================== R8/R9 checkpoint config
def test_r9a_an_acquired_unquantized_snapshot_passes(tmp_path: Path) -> None:
    path, digest = write_checkpoint_snapshot(tmp_path / "snap")
    result = check_checkpoint_quantization(
        config_path=path, declared_config_sha256=digest, forbidden=FORBIDDEN
    )
    assert result.status == PASS
    assert result.measurement["config_sha256"] == digest


def test_r9b_an_acquired_quantized_snapshot_fails(tmp_path: Path) -> None:
    path, digest = write_checkpoint_snapshot(tmp_path / "snap", quantized=True)
    result = check_checkpoint_quantization(
        config_path=path, declared_config_sha256=digest, forbidden=FORBIDDEN
    )
    assert result.status == FAIL
    assert "quantization_config" in result.detail


def test_r9c_an_absent_snapshot_is_not_run_not_pass() -> None:
    result = check_checkpoint_quantization(
        config_path=None, declared_config_sha256="a" * 64, forbidden=FORBIDDEN
    )
    assert result.status == NOT_RUN and result.reason == CHECKPOINT_NOT_ACQUIRED


def test_r9d_a_snapshot_whose_hash_differs_from_the_manifest_fails(tmp_path: Path) -> None:
    path, _ = write_checkpoint_snapshot(tmp_path / "snap")
    result = check_checkpoint_quantization(
        config_path=path, declared_config_sha256="b" * 64, forbidden=FORBIDDEN
    )
    assert result.status == FAIL
    assert "not the checkpoint that was frozen" in result.detail


def test_r8_the_unacquired_panel_reports_checkpoint_not_acquired() -> None:
    """The panel manifest is our own declaration; it cannot certify the checkpoint."""
    for alias in FAMILY_ALIAS.values():
        outcome = run_contract_for(REPO_ROOT, alias, parameter_names=None)
        row = {r.name: r for r in outcome.results}["no_quantization_required"]
        assert row.status == NOT_RUN
        assert row.reason == CHECKPOINT_NOT_ACQUIRED


def test_r8_the_request_policy_and_the_checkpoint_property_are_separate() -> None:
    """The loader refuses a caller asking for 8-bit; this asks what the checkpoint requires."""
    from src.backend.loader import quantization_refusals

    assert quantization_refusals({"load_in_8bit": True}, forbidden=FORBIDDEN)
    assert not quantization_refusals({}, forbidden=FORBIDDEN)


# ==================================================================== C32 nothing claimed
def test_c32_no_real_model_or_h100_state_is_claimed() -> None:
    settings = backend_settings(REPO_ROOT)
    assert material(settings, "backend_integrated") is False
    assert material_text(settings, "suite_scope") == "STATISTICAL_STACK_ONLY"
    readiness = json.loads(
        (REPO_ROOT / "artifacts" / "p0_pre" / "P0_PRE_READINESS.json").read_text(encoding="utf-8")
    )
    assert readiness.get("p0_pre_ready") in (False, None)


def test_the_real_panel_never_reaches_pass() -> None:
    for alias in FAMILY_ALIAS.values():
        outcome = run_contract_for(REPO_ROOT, alias, parameter_names=None)
        assert outcome.status != PASS and outcome.exit_code == 1
        by_name = {r.name: r for r in outcome.results}
        assert by_name["exact_revision_supplied"].status == FAIL


def test_the_mapping_checks_still_run_when_parameter_names_are_supplied() -> None:
    for family, alias in FAMILY_ALIAS.items():
        outcome = run_contract_for(REPO_ROOT, alias, parameter_names=FAMILY_LAYOUTS[family])
        by_name = {r.name: r for r in outcome.results}
        assert by_name["multimodal_components_frozen"].status == PASS
        assert by_name["common_mlp_lora_rank_32_attaches"].status == PASS


# ==================================================================== R20 the real CLI
def test_r20_case1_an_unresolved_revision_writes_nothing_and_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """MAJOR-1: a clean structured NOT_RUN, not a traceback and not a file."""
    target = tmp_path / "evidence.json"
    code = main(
        [
            "--root",
            str(REPO_ROOT),
            "--alias",
            "llama_3_2_3b",
            "--evidence",
            str(target),
            "--run-id",
            "a" * 64,
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert not target.exists(), "no file may claim to be evidence for an unacquired model"
    assert "MODEL_REVISION_NOT_FROZEN" in captured.err
    assert "NOT_WRITTEN" in captured.err
    assert "Traceback" not in captured.err + captured.out


def test_r20_case2_the_real_cli_writes_verifiable_fixture_evidence(tmp_path: Path) -> None:
    """The actual command entry point, against a deterministic local root."""
    snapshot, digest = write_checkpoint_snapshot(tmp_path / "snapshot")
    root = build_cli_fixture_root(tmp_path / "root", config_sha256=digest)
    run_id = "c" * 64

    code = main(
        [
            "--root",
            str(root),
            "--alias",
            "tiny_fixture",
            "--checkpoint-config",
            str(snapshot),
            "--run-id",
            run_id,
            "--evidence",
            "evidence.json",
        ]
    )
    assert code == 1, "a fixture checkpoint is never acquired, so the contract is NOT_RUN"

    written = root / "evidence.json"
    assert written.is_file()
    document = json.loads(written.read_text(encoding="utf-8"))
    assert document["status"] == NOT_RUN
    assert document["run_id"] == run_id
    assert document["attempt_id"] and document["attempt_id"] != run_id
    assert document["git_commit"] != "0" * 40
    assert structural_problems(document) == []

    reasons = {row["reason"] for row in document["checks"] if row["status"] == NOT_RUN}
    # Which reason closes the load rows is a property of the lane, not of the evidence
    # format: transformers is absent here and present on the H100 image, where a fixture is
    # still never fetched. Both are registered NOT_RUN reasons, and neither is a PASS.
    assert reasons & {BACKEND_NOT_INSTALLED, CHECKPOINT_NOT_ACQUIRED}
    assert reasons <= set(NOT_RUN_REASONS)
    assert all(row["reason"] for row in document["checks"] if row["status"] == NOT_RUN)

    binding = RunBinding(**{k: document[k] for k in _BINDING_FIELDS})
    assert verify_contract_evidence(document, binding=binding, root=root) == []


_BINDING_FIELDS = (
    "run_id",
    "git_commit",
    "environment_lock_sha256",
    "model_manifest_sha256",
    "model_revision",
    "backend_code_sha256",
    "scoring_code_sha256",
    "resolved_config_sha256",
)


@pytest.mark.parametrize(
    "field",
    [
        "model_revision",
        "environment_lock_sha256",
        "backend_code_sha256",
        "scoring_code_sha256",
        "run_id",
        "model_manifest_sha256",
    ],
)
def test_r20_tampering_any_binding_makes_verification_fail(tmp_path: Path, field: str) -> None:
    snapshot, digest = write_checkpoint_snapshot(tmp_path / "snapshot")
    root = build_cli_fixture_root(tmp_path / "root", config_sha256=digest)
    main(
        [
            "--root",
            str(root),
            "--alias",
            "tiny_fixture",
            "--checkpoint-config",
            str(snapshot),
            "--run-id",
            "c" * 64,
            "--evidence",
            "evidence.json",
        ]
    )
    document = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    binding = RunBinding(**{k: document[k] for k in _BINDING_FIELDS})
    forged = "d" * (40 if field == "model_revision" else 64)
    problems = verify_contract_evidence({**document, field: forged}, binding=binding, root=root)
    assert any(field in p for p in problems), (field, problems)


def test_the_fixture_cli_record_agrees_with_its_own_exit_code(tmp_path: Path) -> None:
    """The CPU fixture is NOT_RUN with exit 1, and the verifier accepts that pairing."""
    snapshot, digest = write_checkpoint_snapshot(tmp_path / "snapshot")
    root = build_cli_fixture_root(tmp_path / "root", config_sha256=digest)
    code = main(
        [
            "--root",
            str(root),
            "--alias",
            "tiny_fixture",
            "--checkpoint-config",
            str(snapshot),
            "--run-id",
            "c" * 64,
            "--evidence",
            "evidence.json",
        ]
    )
    document = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    assert (document["status"], document["exit_code"], code) == (NOT_RUN, 1, 1)
    binding = RunBinding(**{k: document[k] for k in _BINDING_FIELDS})
    assert verify_contract_evidence(document, binding=binding, root=root) == []

    #: flipping only the exit code makes the record claim a command that succeeded
    forged = {**document, "exit_code": 0}
    problems = verify_contract_evidence(forged, binding=binding, root=root)
    assert any("disagree" in p for p in problems), problems


def test_a_fail_record_with_a_zero_exit_code_is_rejected() -> None:
    """The same rule on the FAIL side: a failing contract did not exit cleanly."""
    outcome = run_contract_for(REPO_ROOT, "llama_3_2_3b", parameter_names=None)
    assert outcome.status == FAIL and outcome.exit_code == 1

    from src.backend.evidence import build_contract_evidence

    binding = RunBinding(
        run_id="e" * 64,
        git_commit="f" * 40,
        environment_lock_sha256="1" * 64,
        model_manifest_sha256="2" * 64,
        model_revision="3" * 40,
        backend_code_sha256="4" * 64,
        scoring_code_sha256="5" * 64,
        resolved_config_sha256="6" * 64,
        attempt_id="7" * 64,
    )
    document = build_contract_evidence(
        binding=binding,
        status=FAIL,
        checks=[result.as_dict() for result in outcome.results],
        artifact_hashes={},
        started_utc="2026-08-30T10:00:00Z",
        ended_utc="2026-08-30T10:01:00Z",
        exit_code=1,
    )
    assert structural_problems(document) == []
    assert any("disagree" in p for p in structural_problems({**document, "exit_code": 0}))


def test_r20_tampering_a_check_status_makes_verification_fail(tmp_path: Path) -> None:
    snapshot, digest = write_checkpoint_snapshot(tmp_path / "snapshot")
    root = build_cli_fixture_root(tmp_path / "root", config_sha256=digest)
    main(
        [
            "--root",
            str(root),
            "--alias",
            "tiny_fixture",
            "--checkpoint-config",
            str(snapshot),
            "--run-id",
            "c" * 64,
            "--evidence",
            "evidence.json",
        ]
    )
    document = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    forged = {**document, "status": PASS}
    assert any("aggregate to" in p for p in structural_problems(forged))


def test_r20_evidence_needs_a_supplied_run_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot, digest = write_checkpoint_snapshot(tmp_path / "snapshot")
    root = build_cli_fixture_root(tmp_path / "root", config_sha256=digest)
    code = main(
        [
            "--root",
            str(root),
            "--alias",
            "tiny_fixture",
            "--checkpoint-config",
            str(snapshot),
            "--evidence",
            "evidence.json",
        ]
    )
    assert code == 1
    assert "--run-id is required" in capsys.readouterr().err
    assert not (root / "evidence.json").exists()


# ==================================================================== C31 DP hook
def _mechanism() -> DPMechanism:
    from src.materials import material_number
    from src.training.settings import dp_settings

    settings = dp_settings(REPO_ROOT)
    return DPMechanism(
        adjacency=material_text(settings, "adjacency"),
        delta=material_number(settings, "fixture_delta", allow_provisional=True),
        clipping_norm=material_number(settings, "fixture_clipping_norm", allow_provisional=True),
        noise_multiplier=material_number(
            settings, "fixture_noise_multiplier", allow_provisional=True
        ),
        sample_rate=material_number(settings, "fixture_sample_rate", allow_provisional=True),
        steps=int(material_number(settings, "fixture_steps", allow_provisional=True)),
        dp_seed=int(material_number(settings, "smoke_seed")),
        requested_epsilon=material_number(settings, "target_epsilon"),
    )


def test_c31_the_dp_hook_consumes_the_frozen_s06_contract() -> None:
    mechanism = _mechanism()
    plan = build_dp_plan(REPO_ROOT, mechanism, model_alias="llama_3_2_3b")
    assert plan.adjacency == mechanism.adjacency
    assert plan.max_grad_norm == mechanism.clipping_norm
    assert plan.noise_multiplier == mechanism.noise_multiplier
    assert plan.sample_rate == mechanism.sample_rate
    assert plan.steps == mechanism.steps
    assert plan.dp_seed == mechanism.dp_seed
    assert plan.accountant_version == ACCOUNTANT_VERSION


def test_the_dp_epsilon_comes_from_the_accepted_accountant_only() -> None:
    from src.backend.dp import accounting_for

    mechanism = _mechanism()
    assert accounting_for(mechanism).achieved_epsilon == account(mechanism).achieved_epsilon
    for path in sorted((REPO_ROOT / "src" / "backend").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "RDPAccountant" not in source and "get_epsilon" not in source, path.name


def test_the_dp_primary_is_llama_and_the_secondary_is_conditional() -> None:
    settings = backend_settings(REPO_ROOT)
    assert material_text(settings, "dp_primary_alias") == "llama_3_2_3b"
    assert material_text(settings, "dp_secondary_alias") == "qwen3_5_4b_base"
    assert dp_role_for(REPO_ROOT, "llama_3_2_3b") == "DP_PRIMARY"
    assert dp_role_for(REPO_ROOT, "qwen3_5_4b_base") == "DP_SECONDARY"
    assert dp_role_for(REPO_ROOT, "gemma_4_e4b") is None


def test_a_model_without_a_dp_role_cannot_be_given_one() -> None:
    with pytest.raises(DPBackendError, match="carries no DP role"):
        build_dp_plan(REPO_ROOT, _mechanism(), model_alias="gemma_4_e4b")


def test_a_non_sample_level_adjacency_cannot_reach_the_backend_at_all() -> None:
    from dataclasses import replace

    from src.dp.mechanism import DPError

    with pytest.raises(DPError, match="sample-level relation"):
        replace(_mechanism(), adjacency="USER_LEVEL")

    import inspect

    source = inspect.getsource(build_dp_plan)
    assert "UNSUPPORTED_ADJACENCY" in source and "SAMPLE_LEVEL_ADJACENCY" in source


def test_the_dp_check_is_not_run_rather_than_pass_without_an_executed_dp_step() -> None:
    """A DP-role row is NOT_RUN on both lanes, and never PASS.

    The distinction the DP contract rests on is "not executed" versus "passed", and it does
    not depend on whether opacus imports: this lane has no opacus, and the H100 image has
    opacus but still runs no DP step during the contract. Asserting the absence of opacus
    would test the machine rather than the contract [AUTH: 02 §C6].
    """
    plan = build_dp_plan(REPO_ROOT, _mechanism(), model_alias="llama_3_2_3b")
    result = check_dp_requirement("DP_PRIMARY", plan)
    assert result.status == NOT_RUN and result.status != PASS
    assert result.reason == "OPACUS_BACKEND_NOT_INSTALLED"

    # The rest of the row's contract, which no lane changes: a DP role with no plan is a
    # FAIL rather than a quiet NOT_RUN, and only a model with no DP role passes outright.
    assert check_dp_requirement("DP_PRIMARY", None).status == FAIL
    assert check_dp_requirement(None, None).status == PASS
