"""C25, C26, C28-C30 — token alignment, the Min-K path, and unforgeable evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from backend_fixtures import DeterministicLogitsSource, token_batches, tokenizer_identity

from src.backend.evidence import (
    FAIL,
    NO_H100,
    NOT_RUN,
    NOT_RUN_REASONS,
    PASS,
    EvidenceError,
    RunBinding,
    build_contract_evidence,
    structural_problems,
    verify_contract_evidence,
    write_contract_evidence,
)
from src.backend.forward import (
    IGNORE_INDEX,
    ForwardError,
    log_softmax,
    token_evidence_from_logits,
)
from src.backend.scoring_backend import TokenBatch
from src.provenance.hashing import JSONValue
from src.scoring.reference import min_k_percent_score, negative_log_likelihood

REPO_ROOT = Path(__file__).resolve().parents[2]


# ==================================================================== C25 reference loss
def test_c25_the_causal_shift_aligns_position_t_with_token_t_plus_one() -> None:
    rng = np.random.default_rng(19)
    logits = rng.normal(size=(7, 13))
    labels = np.array([1, 4, 9, 2, 7, 3, 5])
    evidence = token_evidence_from_logits(logits, labels)

    reference = log_softmax(logits[:-1])
    expected = np.array([reference[t, labels[t + 1]] for t in range(6)])
    assert np.allclose(evidence.log_probs, expected)
    assert evidence.token_count == 6, "the unshiftable first position is excluded"

    unshifted = np.array([reference[t, labels[t]] for t in range(6)])
    assert not np.allclose(evidence.log_probs, unshifted), (
        "an unshifted read scores each token against itself and would inflate every likelihood"
    )


def test_padding_and_ignored_labels_never_enter_a_reduction() -> None:
    rng = np.random.default_rng(23)
    logits = rng.normal(size=(6, 11))
    labels = np.array([1, 5, 6, 7, 0, 0])
    attention = np.array([1, 1, 1, 1, 0, 0])
    evidence = token_evidence_from_logits(logits, labels, attention_mask=attention)
    assert evidence.valid.tolist() == [True, True, True, False, False]
    assert evidence.token_count == 3

    ignored = np.array([1, 5, IGNORE_INDEX, 7, 8, 9])
    masked = token_evidence_from_logits(logits, ignored)
    assert masked.valid.tolist() == [True, False, True, True, True]


def test_a_prompt_mask_stops_prompt_likelihood_leaking_into_a_continuation() -> None:
    rng = np.random.default_rng(29)
    logits = rng.normal(size=(6, 11))
    labels = np.array([1, 2, 3, 4, 5, 6])
    prompt = np.array([True, True, True, False, False, False])
    evidence = token_evidence_from_logits(logits, labels, prompt_mask=prompt)
    assert evidence.valid.tolist() == [False, False, True, True, True]
    assert evidence.token_count == 3


def test_the_reduction_runs_in_float64_regardless_of_the_forward_precision() -> None:
    rng = np.random.default_rng(31)
    logits = rng.normal(size=(5, 9)).astype(np.float32)
    evidence = token_evidence_from_logits(logits, np.arange(5))
    assert evidence.log_probs.dtype == np.float64
    assert np.isclose(float(np.sum(np.exp(log_softmax(logits)), axis=-1)[0]), 1.0)


@pytest.mark.parametrize(
    ("logits", "labels", "match"),
    [
        (np.zeros((1, 4)), np.zeros(1), "shorter than two tokens"),
        (np.zeros((3, 4)), np.zeros(2), "positions"),
        (np.full((3, 4), np.nan), np.zeros(3), "non-finite"),
        (np.zeros((3, 4)), np.array([0, 99, 1]), "outside the vocabulary"),
    ],
)
def test_a_malformed_forward_is_refused(logits: object, labels: object, match: str) -> None:
    with pytest.raises(ForwardError, match=match):
        token_evidence_from_logits(np.asarray(logits), np.asarray(labels, dtype=np.int64))


# ==================================================================== C26 Min-K path
def test_c26_the_backend_supplies_token_values_and_the_accepted_scorer_reduces_them() -> None:
    """00 §34A.2: there is one Min-K implementation and the backend is not it."""
    rng = np.random.default_rng(37)
    logits = rng.normal(size=(10, 15))
    evidence = token_evidence_from_logits(logits, np.arange(10) % 15)

    fraction = 0.4
    values = evidence.valid_log_probs()
    lowest = np.sort(values)[: max(1, int(np.ceil(fraction * values.size)))]
    assert min_k_percent_score(evidence, fraction) == pytest.approx(float(np.mean(lowest)))
    assert negative_log_likelihood(evidence) == pytest.approx(
        -float(np.mean(evidence.valid_log_probs()))
    )

    for path in sorted((REPO_ROOT / "src" / "backend").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "def min_k" not in source, path.name
        assert "def reference_calibrated_score" not in source, path.name
        assert "def negative_log_likelihood" not in source, path.name


def test_the_backend_never_returns_a_score() -> None:
    """The protocol returns token evidence; a scalar score would be a second scorer."""
    import inspect

    from src.backend.scoring_backend import HuggingFaceScoringBackend

    signature = inspect.signature(HuggingFaceScoringBackend.token_evidence)
    assert signature.return_annotation == "TokenEvidence"
    assert not any(
        name.startswith(("score_", "reduce_"))
        for name in dir(HuggingFaceScoringBackend)
        if not name.startswith("_")
    )


def test_the_deterministic_logits_source_is_reproducible() -> None:
    source = DeterministicLogitsSource()
    batch = token_batches(["rec-1"])["rec-1"]
    first = source.logits_for(artifact_id="A", batch=batch)
    second = source.logits_for(artifact_id="A", batch=batch)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, source.logits_for(artifact_id="B", batch=batch))


def test_a_batch_whose_masks_disagree_is_refused() -> None:
    from src.backend.scoring_backend import ScoringBackendError

    with pytest.raises(ScoringBackendError, match="disagree"):
        TokenBatch(record_id="r", input_ids=(1, 2, 3), attention_mask=(1, 1))


# ==================================================================== C28-C30 evidence
def binding(**overrides: object) -> RunBinding:
    values: dict[str, str] = {
        "run_id": "1" * 64,
        "git_commit": "2" * 40,
        "environment_lock_sha256": "3" * 64,
        "model_manifest_sha256": "4" * 64,
        "model_revision": "5" * 40,
        "backend_code_sha256": "6" * 64,
        "scoring_code_sha256": "7" * 64,
        "resolved_config_sha256": "8" * 64,
    }
    values.update({k: str(v) for k, v in overrides.items()})
    return RunBinding(**values)


def evidence(**overrides: JSONValue) -> dict[str, JSONValue]:
    document = build_contract_evidence(
        binding=binding(),
        status=PASS,
        checks=[
            {"name": "exact_revision_supplied", "status": PASS, "detail": "pinned", "reason": ""}
        ],
        artifact_hashes={},
        started_utc="2026-08-30T10:00:00Z",
        ended_utc="2026-08-30T10:05:00Z",
        exit_code=0,
    )
    document.update(overrides)
    return document


def test_c28_a_hand_authored_pass_file_is_rejected() -> None:
    """`{"status": "PASS"}` binds nothing and verifies against nothing."""
    forged: dict[str, JSONValue] = {"status": PASS}
    assert structural_problems(forged)
    assert verify_contract_evidence(forged, binding=binding(), root=REPO_ROOT)


def test_a_well_formed_record_verifies_against_its_own_run() -> None:
    assert verify_contract_evidence(evidence(), binding=binding(), root=REPO_ROOT) == []


def test_c29_a_mismatched_environment_hash_is_rejected() -> None:
    problems = verify_contract_evidence(
        evidence(), binding=binding(environment_lock_sha256="9" * 64), root=REPO_ROOT
    )
    assert any("environment_lock_sha256" in p for p in problems)


def test_c30_a_mismatched_model_manifest_hash_is_rejected() -> None:
    problems = verify_contract_evidence(
        evidence(), binding=binding(model_manifest_sha256="a" * 64), root=REPO_ROOT
    )
    assert any("model_manifest_sha256" in p for p in problems)


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "git_commit",
        "model_revision",
        "backend_code_sha256",
        "scoring_code_sha256",
        "resolved_config_sha256",
    ],
)
def test_every_binding_is_verified_not_merely_recorded(field: str) -> None:
    filler = "b" * (40 if field in {"git_commit", "model_revision"} else 64)
    problems = verify_contract_evidence(
        evidence(), binding=binding(**{field: filler}), root=REPO_ROOT
    )
    assert any(field in p for p in problems), field


#: One check row per overall status, so the aggregation agrees with the claimed status and the
#: exit-code rule is what the case is actually testing.
_ROWS: dict[str, list[dict[str, JSONValue]]] = {
    PASS: [{"name": "a", "status": PASS, "detail": "", "reason": ""}],
    NOT_RUN: [{"name": "a", "status": NOT_RUN, "detail": "", "reason": NO_H100}],
    FAIL: [{"name": "a", "status": FAIL, "detail": "", "reason": ""}],
}


@pytest.mark.parametrize(
    ("status", "exit_code", "valid"),
    [
        (PASS, 0, True),
        (PASS, 1, False),
        (NOT_RUN, 0, False),
        (NOT_RUN, 1, True),
        (FAIL, 0, False),
        (FAIL, 1, True),
    ],
)
def test_the_exit_code_and_the_status_must_agree_both_ways(
    status: str, exit_code: int, valid: bool
) -> None:
    """(exit_code == 0) == (status == PASS). Checking one direction is not enough.

    "PASS implies exit 0" alone leaves NOT_RUN and FAIL free to carry exit 0, which reads to
    every downstream consumer as a command that succeeded [AUTH: 01 §15, §16; 02 §C6].
    """
    document = {**evidence(), "status": status, "checks": _ROWS[status], "exit_code": exit_code}
    problems = [p for p in structural_problems(document) if "disagree" in p]
    assert (not problems) == valid, (status, exit_code, problems)


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [(PASS, 1), (NOT_RUN, 0), (FAIL, 0)],
)
def test_a_disagreeing_exit_code_cannot_be_built(status: str, exit_code: int) -> None:
    with pytest.raises(EvidenceError, match="disagree"):
        build_contract_evidence(
            binding=binding(),
            status=status,
            checks=_ROWS[status],
            artifact_hashes={},
            started_utc="2026-08-30T10:00:00Z",
            ended_utc="2026-08-30T10:05:00Z",
            exit_code=exit_code,
        )


@pytest.mark.parametrize(("status", "exit_code"), [(PASS, 0), (NOT_RUN, 1), (FAIL, 1)])
def test_an_agreeing_exit_code_builds_and_verifies(status: str, exit_code: int) -> None:
    document = build_contract_evidence(
        binding=binding(),
        status=status,
        checks=_ROWS[status],
        artifact_hashes={},
        started_utc="2026-08-30T10:00:00Z",
        ended_utc="2026-08-30T10:05:00Z",
        exit_code=exit_code,
    )
    assert structural_problems(document) == []
    assert verify_contract_evidence(document, binding=binding(), root=REPO_ROOT) == []


def test_a_pass_whose_checks_did_not_all_pass_is_refused() -> None:
    with pytest.raises(EvidenceError, match="aggregate to NOT_RUN"):
        build_contract_evidence(
            binding=binding(),
            status=PASS,
            checks=[
                {"name": "a", "status": PASS, "detail": "", "reason": ""},
                {"name": "b", "status": NOT_RUN, "detail": "", "reason": NO_H100},
            ],
            artifact_hashes={},
            started_utc="2026-08-30T10:00:00Z",
            ended_utc="2026-08-30T10:05:00Z",
            exit_code=0,
        )


# ==================================================================== R11 aggregation
@pytest.mark.parametrize(
    ("claimed", "rows", "match"),
    [
        (
            NOT_RUN,
            [{"name": "a", "status": FAIL, "detail": "", "reason": ""}],
            "aggregate to FAIL",
        ),
        (
            PASS,
            [{"name": "a", "status": FAIL, "detail": "", "reason": ""}],
            "aggregate to FAIL",
        ),
        (
            PASS,
            [{"name": "a", "status": NOT_RUN, "detail": "", "reason": NO_H100}],
            "aggregate to NOT_RUN",
        ),
        (
            FAIL,
            [{"name": "a", "status": PASS, "detail": "", "reason": ""}],
            "aggregate to PASS",
        ),
    ],
)
def test_r11_the_overall_status_must_equal_the_frozen_aggregation(
    claimed: str, rows: list[dict[str, JSONValue]], match: str
) -> None:
    """Any FAIL -> FAIL; else any NOT_RUN -> NOT_RUN; else PASS. Enforced independently."""
    document = {**evidence(), "status": claimed, "checks": rows, "exit_code": 1}
    problems = structural_problems(document)
    assert any(match in p for p in problems), problems


def test_r11_the_aggregation_helper_is_the_one_rule() -> None:
    from src.backend.evidence import aggregate_status

    def rows(*statuses: str) -> list[dict[str, JSONValue]]:
        return [{"name": f"c{i}", "status": s} for i, s in enumerate(statuses)]

    assert aggregate_status(rows(PASS, PASS)) == PASS
    assert aggregate_status(rows(PASS, NOT_RUN)) == NOT_RUN
    assert aggregate_status(rows(PASS, FAIL)) == FAIL
    assert aggregate_status(rows(NOT_RUN, FAIL)) == FAIL
    with pytest.raises(EvidenceError, match="not PASS, FAIL or NOT_RUN"):
        aggregate_status(rows("MAYBE"))


# ==================================================================== R14 specific reasons
def test_r14_a_not_run_row_without_a_declared_reason_is_refused() -> None:
    document = {
        **evidence(),
        "status": NOT_RUN,
        "checks": [{"name": "a", "status": NOT_RUN, "detail": "", "reason": ""}],
        "exit_code": 1,
    }
    problems = structural_problems(document)
    assert any("NOT_RUN without a declared reason" in p for p in problems)


@pytest.mark.parametrize("reason", list(NOT_RUN_REASONS))
def test_r14_each_declared_reason_is_accepted_and_serialized(reason: str) -> None:
    document = build_contract_evidence(
        binding=binding(),
        status=NOT_RUN,
        checks=[{"name": "a", "status": NOT_RUN, "detail": "x", "reason": reason}],
        artifact_hashes={},
        started_utc="2026-08-30T10:00:00Z",
        ended_utc="2026-08-30T10:05:00Z",
        exit_code=1,
    )
    checks = document["checks"]
    assert isinstance(checks, list)
    row = checks[0]
    assert isinstance(row, dict)
    assert row["reason"] == reason


def test_r14_an_unknown_reason_is_refused() -> None:
    document = {
        **evidence(),
        "status": NOT_RUN,
        "checks": [{"name": "a", "status": NOT_RUN, "detail": "", "reason": "BECAUSE"}],
        "exit_code": 1,
    }
    assert any("declared reason" in p for p in structural_problems(document))


# ==================================================================== R12/R13 provenance
def test_r12_the_all_zero_commit_is_refused_as_a_placeholder() -> None:
    problems = structural_problems({**evidence(), "git_commit": "0" * 40})
    assert any("all-zero sha" in p for p in problems)


def test_r13_the_record_carries_a_supplied_run_id_and_a_separate_attempt_id() -> None:
    document = evidence()
    assert document["run_id"] == "1" * 64
    assert "attempt_id" in document
    assert document["attempt_id"] != document["run_id"]


def test_r13_no_backend_module_derives_a_run_id() -> None:
    """01 §15/§16 run identity belongs to the S09 run-manifest layer."""
    import ast

    for path in sorted((REPO_ROOT / "src" / "backend").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                assert "run_id" not in node.name, (path.name, node.name)
    source = (REPO_ROOT / "scripts" / "model_compatibility_contract.py").read_text(encoding="utf-8")
    assert "run_id=sha256_canonical" not in source
    assert "attempt_id = sha256_canonical" in source


def test_a_record_that_ends_before_it_starts_is_refused() -> None:
    with pytest.raises(EvidenceError, match="ends before it starts"):
        build_contract_evidence(
            binding=binding(),
            status=FAIL,
            checks=[{"name": "a", "status": FAIL, "detail": ""}],
            artifact_hashes={},
            started_utc="2026-08-30T10:05:00Z",
            ended_utc="2026-08-30T10:00:00Z",
            exit_code=1,
        )


def test_a_floating_model_revision_cannot_appear_in_evidence() -> None:
    problems = structural_problems({**evidence(), "model_revision": "main"})
    assert any("immutable 40-hex revision" in p for p in problems)


def test_an_artifact_hash_is_recomputed_from_the_file(tmp_path: Path) -> None:
    from src.provenance.hashing import sha256_file

    artifact = tmp_path / "out.json"
    artifact.write_text('{"a": 1}\n', encoding="utf-8")
    document = build_contract_evidence(
        binding=binding(),
        status=PASS,
        checks=[{"name": "a", "status": PASS, "detail": ""}],
        artifact_hashes={"out.json": sha256_file(artifact)},
        started_utc="2026-08-30T10:00:00Z",
        ended_utc="2026-08-30T10:05:00Z",
        exit_code=0,
    )
    assert verify_contract_evidence(document, binding=binding(), root=tmp_path) == []

    artifact.write_text('{"a": 2}\n', encoding="utf-8")
    problems = verify_contract_evidence(document, binding=binding(), root=tmp_path)
    assert any("changed after the record was written" in p for p in problems)


def test_a_malformed_record_cannot_be_written(tmp_path: Path) -> None:
    malformed: dict[str, JSONValue] = {"status": PASS}
    with pytest.raises(EvidenceError):
        write_contract_evidence(tmp_path, "evidence.json", malformed)
    assert not (tmp_path / "evidence.json").exists()


def test_the_tokenizer_identity_is_content_addressed() -> None:
    first = tokenizer_identity()
    assert first.identity() == tokenizer_identity().identity()
    assert first.identity() != tokenizer_identity(revision="2" * 40).identity()
    assert first.identity() != tokenizer_identity(max_sequence_length=256).identity()
    assert first.identity() != tokenizer_identity(config_sha256="e" * 64).identity()
    assert first.identity() != tokenizer_identity(special_tokens={"pad_token_id": 9}).identity()
    document = json.loads(json.dumps(first.as_dict()))
    assert document["truncation_policy"] and document["padding_policy"]
    assert document["loss_mask_policy"] and document["causal_shift"]
