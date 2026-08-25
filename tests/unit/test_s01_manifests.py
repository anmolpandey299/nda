"""S01 model and data provenance manifest contracts [AUTH: 01 §13, §14, §8G; 00 §33]."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from check_repo_invariants import RUN_MANIFEST_REQUIRED_FIELDS as CHECKER_FIELDS

from src.provenance.data_manifest import (
    DataManifestError,
    DataManifestExistsError,
    data_manifest_sha256,
    partition_sha256,
    read_data_manifest,
    stamp_manifest_sha256,
    validate_data_manifest,
    write_data_manifest,
)
from src.provenance.model_manifest import (
    FIXTURE_ROLE,
    ManifestExistsError,
    ModelManifestError,
    is_research_subject,
    read_model_manifest,
    validate_model_manifest,
    write_model_manifest,
)
from src.provenance.run_manifest import RUN_MANIFEST_REQUIRED_FIELDS


def _model(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "model_id": "local/tiny-fixture",
        "revision": "a" * 40,
        "config_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "license_snapshot_sha256": "3" * 64,
        "download_timestamp_utc": "2026-08-25T09:00:00Z",
        "parameter_count": 82240,
        "dtype_on_disk": "float32",
        "role": FIXTURE_ROLE,
    }
    document.update(overrides)
    return document


def _data(**overrides: Any) -> dict[str, Any]:
    splits = {"calibration": ["nat-0000", "nat-0001"], "evaluation": ["nat-0002"]}
    document: dict[str, Any] = {
        "source_query": "fixture corpus",
        "source_version": "v1",
        "download_timestamp_utc": "2026-08-25T09:00:00Z",
        "raw_data_sha256": "4" * 64,
        "preprocessing_code_commit": "b" * 40,
        "preprocessing_config_sha256": "5" * 64,
        "split_rng_seed": 101,
        "split_ids": splits,
        "deduplication_method": "exact-sha256",
        "deduplication_version": "1",
        "final_split_sha256": {k: partition_sha256(v) for k, v in splits.items()},
    }
    document.update(overrides)
    return document


# ------------------------------------------------------------------ run manifest vocabulary
def test_run_manifest_fields_match_the_invariant_checker() -> None:
    assert RUN_MANIFEST_REQUIRED_FIELDS == CHECKER_FIELDS


# ------------------------------------------------------------------ model manifest
def test_a_complete_model_manifest_validates() -> None:
    assert validate_model_manifest(_model()) == []


@pytest.mark.parametrize("field", sorted(_model()))
def test_every_required_model_field_is_required(field: str) -> None:
    document = _model()
    document.pop(field)
    assert any("missing required field" in p for p in validate_model_manifest(document))


def test_a_floating_revision_is_refused() -> None:
    for revision in ("main", "MASTER", "head"):
        problems = validate_model_manifest(_model(revision=revision))
        assert any("floating branch" in p for p in problems)


def test_malformed_evidentiary_values_are_refused() -> None:
    assert any("SHA256" in p for p in validate_model_manifest(_model(config_sha256="short")))
    assert any(
        "timestamp" in p
        for p in validate_model_manifest(_model(download_timestamp_utc="25 August 2026"))
    )
    assert any("parameter_count" in p for p in validate_model_manifest(_model(parameter_count=0)))
    assert any("empty" in p for p in validate_model_manifest(_model(dtype_on_disk="  ")))
    assert any("unknown field" in p for p in validate_model_manifest(_model(extra_field=1)))


def _research(**overrides: Any) -> dict[str, Any]:
    document = _model(
        role="RESEARCH_SUBJECT",
        revision="a" * 40,
        architecture_family="llama",
        base_or_instruct="base",
        weight_file_sha256="6" * 64,
        lora_target_mapping={"mlp": ["gate_proj", "up_proj", "down_proj"]},
        frozen_modules=["embed_tokens", "lm_head"],
        license="llama-community",
    )
    document.update(overrides)
    return document


# ------------------------------------------------------------------ F4
def test_the_contract_is_derived_from_the_manifest_role() -> None:
    assert is_research_subject(_research()) is True
    assert is_research_subject(_model()) is False


def test_a_caller_cannot_downgrade_a_research_model_to_the_fixture_contract() -> None:
    """The reviewer counterexample: omitting a flag once let a declared research subject be
    validated against the weaker fixture contract [AUTH: 01 §8G, §13]."""
    bare = _model(role="RESEARCH_SUBJECT", revision="a" * 40)
    problems = validate_model_manifest(bare)
    assert any("architecture_family" in p for p in problems), problems
    assert any("weight_file_sha256" in p for p in problems), problems


def test_a_complete_research_manifest_validates() -> None:
    assert validate_model_manifest(_research()) == []


@pytest.mark.parametrize("revision", ["main", "v1.0", "refs/heads/release", "abc123"])
def test_a_non_immutable_research_revision_is_rejected(revision: str) -> None:
    problems = validate_model_manifest(_research(revision=revision))
    assert any("floating branch" in p or "immutable" in p for p in problems), problems


@pytest.mark.parametrize("weight", ["not-a-digest", "6" * 63, "xet:", ""])
def test_a_malformed_weight_identity_is_rejected(weight: str) -> None:
    assert validate_model_manifest(_research(weight_file_sha256=weight)) != []


def test_a_xet_weight_identity_is_accepted() -> None:
    assert validate_model_manifest(_research(weight_file_sha256="xet:abcdef0123456789")) == []


def test_the_fixture_contract_still_works() -> None:
    assert validate_model_manifest(_model()) == []
    assert validate_model_manifest(_model(revision="local-fixture-v1")) == []


def test_model_manifest_round_trips(tmp_path: Path) -> None:
    path, digest = write_model_manifest(tmp_path, "tiny_fixture", _model())
    assert path == tmp_path / "manifests" / "models" / "tiny_fixture.json"
    assert read_model_manifest(path) == _model()
    assert len(digest) == 64


# ------------------------------------------------------------------ F3
def test_rewriting_an_identical_model_manifest_is_idempotent(tmp_path: Path) -> None:
    first, digest = write_model_manifest(tmp_path, "tiny_fixture", _model())
    before = first.read_bytes()
    second, again = write_model_manifest(tmp_path, "tiny_fixture", _model())
    assert (second, again) == (first, digest)
    assert first.read_bytes() == before


def test_a_differing_model_manifest_never_overwrites(tmp_path: Path) -> None:
    path, _ = write_model_manifest(tmp_path, "tiny_fixture", _model())
    before = path.read_bytes()
    with pytest.raises(ManifestExistsError, match="never overwritten"):
        write_model_manifest(tmp_path, "tiny_fixture", _model(parameter_count=999))
    assert path.read_bytes() == before


def test_an_invalid_model_manifest_never_reaches_the_tree(tmp_path: Path) -> None:
    with pytest.raises(ModelManifestError):
        write_model_manifest(tmp_path, "bad", _model(revision="main"))
    assert not (tmp_path / "manifests" / "models" / "bad.json").exists()


def test_a_traversing_alias_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ModelManifestError, match="invalid model alias"):
        write_model_manifest(tmp_path, "../escape", _model())


# ------------------------------------------------------------------ data manifest
def test_a_complete_data_manifest_validates() -> None:
    assert validate_data_manifest(_data()) == []


@pytest.mark.parametrize("field", sorted(_data()))
def test_every_required_data_field_is_required(field: str) -> None:
    document = _data()
    document.pop(field)
    assert validate_data_manifest(document) != []


def test_overlapping_partitions_are_refused() -> None:
    splits = {"calibration": ["r1", "r2"], "evaluation": ["r2"]}
    document = _data(
        split_ids=splits, final_split_sha256={k: partition_sha256(v) for k, v in splits.items()}
    )
    assert any("disjoint" in p for p in validate_data_manifest(document))


def test_a_repeated_record_inside_one_partition_is_refused() -> None:
    splits = {"calibration": ["r1", "r1"]}
    document = _data(
        split_ids=splits, final_split_sha256={k: partition_sha256(v) for k, v in splits.items()}
    )
    assert any("repeats a record id" in p for p in validate_data_manifest(document))


def test_a_declared_split_hash_must_match_its_ids() -> None:
    document = _data(final_split_sha256={"calibration": "0" * 64, "evaluation": "0" * 64})
    assert any("does not match its recorded ids" in p for p in validate_data_manifest(document))


def test_an_undeclared_partition_is_refused() -> None:
    document = _data()
    document["final_split_sha256"]["extra"] = "0" * 64
    assert any("unknown partition" in p for p in validate_data_manifest(document))


def test_partition_identity_is_order_sensitive() -> None:
    assert partition_sha256(["a", "b"]) != partition_sha256(["b", "a"])


def test_manifest_identity_excludes_itself(tmp_path: Path) -> None:
    document = _data()
    identity = data_manifest_sha256(document)
    stamped = stamp_manifest_sha256(document)
    assert stamped["manifest_sha256"] == identity
    assert data_manifest_sha256(stamped) == identity

    path, written = write_data_manifest(tmp_path, "fixture_corpus", document)
    assert written == identity
    assert read_data_manifest(path)["manifest_sha256"] == identity


def test_an_invalid_data_manifest_never_reaches_the_tree(tmp_path: Path) -> None:
    with pytest.raises(DataManifestError):
        write_data_manifest(tmp_path, "bad", _data(split_rng_seed="one"))
    assert not (tmp_path / "manifests" / "data" / "bad.json").exists()


# ------------------------------------------------------------------ F3
def test_rewriting_an_identical_data_manifest_is_idempotent(tmp_path: Path) -> None:
    first, identity = write_data_manifest(tmp_path, "corpus", _data())
    before = first.read_bytes()
    second, again = write_data_manifest(tmp_path, "corpus", _data())
    assert (second, again) == (first, identity)
    assert first.read_bytes() == before


def test_a_differing_split_never_overwrites_a_dataset_alias(tmp_path: Path) -> None:
    """v1 is written, v2 moves a record between partitions, and the alias must not change
    under the results that already cite it [AUTH: 01 §14, §15, §27; 00 §33.3]."""
    path, identity = write_data_manifest(tmp_path, "corpus", _data())
    before = path.read_bytes()

    moved = {"calibration": ["nat-0000"], "evaluation": ["nat-0001", "nat-0002"]}
    version_two = _data(
        split_ids=moved,
        final_split_sha256={k: partition_sha256(v) for k, v in moved.items()},
    )
    with pytest.raises(DataManifestExistsError, match="never overwritten"):
        write_data_manifest(tmp_path, "corpus", version_two)

    assert path.read_bytes() == before
    assert read_data_manifest(path)["manifest_sha256"] == identity


def test_a_stale_stamped_identity_is_rejected() -> None:
    """Mutate the content, keep the stamp, and the manifest still claims the old identity."""
    stamped = stamp_manifest_sha256(_data())
    assert validate_data_manifest(stamped) == []
    stale = {**stamped, "raw_data_sha256": "9" * 64}
    problems = validate_data_manifest(stale)
    assert any("stale" in p for p in problems), problems


def test_a_malformed_stamp_is_rejected() -> None:
    assert validate_data_manifest({**_data(), "manifest_sha256": "not-a-digest"}) != []


def test_split_hashes_must_be_digests() -> None:
    document = _data()
    document["final_split_sha256"] = dict.fromkeys(document["split_ids"], "nope")
    problems = validate_data_manifest(document)
    assert any("not a SHA256 digest" in p for p in problems), problems


# ================================================== IDENTITY_FIELD_TYPE_FAIL_CLOSED
@pytest.mark.parametrize(
    "field",
    [
        "raw_data_sha256",
        "preprocessing_config_sha256",
        "preprocessing_code_commit",
        "download_timestamp_utc",
    ],
)
def test_a_numeric_data_identity_field_is_rejected(field: str) -> None:
    """`isinstance(value, str) and not pattern.match(value)` let an integer skip the check."""
    problems = validate_data_manifest(_data(**{field: 123}))
    assert any("must be a string" in p for p in problems), problems


def test_a_malformed_document_is_not_stampable_into_a_valid_manifest() -> None:
    stamped = stamp_manifest_sha256(_data(raw_data_sha256=123))
    problems = validate_data_manifest(stamped)
    assert any("must be a string" in p for p in problems), problems


def test_a_numeric_data_identity_field_never_reaches_the_tree(tmp_path: Path) -> None:
    with pytest.raises(DataManifestError, match="must be a string"):
        write_data_manifest(tmp_path, "corpus", _data(raw_data_sha256=123))
    assert not (tmp_path / "manifests" / "data" / "corpus.json").exists()


@pytest.mark.parametrize(
    "field",
    [
        "config_sha256",
        "tokenizer_sha256",
        "license_snapshot_sha256",
        "download_timestamp_utc",
        "model_id",
        "dtype_on_disk",
        "revision",
    ],
)
def test_a_numeric_model_identity_field_is_rejected(field: str) -> None:
    problems = validate_model_manifest(_model(**{field: 123}))
    assert any("must be a string" in p for p in problems), problems


def test_a_numeric_research_revision_is_rejected() -> None:
    problems = validate_model_manifest(_research(revision=123))
    assert any("must be a string" in p for p in problems), problems


def test_a_numeric_weight_identity_is_rejected() -> None:
    problems = validate_model_manifest(_research(weight_file_sha256=123))
    assert any("must be a string" in p for p in problems), problems


def test_a_numeric_model_identity_field_never_reaches_the_tree(tmp_path: Path) -> None:
    with pytest.raises(ModelManifestError, match="must be a string"):
        write_model_manifest(tmp_path, "tiny_fixture", _model(config_sha256=123))
    assert not (tmp_path / "manifests" / "models" / "tiny_fixture.json").exists()


def test_a_legitimately_numeric_field_is_still_numeric() -> None:
    """`parameter_count` and `split_rng_seed` must not be forced to strings."""
    assert validate_model_manifest(_model(parameter_count=82240)) == []
    assert validate_data_manifest(_data(split_rng_seed=101)) == []
    assert validate_model_manifest(_model(parameter_count="82240")) != []
    assert validate_data_manifest(_data(split_rng_seed="101")) != []
