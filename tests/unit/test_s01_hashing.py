"""S01 canonicalisation and artifact hashing [AUTH: 01 §13-§16, §19]."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.provenance.hashing import (
    ArtifactRecord,
    ArtifactVerificationError,
    CanonicalisationError,
    JSONValue,
    canonical_json_bytes,
    read_canonical_json,
    record_artifact,
    sha256_bytes,
    sha256_canonical,
    sha256_file,
    verify_artifact,
    verify_artifacts,
    write_canonical_json,
)


def test_key_order_does_not_change_the_identity() -> None:
    left: JSONValue = {"b": 1, "a": {"z": [1, 2], "y": "x"}}
    right: JSONValue = {"a": {"y": "x", "z": [1, 2]}, "b": 1}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_canonical(left) == sha256_canonical(right)


def test_material_change_changes_the_identity() -> None:
    assert sha256_canonical({"seed": 101}) != sha256_canonical({"seed": 102})
    # 1 and 1.0 are different JSON values and must not collide.
    assert sha256_canonical({"x": 1}) != sha256_canonical({"x": 1.0})
    # list order is material
    assert sha256_canonical({"x": [1, 2]}) != sha256_canonical({"x": [2, 1]})


def test_written_bytes_are_exactly_the_hashed_bytes(tmp_path: Path) -> None:
    document: JSONValue = {"beta": [1, 2, 3], "alpha": "value"}
    path = tmp_path / "nested" / "doc.json"
    digest = write_canonical_json(path, document)
    assert digest == sha256_canonical(document)
    assert digest == sha256_file(path)
    assert path.read_bytes() == canonical_json_bytes(document)
    assert path.read_bytes().endswith(b"\n")
    assert read_canonical_json(path) == document


def test_non_finite_floats_fail_closed() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(CanonicalisationError):
            canonical_json_bytes({"x": value})


def test_non_string_keys_fail_closed() -> None:
    with pytest.raises(CanonicalisationError):
        canonical_json_bytes({1: "one"})  # type: ignore[dict-item]


def test_non_json_types_fail_closed() -> None:
    with pytest.raises(CanonicalisationError):
        canonical_json_bytes({"x": {1, 2}})  # type: ignore[dict-item]


def test_unreadable_json_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CanonicalisationError):
        read_canonical_json(path)


# ------------------------------------------------------------------ artifacts 18-20
def test_stable_artifact_verifies(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "table.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    record = record_artifact(tmp_path, "results/table.json")
    assert record.sha256 == sha256_bytes((tmp_path / "results/table.json").read_bytes())
    verify_artifact(tmp_path, record)
    assert verify_artifacts(tmp_path, [record]) == []


def test_modified_artifact_fails_verification(tmp_path: Path) -> None:
    target = tmp_path / "table.json"
    target.write_text("original", encoding="utf-8")
    record = record_artifact(tmp_path, "table.json")
    target.write_text("mutated", encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="hash mismatch"):
        verify_artifact(tmp_path, record)
    assert len(verify_artifacts(tmp_path, [record])) == 1


def test_missing_artifact_fails_verification(tmp_path: Path) -> None:
    target = tmp_path / "table.json"
    target.write_text("original", encoding="utf-8")
    record = record_artifact(tmp_path, "table.json")
    target.unlink()
    with pytest.raises(ArtifactVerificationError, match="absent"):
        verify_artifact(tmp_path, record)


def test_recording_an_absent_artifact_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ArtifactVerificationError, match="absent"):
        record_artifact(tmp_path, "never/written.json")


def test_artifact_paths_cannot_escape_the_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"x")
    for candidate in ("../outside.bin", str(outside)):
        with pytest.raises(ArtifactVerificationError):
            record_artifact(tmp_path, candidate)


def test_artifact_record_serialises_for_a_manifest(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"payload")
    record = record_artifact(tmp_path, "a.bin")
    assert record.as_dict() == {
        "path": "a.bin",
        "sha256": record.sha256,
        "size_bytes": 7,
    }
    assert isinstance(ArtifactRecord("a.bin", record.sha256, 7), ArtifactRecord)
