"""S01 deterministic configuration resolution [AUTH: 01 §17, §18]."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.provenance.config import (
    ConfigError,
    ConfigSchema,
    deep_merge,
    load_config_document,
    material_values,
    resolve_and_persist,
    resolve_config,
    resolved_config_sha256,
    write_resolved_config,
)
from src.provenance.hashing import read_canonical_json

SCHEMA = ConfigSchema(
    name="test.v1", required=frozenset({"alpha", "seed"}), optional=frozenset({"note"})
)


def _write(path: Path, document: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------------ contract 1, 2
def test_same_logical_config_has_one_identity(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.json", {"alpha": 0.5, "seed": 101, "nested": {"x": 1, "y": 2}})
    b = _write(tmp_path / "b.json", {"nested": {"y": 2, "x": 1}, "seed": 101, "alpha": 0.5})
    left = resolve_config(a, config_root=tmp_path)
    right = resolve_config(b, config_root=tmp_path)
    assert left == right
    assert resolved_config_sha256(left) == resolved_config_sha256(right)


def test_layering_through_extends_is_order_independent_in_the_hash(tmp_path: Path) -> None:
    _write(tmp_path / "base.json", {"alpha": 0.1, "seed": 1, "nested": {"keep": True}})
    child = _write(tmp_path / "child.json", {"extends": "base.json", "alpha": 0.9})
    resolved = resolve_config(child, config_root=tmp_path)
    assert resolved == {"alpha": 0.9, "seed": 1, "nested": {"keep": True}}
    # 'extends' is structural: an inlined equivalent has the same identity.
    inlined = _write(tmp_path / "inlined.json", {"alpha": 0.9, "seed": 1, "nested": {"keep": True}})
    assert resolved_config_sha256(resolved) == resolved_config_sha256(
        resolve_config(inlined, config_root=tmp_path)
    )


def test_extends_list_applies_in_order_and_the_document_wins(tmp_path: Path) -> None:
    _write(tmp_path / "one.json", {"alpha": 1, "seed": 1, "only_one": True})
    _write(tmp_path / "two.json", {"alpha": 2, "seed": 2})
    child = _write(tmp_path / "c.json", {"extends": ["one.json", "two.json"], "seed": 99})
    assert resolve_config(child, config_root=tmp_path) == {"alpha": 2, "seed": 99, "only_one": True}


# ------------------------------------------------------------------ contract 3
def test_material_change_changes_the_hash(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.json", {"alpha": 0.5, "seed": 101})
    b = _write(tmp_path / "b.json", {"alpha": 0.5, "seed": 102})
    assert resolved_config_sha256(resolve_config(a)) != resolved_config_sha256(resolve_config(b))


def test_a_change_in_an_extended_layer_changes_the_hash(tmp_path: Path) -> None:
    base = _write(tmp_path / "base.json", {"alpha": 0.1, "seed": 1})
    child = _write(tmp_path / "child.json", {"extends": "base.json"})
    before = resolved_config_sha256(resolve_config(child, config_root=tmp_path))
    _write(base, {"alpha": 0.2, "seed": 1})
    after = resolved_config_sha256(resolve_config(child, config_root=tmp_path))
    assert before != after


# ------------------------------------------------------------------ contract 4
def test_resolved_config_round_trips(tmp_path: Path) -> None:
    source = _write(tmp_path / "src.json", {"alpha": 0.5, "seed": 101, "note": "n"})
    destination = tmp_path / "resolved" / "config.json"
    resolved, digest = resolve_and_persist(source, destination, config_root=tmp_path, schema=SCHEMA)
    assert read_canonical_json(destination) == resolved
    assert digest == resolved_config_sha256(resolved)
    assert write_resolved_config(tmp_path / "again.json", resolved) == digest


# ------------------------------------------------------------------ contract 5
def test_missing_required_key_fails_closed(tmp_path: Path) -> None:
    source = _write(tmp_path / "c.json", {"alpha": 0.5})
    with pytest.raises(ConfigError, match="missing seed"):
        resolve_config(source, schema=SCHEMA)


def test_unknown_key_fails_closed(tmp_path: Path) -> None:
    source = _write(tmp_path / "c.json", {"alpha": 0.5, "seed": 1, "typo_rank": 32})
    with pytest.raises(ConfigError, match="unknown key"):
        resolve_config(source, schema=SCHEMA)


def test_null_required_value_fails_closed(tmp_path: Path) -> None:
    source = _write(tmp_path / "c.json", {"alpha": None, "seed": 1})
    with pytest.raises(ConfigError, match="requires a value"):
        resolve_config(source, schema=SCHEMA)


def test_malformed_json_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(ConfigError):
        resolve_config(path)


def test_non_object_document_fails_closed(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.json", [1, 2, 3])
    with pytest.raises(ConfigError, match="JSON object"):
        load_config_document(path)


def test_absent_config_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        resolve_config(tmp_path / "nope.json")


def test_extends_cycle_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", {"extends": "b.json"})
    _write(tmp_path / "b.json", {"extends": "a.json"})
    with pytest.raises(ConfigError, match="cycle"):
        resolve_config(tmp_path / "a.json", config_root=tmp_path)


def test_extends_cannot_escape_the_config_root(tmp_path: Path) -> None:
    root = tmp_path / "configs"
    _write(root / "c.json", {"extends": "../outside.json"})
    _write(tmp_path / "outside.json", {"alpha": 1})
    with pytest.raises(ConfigError, match="escapes"):
        resolve_config(root / "c.json", config_root=root)


def test_extends_of_the_wrong_type_fails_closed(tmp_path: Path) -> None:
    source = _write(tmp_path / "c.json", {"extends": {"not": "allowed"}})
    with pytest.raises(ConfigError, match="must be a string"):
        resolve_config(source, config_root=tmp_path)


# ------------------------------------------------------------------ merge semantics
def test_deep_merge_replaces_lists_and_merges_mappings() -> None:
    merged = deep_merge({"a": {"x": 1, "y": 2}, "l": [1, 2, 3]}, {"a": {"y": 9}, "l": [7]})
    assert merged == {"a": {"x": 1, "y": 9}, "l": [7]}


def test_material_values_fails_closed_on_a_missing_setting() -> None:
    assert material_values({"lora_rank": 32}, ["lora_rank"]) == {"lora_rank": 32}
    with pytest.raises(ConfigError, match="material setting"):
        material_values({"lora_rank": 32}, ["merge_alphas"])


# ------------------------------------------------------------------ the real repo config
def test_the_committed_fixture_config_resolves(repo_root: Path) -> None:
    from src.models.fixtures import FIXTURE_SCHEMA

    resolved = resolve_config(
        repo_root / "configs" / "models" / "tiny_fixture.json",
        config_root=repo_root / "configs",
        schema=FIXTURE_SCHEMA,
    )
    assert resolved["lora_rank"] == 32
    assert "extends" not in resolved
    assert resolved["role"] == "TEST_FIXTURE_NOT_A_RESEARCH_SUBJECT"
