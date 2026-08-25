"""Deterministic configuration resolution — one path, one identity.

Every scientific choice comes from version-controlled config, never from a source literal
[AUTH: 01 §17]. A CLI invocation must be reconstructable from the saved resolved config, so
resolution is deterministic, layered through explicit `extends`, and persisted with a stable
SHA256 before execution begins.

Format is JSON, not YAML: the CPU/dev lane locked in `uv.lock` contains no YAML parser (the
only PyYAML in the lock is a transitive dependency of the Linux-only `science` extra), and
adding one would change `uv_lock_sha256` and therefore the accepted ENVIRONMENT_LOCK_SHA256
[AUTH: 01 §12, §15]. 01 §18 fixes the CLI shape, not the serialisation format.

Resolution rules, all deliberately boring:

* `extends` names one config or an ordered list; later entries override earlier ones, and
  the including document overrides all of them;
* mappings merge recursively, every other type replaces wholesale;
* `extends` is structural and is removed from the resolved document, so two files that mean
  the same thing get the same identity;
* a cycle, an escape out of the config root, a non-object document, a missing required key,
  an unknown key or a null required value all fail closed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.provenance.hashing import (
    CanonicalisationError,
    JSONDocument,
    JSONValue,
    read_canonical_json,
    sha256_canonical,
    write_canonical_json,
)

EXTENDS_KEY: Final = "extends"


class ConfigError(ValueError):
    """A config cannot be resolved, or does not satisfy its declared contract."""


@dataclass(frozen=True)
class ConfigSchema:
    """The contract a resolved config must satisfy. Unknown keys are an error, not noise.

    A silently ignored key is how a run acquires a setting nobody applied: the operator
    believes the value took effect and the resolved config records it, while the code never
    read it [AUTH: 01 §17].
    """

    name: str
    required: frozenset[str]
    optional: frozenset[str] = frozenset()

    def validate(self, document: JSONDocument, *, where: str = "<config>") -> None:
        keys = set(document)
        missing = sorted(self.required - keys)
        if missing:
            raise ConfigError(f"{where}: schema {self.name} is missing {', '.join(missing)}")
        unknown = sorted(keys - self.required - self.optional)
        if unknown:
            raise ConfigError(
                f"{where}: schema {self.name} rejects unknown key(s) {', '.join(unknown)}"
            )
        null_required = sorted(k for k in self.required if document[k] is None)
        if null_required:
            raise ConfigError(
                f"{where}: schema {self.name} requires a value for {', '.join(null_required)}"
            )


def deep_merge(base: JSONDocument, override: JSONDocument) -> dict[str, JSONValue]:
    """Recursive merge for mappings; any other type is replaced, never concatenated.

    Lists replace rather than append so that a config can shorten a list. Appending would
    make it impossible to express "exactly these three seeds" in an overriding layer.
    """
    merged: dict[str, JSONValue] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_config_document(path: Path) -> dict[str, JSONValue]:
    """Read one config file. A config is always a JSON object."""
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    try:
        document = read_canonical_json(path)
    except CanonicalisationError as exc:
        raise ConfigError(str(exc)) from exc
    if not isinstance(document, Mapping):
        raise ConfigError(f"{path}: a config must be a JSON object")
    return dict(document)


def _extends_targets(document: JSONDocument, path: Path) -> list[str]:
    raw = document.get(EXTENDS_KEY)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence) and all(isinstance(item, str) for item in raw):
        return [str(item) for item in raw]
    raise ConfigError(f"{path}: '{EXTENDS_KEY}' must be a string or a list of strings")


def _resolve_within(config_root: Path, base_dir: Path, target: str) -> Path:
    candidate = (base_dir / target).resolve()
    try:
        candidate.relative_to(config_root.resolve())
    except ValueError as exc:
        raise ConfigError(
            f"'{EXTENDS_KEY}' target {target!r} escapes the config root {config_root}"
        ) from exc
    return candidate


def resolve_config(
    path: Path,
    *,
    config_root: Path | None = None,
    schema: ConfigSchema | None = None,
) -> dict[str, JSONValue]:
    """Resolve `path` and every layer it extends into one deterministic document."""
    start = path.resolve()
    root = (config_root or start.parent).resolve()
    resolved = _resolve(start, root, [])
    if schema is not None:
        schema.validate(resolved, where=str(path))
    return resolved


def _resolve(path: Path, config_root: Path, stack: list[Path]) -> dict[str, JSONValue]:
    if path in stack:
        cycle = " -> ".join(p.name for p in [*stack, path])
        raise ConfigError(f"'{EXTENDS_KEY}' cycle: {cycle}")
    document = load_config_document(path)
    merged: dict[str, JSONValue] = {}
    for target in _extends_targets(document, path):
        parent = _resolve_within(config_root, path.parent, target)
        merged = deep_merge(merged, _resolve(parent, config_root, [*stack, path]))
    document.pop(EXTENDS_KEY, None)
    return deep_merge(merged, document)


def resolved_config_sha256(resolved: JSONDocument) -> str:
    """CONFIG_SHA256 — a RUN_ID input [AUTH: 01 §15, §17]."""
    return sha256_canonical(resolved)


def write_resolved_config(path: Path, resolved: JSONDocument) -> str:
    """Persist the resolved config before execution and return its identity."""
    return write_canonical_json(path, resolved)


def resolve_and_persist(
    source: Path,
    destination: Path,
    *,
    config_root: Path | None = None,
    schema: ConfigSchema | None = None,
) -> tuple[dict[str, JSONValue], str]:
    """The whole Part-1 contract in one call: resolve, validate, persist, identify."""
    resolved = resolve_config(source, config_root=config_root, schema=schema)
    digest = write_resolved_config(destination, resolved)
    return resolved, digest


def material_values(resolved: JSONDocument, keys: Iterable[str]) -> dict[str, JSONValue]:
    """Fetch declared material settings, failing closed on any that is absent.

    Call sites read experimental constants through this, so a typo is an error rather than a
    silent fallback to a source default [AUTH: 01 §17].
    """
    out: dict[str, JSONValue] = {}
    for key in keys:
        if key not in resolved:
            raise ConfigError(f"resolved config does not define material setting {key!r}")
        out[key] = resolved[key]
    return out
