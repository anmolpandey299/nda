"""Resolved-config access for the production backend [AUTH: 01 §17].

One path, Block A's. Nothing here re-implements resolution; it names the file and hands back
the resolved document plus its identity, so a backend constant cannot be shadowed by a source
default or a test literal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from src.analysis.settings import ResolvedSettings, load_settings
from src.provenance.hashing import sha256_canonical

BACKEND_CONFIG: Final = "backend/runtime.json"


def backend_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, BACKEND_CONFIG)


def backend_runtime_config_sha256(root: Path) -> str:
    """The identity of the RESOLVED backend runtime document [AUTH: 02 §C7].

    Source bytes are not the whole of executable backend semantics: the attention
    implementation, the forward precision, `trust_remote_code`, the LoRA scaling rule and
    `use_rslora` all change what a forward pass or an induced update computes while leaving
    every `.py` file untouched. So the canonical resolved document is hashed and bound into
    the backend identity and the scoring cache key. Hashing the path name would prove nothing.
    """
    return sha256_canonical(dict(backend_settings(root).document))
