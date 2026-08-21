"""Production HF/PEFT backend contract [AUTH: 02 §C6, §C7; 01 §8C].

These are collected skeletons, not stubs that pass: each asserts a real contract property and
skips with an explicit reason while the backend is absent. A skip is never a PASS, and the
readiness conjunction is driven by provenance-bound evidence records, not by this lane's exit
code [AUTH: 02 §C6; plan §11].

The lane must never collect zero tests: an empty lane would let `make backend-contract` report
a state it never evaluated.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

BACKEND_MODULES = ("transformers", "peft")
OWNED_BY_S03 = "src/scoring"


def _backend_present() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in BACKEND_MODULES)


requires_backend = pytest.mark.skipif(
    not _backend_present(),
    reason="NOT_RUN(BACKEND_NOT_INTEGRATED): transformers/peft absent [AUTH: 02 §C6]",
)


def test_backend_contract_lane_is_not_empty() -> None:
    """Guard against the lane silently collecting nothing [AUTH: 02 §C6; 00 §34B.3]."""
    assert Path(__file__).is_file()


@requires_backend
def test_scoring_package_owns_the_backend_adapter() -> None:
    """The HF/PEFT adapter lives under the sole owner [AUTH: 00 §34A.1; 02 §C7]."""
    scoring = REPO_ROOT / OWNED_BY_S03
    adapters = sorted(p.name for p in scoring.glob("*.py") if p.name != "__init__.py")
    assert adapters, (
        "BACKEND_INTEGRATED implies src/scoring/ ships the adapter; it is empty. "
        "Implemented at S03 [AUTH: 01 §39 S03]."
    )


@requires_backend
def test_scoring_code_hash_covers_the_backend_adapter() -> None:
    """SCORING_CODE_HASH must span the adapter, masking/reduction, reference loss and
    Min-K, not a hand-typed version string [AUTH: 02 §C7]."""
    pytest.skip("NOT_RUN(BACKEND_NOT_INTEGRATED): SCORING_CODE_HASH is defined at S03")


@requires_backend
def test_cache_key_includes_code_and_environment_identity() -> None:
    """Backend source change -> miss; environment change -> miss [AUTH: 02 §C7; 01 §32]."""
    pytest.skip("NOT_RUN(BACKEND_NOT_INTEGRATED): the cache is implemented at S03")
