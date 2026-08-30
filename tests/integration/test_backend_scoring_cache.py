"""C19-C24 — one scorer, and a cache identity that covers executable backend semantics.

02 §C7 is binding: the cache key must change whenever what a score *means* changes. A
hand-typed `scorer_version` is explicitly insufficient, so the identity is taken over the
scoring source, the backend source, the resolved config, the tokenizer identity, the model
revision, the precision and the record set. Each of the four negative cases below moves
exactly one of those and requires a miss; the positive case moves none and requires a hit that
returns bitwise-identical scalars.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from backend_fixtures import DeterministicLogitsSource, token_batches, tokenizer_identity

from src.backend.scoring_backend import (
    BACKEND_SOURCE_FILES,
    HuggingFaceScoringBackend,
    ScoringBackendError,
    backend_code_hash,
)
from src.backend.settings import backend_settings
from src.scoring.backends import record_set_identity
from src.scoring.cache import SCORING_SOURCE_FILES, CacheIdentity, ScoreCache, scoring_code_hash
from src.scoring.engine import Arm, ScoreFamily, ScoringContext, View, score_records
from src.scoring.reference import TokenEvidence

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDS = ("mem-1", "mem-2", "non-1", "non-2")


def make_backend(root: Path = REPO_ROOT, **overrides: object) -> HuggingFaceScoringBackend:
    values: dict[str, object] = {
        "root": root,
        "model_revision": "9" * 40,
        "tokenizer": tokenizer_identity(),
        "precision": "bfloat16",
        "source": DeterministicLogitsSource(),
        "batches": token_batches(RECORDS),
    }
    values.update(overrides)
    return HuggingFaceScoringBackend(**values)  # type: ignore[arg-type]


# ==================================================================== C19 one scorer
def test_c19_the_backend_implements_the_accepted_scoring_protocol() -> None:
    """The engine consumes it unchanged: no second scorer, no second reducer."""
    backend = make_backend()
    evidence = backend.token_evidence(artifact_id="art-1", record_id="mem-1")
    assert isinstance(evidence, TokenEvidence)
    assert evidence.token_count > 0

    #: structural conformance to the accepted protocol, checked by shape
    import inspect

    from src.scoring.backends import ScoringBackend

    required = inspect.signature(ScoringBackend.token_evidence).parameters
    assert set(inspect.signature(backend.token_evidence).parameters) == set(required) - {"self"}
    assert isinstance(backend.identity, str) and len(backend.identity) == 64


def test_the_canonical_engine_scores_through_the_production_backend() -> None:
    from src.provenance.config import resolve_config

    configs = REPO_ROOT / "configs"
    scoring = resolve_config(configs / "attacks" / "scoring.json", config_root=configs)
    backend = make_backend()
    context = ScoringContext(
        root=REPO_ROOT,
        backend=backend,
        reference_backend=make_backend(source=DeterministicLogitsSource(salt="reference")),
        resolved_config=scoring,
        model_revision=backend.model_revision,
        tokenizer_sha256=backend.tokenizer.identity(),
        environment_lock_sha256="a" * 64,
    )
    table = score_records(
        arm=Arm.CANARY,
        score=ScoreFamily.REFERENCE_LOSS,
        view=View.DESCENDANT,
        artifact="art-1",
        seed=101,
        fold=None,
        record_ids=list(RECORDS),
        membership={record: int(record.startswith("mem-")) for record in RECORDS},
        context=context,
    )
    assert len(table.rows) == len(RECORDS)
    assert all(np.isfinite(row.scalar_score) for row in table.rows)


def test_r18_an_over_length_record_is_right_truncated_as_the_policy_declares() -> None:
    """The declared policy is EXECUTED: provenance and execution must describe one thing."""
    full = make_backend()
    batch = full.batches["mem-1"]
    assert len(batch.input_ids) > 4

    truncated = make_backend(tokenizer=tokenizer_identity(max_sequence_length=4))
    evidence = truncated.token_evidence(artifact_id="art-1", record_id="mem-1")
    assert evidence.valid.size == 3, "four retained tokens leave three predicted positions"

    kept = truncated.truncate(batch)
    assert kept.input_ids == batch.input_ids[:4], "the LEFT prefix is retained"
    assert kept.attention_mask == batch.attention_mask[:4]


@pytest.mark.parametrize("limit", [9, 8, 3])
def test_r18_every_aligned_structure_truncates_together(limit: int) -> None:
    from src.backend.scoring_backend import TokenBatch

    batch = TokenBatch(
        record_id="r",
        input_ids=tuple(range(9)),
        attention_mask=(1,) * 8 + (0,),
        prompt_mask=(True,) * 3 + (False,) * 6,
    )
    kept = batch.right_truncated(limit)
    assert len(kept.input_ids) == min(limit, 9)
    assert len(kept.attention_mask) == len(kept.input_ids)
    assert kept.prompt_mask is not None and len(kept.prompt_mask) == len(kept.input_ids)
    assert kept.input_ids == batch.input_ids[: len(kept.input_ids)]


def test_r18_a_record_at_exactly_the_limit_is_untouched() -> None:
    from src.backend.scoring_backend import TokenBatch

    batch = TokenBatch(record_id="r", input_ids=(1, 2, 3, 4), attention_mask=(1, 1, 1, 1))
    assert batch.right_truncated(4) is batch


def test_r18_an_unimplemented_truncation_policy_fails_closed() -> None:
    from dataclasses import replace

    backend = make_backend()
    other = replace(backend, tokenizer=replace(backend.tokenizer, truncation_policy="LEFT"))
    with pytest.raises(ScoringBackendError, match="not implemented by this backend"):
        other.token_evidence(artifact_id="art-1", record_id="mem-1")


def test_r18_the_declared_policy_is_the_one_the_backend_implements() -> None:
    from src.backend.scoring_backend import RIGHT_TRUNCATE
    from src.materials import material_text

    assert material_text(backend_settings(REPO_ROOT), "truncation_policy") == RIGHT_TRUNCATE
    assert make_backend().tokenizer.truncation_policy == RIGHT_TRUNCATE


def test_an_untokenized_record_is_refused_not_invented() -> None:
    with pytest.raises(ScoringBackendError, match="was not tokenized"):
        make_backend().token_evidence(artifact_id="art-1", record_id="unknown")


# ==================================================================== C20 identity bound
def test_c20_the_backend_identity_binds_code_revision_tokenizer_and_precision() -> None:
    baseline = make_backend().identity
    assert baseline == make_backend().identity
    assert baseline != make_backend(model_revision="8" * 40).identity
    assert baseline != make_backend(tokenizer=tokenizer_identity(revision="2" * 40)).identity
    assert baseline != make_backend(precision="float32").identity


def test_the_cache_identity_spans_the_backend_source() -> None:
    """02 §C7: the adapter, masking/reduction, reference loss and Min-K are all covered."""
    assert set(BACKEND_SOURCE_FILES) <= set(SCORING_SOURCE_FILES)
    for required in (
        "src/scoring/reference.py",
        "src/backend/forward.py",
        "src/backend/scoring_backend.py",
        "src/backend/adapters.py",
    ):
        assert required in SCORING_SOURCE_FILES


# ==================================================================== C21-C24 cache
def identity_for(**overrides: object) -> CacheIdentity:
    backend = make_backend()
    values: dict[str, object] = {
        "artifact_id": "art-1",
        "model_revision": backend.model_revision,
        "tokenizer_sha256": backend.tokenizer.identity(),
        "record_set_sha256": record_set_identity(RECORDS),
        "precision": "bfloat16",
        "max_sequence_length": 128,
        "scoring_code_hash": scoring_code_hash(REPO_ROOT, {"k": 1}),
        "backend_identity": backend.identity,
        "reference_backend_identity": make_backend(
            source=DeterministicLogitsSource(salt="reference")
        ).identity,
        "score_family": "REFERENCE_LOSS",
        "resolved_config_sha256": "b" * 64,
        "environment_lock_sha256": "c" * 64,
    }
    values.update(overrides)
    return CacheIdentity(**values)  # type: ignore[arg-type]


def _rows() -> list[dict[str, object]]:
    return [
        {"record_id": record, "scalar_score": float(index) / 7.0, "token_count": 6}
        for index, record in enumerate(RECORDS)
    ]


def test_c24_an_unchanged_call_is_a_stable_hit_with_identical_scalars(tmp_path: Path) -> None:
    cache = ScoreCache(root=tmp_path)
    identity = identity_for()
    stored = _rows()
    cache.store(identity, stored)  # type: ignore[arg-type]
    served = cache.load(identity_for())
    assert served is not None
    assert served == stored, "a hit must return bitwise-identical scalars"


def test_c21_a_tokenizer_identity_change_is_a_miss(tmp_path: Path) -> None:
    cache = ScoreCache(root=tmp_path)
    cache.store(identity_for(), _rows())  # type: ignore[arg-type]
    other = tokenizer_identity(revision="7" * 40).identity()
    assert cache.load(identity_for(tokenizer_sha256=other)) is None


def test_c23_a_model_revision_change_is_a_miss(tmp_path: Path) -> None:
    cache = ScoreCache(root=tmp_path)
    cache.store(identity_for(), _rows())  # type: ignore[arg-type]
    assert cache.load(identity_for(model_revision="0" * 40)) is None


def test_a_precision_or_record_set_change_is_a_miss(tmp_path: Path) -> None:
    cache = ScoreCache(root=tmp_path)
    cache.store(identity_for(), _rows())  # type: ignore[arg-type]
    assert cache.load(identity_for(precision="float32")) is None
    assert cache.load(identity_for(record_set_sha256=record_set_identity(RECORDS[:2]))) is None


def test_c22_a_backend_source_change_is_a_miss(tmp_path: Path) -> None:
    """Edit the executable backend semantics on a copied tree; the identity must move."""
    root = tmp_path / "repo"
    root.mkdir()
    for relative in ("src", "configs"):
        shutil.copytree(REPO_ROOT / relative, root / relative, dirs_exist_ok=True)

    before_code = backend_code_hash(root)
    before_scoring = scoring_code_hash(root, {"k": 1})

    target = root / "src" / "backend" / "forward.py"
    source = target.read_text(encoding="utf-8")
    target.write_text(
        source.replace(
            "predicted = log_softmax(values[:-1, :])", "predicted = log_softmax(values)"
        ),
        encoding="utf-8",
    )
    assert backend_code_hash(root) != before_code, "the backend identity must move"
    assert scoring_code_hash(root, {"k": 1}) != before_scoring, "the cache identity must move"

    cache = ScoreCache(root=tmp_path)
    cache.store(identity_for(), _rows())  # type: ignore[arg-type]
    assert cache.load(identity_for(scoring_code_hash=scoring_code_hash(root, {"k": 1}))) is None


def test_a_reducer_change_is_also_a_miss(tmp_path: Path) -> None:
    """The scorer half of 02 §C7's requirement, checked the same way."""
    root = tmp_path / "repo"
    root.mkdir()
    for relative in ("src", "configs"):
        shutil.copytree(REPO_ROOT / relative, root / relative, dirs_exist_ok=True)
    before = scoring_code_hash(root, {"k": 1})
    target = root / "src" / "scoring" / "reference.py"
    target.write_text(
        target.read_text(encoding="utf-8").replace("count = max(1,", "count = max(2,"),
        encoding="utf-8",
    )
    assert scoring_code_hash(root, {"k": 1}) != before


def test_the_resolved_config_is_part_of_the_identity() -> None:
    assert scoring_code_hash(REPO_ROOT, {"k": 1}) != scoring_code_hash(REPO_ROOT, {"k": 2})


# ==================================================================== R6 runtime config identity
def _rewritten_root(tmp_path: Path, **settings: object) -> Path:
    """A copied tree whose backend runtime config carries the given material values."""
    import shutil

    root = tmp_path / f"repo-{abs(hash(tuple(sorted(settings.items())))) % 10**8}"
    root.mkdir()
    for relative in ("src", "configs"):
        shutil.copytree(REPO_ROOT / relative, root / relative, dirs_exist_ok=True)
    path = root / "configs" / "backend" / "runtime.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    for key, value in settings.items():
        document[key]["value"] = value
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return root


def test_r6_the_resolved_backend_runtime_config_is_bound_into_the_identity() -> None:
    """02 §C7: source bytes are only half of executable backend semantics."""
    from src.backend.settings import backend_runtime_config_sha256
    from src.scoring.cache import backend_runtime_config_sha256 as cache_side

    assert backend_runtime_config_sha256(REPO_ROOT) == cache_side(REPO_ROOT)
    assert len(backend_runtime_config_sha256(REPO_ROOT)) == 64


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("attn_implementation", "sdpa"),
        ("attn_implementation", "flash_attention_2"),
        ("forward_precision", "float32"),
        ("trust_remote_code", True),
        ("lora_scaling_rule", "ALPHA_OVER_SQRT_RANK"),
        ("use_rslora", True),
        ("training_precision", "float32"),
        ("local_files_only", False),
    ],
)
def test_r6_a_material_backend_setting_change_is_a_cache_miss(
    tmp_path: Path, key: str, value: object
) -> None:
    """None of these edits a single source byte, and every one changes the numbers."""
    from src.backend.scoring_backend import backend_code_hash

    root = _rewritten_root(tmp_path, **{key: value})
    assert backend_code_hash(root) != backend_code_hash(REPO_ROOT), key
    moved = scoring_code_hash(root, {"k": 1})
    assert moved != scoring_code_hash(REPO_ROOT, {"k": 1}), key

    cache = ScoreCache(root=tmp_path)
    cache.store(identity_for(), _rows())  # type: ignore[arg-type]
    assert cache.load(identity_for(scoring_code_hash=moved)) is None, key


def test_r6_an_unchanged_runtime_config_leaves_the_identity_alone(tmp_path: Path) -> None:
    from src.backend.scoring_backend import backend_code_hash

    root = _rewritten_root(tmp_path, attn_implementation="eager")
    assert backend_code_hash(root) == backend_code_hash(REPO_ROOT)
    assert scoring_code_hash(root, {"k": 1}) == scoring_code_hash(REPO_ROOT, {"k": 1})

    cache = ScoreCache(root=tmp_path)
    stored = _rows()
    cache.store(identity_for(), stored)  # type: ignore[arg-type]
    served = cache.load(identity_for(scoring_code_hash=scoring_code_hash(root, {"k": 1})))
    assert served == stored, "an unchanged config must still be a bitwise-identical hit"


def test_r6_a_missing_backend_runtime_config_is_a_distinct_identity(tmp_path: Path) -> None:
    """Absence is recorded, not silently equal to some present configuration."""
    from src.scoring.cache import BACKEND_RUNTIME_ABSENT, backend_runtime_config_sha256

    root = tmp_path / "stripped"
    root.mkdir()
    for relative in ("src", "configs"):
        shutil.copytree(REPO_ROOT / relative, root / relative, dirs_exist_ok=True)
    (root / "configs" / "backend" / "runtime.json").unlink()

    assert backend_runtime_config_sha256(root) == BACKEND_RUNTIME_ABSENT
    assert backend_runtime_config_sha256(root) != backend_runtime_config_sha256(REPO_ROOT)
    assert scoring_code_hash(root, {"k": 1}) != scoring_code_hash(REPO_ROOT, {"k": 1})


def test_r7_the_cache_change_is_bounded_to_identity(tmp_path: Path) -> None:
    """02 §C7 authorises the identity surface only; lookup, write and rows are untouched."""
    import subprocess

    diff = subprocess.run(  # noqa: S603
        ["git", "-C", str(REPO_ROOT), "diff", "--", "src/scoring/cache.py"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    changed = [
        line[1:].strip()
        for line in diff.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    forbidden = ("def store", "def load", "def path_for", "CACHE_PAYLOAD_FIELDS", "scalar_score")
    for line in changed:
        assert not any(marker in line for marker in forbidden), line
