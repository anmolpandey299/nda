"""Deterministic tiny fixtures for Block C. No network, no research data, no real model.

Every record here is generated from a fixed label. Nothing in this module reads PubMed, and
the corpora it builds are labelled `TEST_FIXTURE_NOT_RESEARCH_DATA` wherever they reach a
manifest [AUTH: 01 §20].
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from src.data.corpus import SourceRecord
from src.data.normalise import NormalisedRecord, normalise_record

_TOPICS = (
    "hepatic microvascular perfusion",
    "cortical spreading depolarisation",
    "renal tubular acidosis",
    "myeloid lineage commitment",
    "atrial conduction remodelling",
    "chondrocyte mechanotransduction",
    "pulmonary surfactant turnover",
    "retinal ganglion resilience",
)
_VERBS = ("modulates", "predicts", "attenuates", "reflects", "constrains")
_OBJECTS = (
    "outcome in a prospective cohort",
    "recovery after controlled injury",
    "response to graded stimulation",
    "variability across observers",
)


def synthetic_text(index: int) -> tuple[str, str]:
    """A title/abstract pair that is deterministic in `index` and never a real record."""
    digest = hashlib.sha256(f"blockc/fixture/{index}".encode()).digest()
    topic = _TOPICS[digest[0] % len(_TOPICS)]
    verb = _VERBS[digest[1] % len(_VERBS)]
    obj = _OBJECTS[digest[2] % len(_OBJECTS)]
    title = f"Synthetic study {index:05d}: {topic} {verb} {obj}"
    abstract = (
        f"Background. We examined whether {topic} {verb} {obj}. "
        f"Methods. Fixture cohort {digest[3]:03d} with {digest[4] % 90 + 10} simulated units. "
        f"Results. Effect index {digest[5]:03d} of {digest[6]:03d}, marker {digest[7]:02x}. "
        "Conclusion. This record is generated and describes nothing observed."
    )
    return title, abstract


def source_records(
    count: int,
    *,
    start: int = 0,
    publication_date: str = "2026-05-14",
    indexing_date: str = "2026-06-02",
) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for offset in range(count):
        index = start + offset
        title, abstract = synthetic_text(index)
        records.append(
            SourceRecord(
                record_id=f"FIX{index:06d}",
                pmid=f"9{index:07d}",
                publication_date=publication_date,
                indexing_date=indexing_date,
                source_identity=f"fixture-source/{index}",
                fields={"title": title, "abstract": abstract},
            )
        )
    return records


def normalised(records: Sequence[SourceRecord]) -> list[NormalisedRecord]:
    return [normalise_record(record.record_id, record.fields) for record in records]


def tiny_corpus(count: int = 120) -> list[NormalisedRecord]:
    return normalised(source_records(count))


# ----------------------------------------------------------------------------------------
# Block C object builders. Every one produces a NON_EVIDENTIARY fixture object.
# ----------------------------------------------------------------------------------------

FIXTURE_STUDY_SEED = 20260820
FIXTURE_INCLUSION_PROBABILITY = 0.5


def candidate_pool(trained_natural_ids: Sequence[str], pool: Sequence[Any]) -> Any:
    from src.data.membership import natural_candidate_pool

    return natural_candidate_pool(
        list(trained_natural_ids),
        pool,
        study_seed=FIXTURE_STUDY_SEED,
    )


def inclusion_for(pool: Sequence[Any], training_seed: int) -> Any:
    from src.data.membership import derive_canary_inclusion
    from src.training.seeds import seed_families

    return derive_canary_inclusion(
        pool,
        seeds=seed_families(training_seed, differentially_private=False),
        probability=FIXTURE_INCLUSION_PROBABILITY,
    )


def fixture_model() -> Any:
    """A panel member standing in for an unacquired research model."""
    from src.training.model_contract import PanelMember

    return PanelMember(
        alias="tiny_fixture",
        model_id="fixture/tiny",
        architecture_family="tiny_fixture",
        base_or_instruct="base",
        role="TEST_FIXTURE_NOT_A_RESEARCH_SUBJECT",
    )


def fixture_contract(
    *,
    plan: Any,
    base_parameter_names: Sequence[str],
    mapping: Any,
    training_seed: int,
    optimizer_step_budget: int,
    rank: int = 8,
    scaling: float = 2.0,
    learning_rate: float = 0.5,
    batch_size: int = 3,
    differentially_private: bool = False,
) -> Any:
    from src.training.execution import (
        DP_SAMPLE_LEVEL,
        NON_DP,
        authorise_training,
        reference_backend,
    )
    from src.training.seeds import seed_families

    return authorise_training(
        plan=plan,
        model=fixture_model(),
        mapping=mapping,
        base_parameter_names=list(base_parameter_names),
        rank=rank,
        scaling=scaling,
        dropout=0.0,
        learning_rate=learning_rate,
        batch_size=batch_size,
        optimizer_step_budget=optimizer_step_budget,
        precision="float64",
        sequence_length=128,
        seeds=seed_families(training_seed, differentially_private=differentially_private),
        privacy_regime=DP_SAMPLE_LEVEL if differentially_private else NON_DP,
        backend=reference_backend(),
        resolved_config_sha256="f" * 64,
    )
