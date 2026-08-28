"""Resolved-config access for the S07 operator layer [AUTH: 01 §17; 00 §10, §24, §24A].

Every material merge choice is read through `src.materials`, which carries the calibration
status Block C froze. Three keys are `REQUIRED_NOT_CALIBRATED` on purpose:

    dare_selected_drop_probability   p*   [00 §24]
    svd_selected_retained_rank       s*   [00 §24A]
    primary_lossy_operator                [00 §24A.7]

Each is the outcome of an empirical gate that has not been run, so reading one raises. That
is how B13 is enforced structurally rather than by discipline: no S07 code can select p*,
select s*, or name a primary lossy operator, because there is no value to read.

This module names the file and provides typed views. It does not re-implement resolution, and
it does not encode the P0/P1 experiment registry, which S09 owns.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from src.analysis.settings import ResolvedSettings, load_settings
from src.materials import material, material_integers, material_number, material_text
from src.provenance.config import resolved_config_sha256
from src.provenance.hashing import JSONValue

MERGE_CONFIG: Final = "merge/operators.json"


def merge_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, MERGE_CONFIG)


def merge_document_settings(document: Mapping[str, JSONValue]) -> ResolvedSettings:
    """Wrap an already-resolved config snapshot so the material accessors can read it.

    `load_merge_execution_context` takes the snapshot rather than a path, so the context hash
    covers exactly the document consumed.
    """
    return ResolvedSettings(
        name=MERGE_CONFIG, document=dict(document), sha256=resolved_config_sha256(dict(document))
    )


def operator_version(settings: ResolvedSettings) -> str:
    return material_text(settings, "operator_version")


def implemented_operators(settings: ResolvedSettings) -> tuple[str, ...]:
    value = material(settings, "implemented_operators")
    if not isinstance(value, list):
        raise TypeError("implemented_operators must be a list")
    return tuple(str(item) for item in value)


def deferred_operators(settings: ResolvedSettings) -> tuple[str, ...]:
    value = material(settings, "deferred_operators")
    if not isinstance(value, list):
        raise TypeError("deferred_operators must be a list")
    return tuple(str(item) for item in value)


def descendant_counts(settings: ResolvedSettings) -> tuple[int, ...]:
    """The registered k values [AUTH: 00 §11]."""
    return material_integers(settings, "descendant_counts")


def deferred_descendant_counts(settings: ResolvedSettings) -> tuple[int, ...]:
    return material_integers(settings, "deferred_descendant_counts")


def dare_candidates(settings: ResolvedSettings) -> tuple[float, ...]:
    """The 00 §24 candidate drop-ratio grid. A grid, not a selection."""
    value = material(settings, "dare_candidate_drop_probabilities")
    if not isinstance(value, list):
        raise TypeError("dare_candidate_drop_probabilities must be a list")
    return tuple(float(str(item)) for item in value)


def dare_minimum(settings: ResolvedSettings) -> float:
    return material_number(settings, "dare_minimum_drop_probability")


def dare_mask_scheme(settings: ResolvedSettings) -> str:
    """The mask scheme. Execution reads it through `MergeExecutionContext`, not from here."""
    return material_text(settings, "dare_mask_scheme")


def arithmetic_dtype(settings: ResolvedSettings) -> str:
    """01 §10 merge/recovery arithmetic dtype."""
    return material_text(settings, "arithmetic_dtype")


def diagnostic_dtype(settings: ResolvedSettings) -> str:
    """01 §10 dtype for derived scalar quantities."""
    return material_text(settings, "diagnostic_dtype")


def svd_workspace_dtype(settings: ResolvedSettings) -> str:
    return material_text(settings, "svd_workspace_dtype")


def fixture_dare_merge_seed(settings: ResolvedSettings) -> int:
    """One fixture merge seed. Constituent substreams are derived, never caller-selected."""
    from src.materials import material_integer

    return material_integer(settings, "fixture_dare_merge_seed", allow_provisional=True)


def svd_candidate_ranks(settings: ResolvedSettings) -> tuple[int, ...]:
    """The 00 §10.3 candidate retained ranks. A grid, not a selection."""
    return material_integers(settings, "svd_candidate_retained_ranks")


def svd_backend(settings: ResolvedSettings) -> str:
    return material_text(settings, "svd_backend")


def effective_rank_tolerance(settings: ResolvedSettings) -> float:
    """00 §24A.6: sigma_j / sigma_1 >= 1e-6, resolved rather than hard-coded."""
    return material_number(settings, "effective_rank_tolerance")


def gate_statuses(settings: ResolvedSettings) -> dict[str, str]:
    """The three statuses S07 may state, all of which are NOT_RUN or CONDITIONAL."""
    return {
        "O3_STATUS": material_text(settings, "o3_status"),
        "DARE_GATE_STATUS": material_text(settings, "dare_gate_status"),
        "O3_GATE_STATUS": material_text(settings, "o3_gate_status"),
    }
