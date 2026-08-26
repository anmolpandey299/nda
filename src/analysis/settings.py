"""Resolved-config access for the S03/S04 engine. One path, Block A's [AUTH: 01 §17].

Every material constant this block executes on is read from `configs/**` through the
accepted Block A resolver, so mutating a config changes behaviour and cannot be shadowed by a
source default or a test literal. Nothing here re-implements resolution; it names the files
and hands back the resolved documents plus their identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.provenance.config import ConfigError, resolve_config, resolved_config_sha256
from src.provenance.hashing import JSONValue

SCORING_CONFIG: Final = "attacks/scoring.json"
ANALYSIS_CONFIG: Final = "p0/analysis.json"
DRY_RUN_CONFIG: Final = "p0/dry_run.json"


@dataclass(frozen=True)
class ResolvedSettings:
    """One resolved config plus the identity that binds a run to it [AUTH: 01 §15, §17]."""

    name: str
    document: dict[str, JSONValue]
    sha256: str

    def number(self, key: str) -> float:
        value = self.document.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{self.name}: {key!r} is not a number")
        return float(value)

    def integer(self, key: str) -> int:
        value = self.document.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{self.name}: {key!r} is not an integer")
        return value

    def text(self, key: str) -> str:
        value = self.document.get(key)
        if not isinstance(value, str):
            raise ConfigError(f"{self.name}: {key!r} is not a string")
        return value

    def numbers(self, key: str) -> tuple[float, ...]:
        value = self.document.get(key)
        if isinstance(value, str) or not isinstance(value, list):
            raise ConfigError(f"{self.name}: {key!r} is not a list")
        return tuple(float(item) for item in value)

    def integers(self, key: str) -> tuple[int, ...]:
        return tuple(int(item) for item in self.numbers(key))


def load_settings(root: Path, relative: str) -> ResolvedSettings:
    """Resolve one config under `configs/` through the Block A resolver."""
    configs = root / "configs"
    document = resolve_config(configs / relative, config_root=configs)
    return ResolvedSettings(
        name=relative, document=document, sha256=resolved_config_sha256(document)
    )


# Deliberately uncached. Resolution is cheap, and a memoised settings object would keep
# serving a configuration that is no longer on disk, which is the opposite of what a
# config-controls-execution guarantee means [AUTH: 01 §17].


def scoring_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, SCORING_CONFIG)


def analysis_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, ANALYSIS_CONFIG)


def dry_run_settings(root: Path) -> ResolvedSettings:
    return load_settings(root, DRY_RUN_CONFIG)


# ----------------------------------------------------------------------------------------
# Typed views. Every material key a Block B execution path needs is read here, so a config
# field that stops being consumed is visible as a dead key rather than a silent default.
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoringConstants:
    """configs/attacks/scoring.json [AUTH: 00 §14, §16, §34A; 02 §C1, §C2, §C7]."""

    target_fpr: float
    min_k_fraction: float
    precision: str
    max_sequence_length: int
    n_outer_folds: int
    fold_assignment_seed: int
    ridge_regularisation: float
    sha256: str


def scoring_constants(root: Path) -> ScoringConstants:
    settings = scoring_settings(root)
    return ScoringConstants(
        target_fpr=settings.number("target_fpr"),
        min_k_fraction=settings.number("min_k_fraction"),
        precision=settings.text("precision"),
        max_sequence_length=settings.integer("max_sequence_length"),
        n_outer_folds=settings.integer("n_outer_folds"),
        fold_assignment_seed=settings.integer("fold_assignment_seed"),
        ridge_regularisation=settings.number("ridge_regularisation"),
        sha256=settings.sha256,
    )


@dataclass(frozen=True)
class AnalysisConstants:
    """configs/p0/analysis.json [AUTH: 00 §17-§20.8, §23A, §29, §32]."""

    null_level: float
    eligibility_median_floor: float
    bootstrap_replicates: int
    bootstrap_alpha: float
    denominator_unstable_above: float
    denominator_unusable_above: float
    functional_relative_floor: float
    functional_max_relative_se: float
    e50_level: float
    e50_min_support: int
    sha256: str


def analysis_constants(root: Path) -> AnalysisConstants:
    settings = analysis_settings(root)
    return AnalysisConstants(
        null_level=settings.number("null_level"),
        eligibility_median_floor=settings.number("eligibility_median_floor"),
        bootstrap_replicates=settings.integer("bootstrap_replicates"),
        bootstrap_alpha=settings.number("bootstrap_alpha"),
        denominator_unstable_above=settings.number("denominator_unstable_above"),
        denominator_unusable_above=settings.number("denominator_unusable_above"),
        functional_relative_floor=settings.number("functional_relative_floor"),
        functional_max_relative_se=settings.number("functional_max_relative_se"),
        e50_level=settings.number("e50_level"),
        e50_min_support=settings.integer("e50_min_support"),
        sha256=settings.sha256,
    )


def trial_design(root: Path):  # type: ignore[no-untyped-def]
    """The frozen planted-null design, built from resolved dry-run settings [00 §34B.1]."""
    from src.analysis.dryrun import TrialDesign

    settings = dry_run_settings(root)
    return TrialDesign(
        n_cells=settings.integer("n_registry_cells"),
        seeds=settings.integers("training_seeds"),
        operators=tuple(
            str(name)
            for name in settings.document["operators"]  # type: ignore[union-attr]
        ),
        error_low=settings.number("error_low"),
        error_high=settings.number("error_high"),
        cell_sd=settings.number("cell_sd"),
        row_sd=settings.number("row_sd"),
        recovery_base=settings.number("recovery_base"),
        gap_cell_sd=settings.number("gap_cell_sd"),
        gap_row_sd=settings.number("gap_row_sd"),
    )
