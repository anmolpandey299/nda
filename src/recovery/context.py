"""The one resolved execution context every S08 solver runs under [AUTH: 01 §10, §17].

Opaque and factory-issued, following the S07 precedent: `__init__` raises, `__setattr__`
raises, and this is not a dataclass, so there is no `dataclasses.replace` path and no ordinary
API that can mint a context claiming an arbitrary config hash, version, dtype or algorithm.
The only way to obtain one is `load_recovery_execution_context`, which reads the canonical
resolved recovery config and hashes the exact snapshot it consumed.

Source declares which values this code can execute. That is a capability set, not a second
copy of the scientific truth: a config naming something outside it fails closed rather than
being silently downgraded.

**One snapshot.** The factory takes a single deep canonical copy of the supplied document and
parses every setting AND the config hash from that one copy. A Mapping whose repeated reads
return different values therefore cannot make the executed settings and the recorded hash
describe different documents [AUTH: 01 §16, §17].

**One scientific profile.** `s08.recovery.v1` names a specific frozen method, so its material
values are pinned here as well as declared in config: a document claiming that version while
changing the iteration count, the residual rank, the numerical-rank tolerance or the artifact
dtype describes a different method and is refused rather than hashed under the same name. A
planted-truth synthetic scenario that needs a shorter solve uses `fixture_recovery_context`,
which stamps `FIXTURE_ONLY_NOT_SCIENTIFIC` on the context and on every result issued under it.

Precision follows the frozen policy, because it changes the bytes:

    arithmetic_dtype     float32   scientific recovered artifacts   [01 §10; 00 §25 P0-C1]
    diagnostic_dtype     float64   derived scalar quantities        [01 §10]
    svd_workspace_dtype  float64   truncation workspace, cast back before it re-enters
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.provenance.hashing import JSONDocument, JSONValue, sha256_canonical

CONTEXT_SCHEMA: Final = "s08.recovery-execution-context.v1"

SUPPORTED_RECOVERY_VERSIONS: Final[frozenset[str]] = frozenset({"s08.recovery.v1"})
SUPPORTED_C2_METHODS: Final[frozenset[str]] = frozenset(
    {"SPECTRAL_DETUNING_CORE_FIXED_RANK_REPRODUCTION"}
)
SUPPORTED_C2_METHOD_VERSIONS: Final[frozenset[str]] = frozenset({"s08.c2-core.v1"})
SUPPORTED_RANK_SCHEDULERS: Final[frozenset[str]] = frozenset({"DISABLED_FIXED_STRUCTURAL_RANK"})
SUPPORTED_SVD_BACKENDS: Final[frozenset[str]] = frozenset(
    {
        "numpy.linalg.svd(full_matrices=False, dtype=float64) -> "
        "scipy.linalg.svd(lapack_driver='gesvd') on non-convergence"
    }
)
#: 01 §10 / 00 §25: the scientific artifact dtype is float32 and nothing else.
SUPPORTED_ARITHMETIC_DTYPES: Final[frozenset[str]] = frozenset({"float32"})
SUPPORTED_DIAGNOSTIC_DTYPES: Final[frozenset[str]] = frozenset({"float64"})
SUPPORTED_WORKSPACE_DTYPES: Final[frozenset[str]] = frozenset({"float32", "float64"})

#: The three method-authorisation strings, which are scientific scope rather than tuning. S08
#: implements no hidden-alpha recovery and no DARE incomplete-lineage recovery, and the 00 §28B
#: O3 experiment is gated on an uncalibrated operator choice — so a config declaring any of them
#: open describes a run this code cannot perform, and the context refuses it rather than
#: recording an authorisation it would then ignore [AUTH: 00 §24A.7, §25, §28B; 01 §17].
SUPPORTED_O3_AUTHORIZATIONS: Final[frozenset[str]] = frozenset({"CONDITIONAL_NOT_RUN"})
SUPPORTED_DARE_INCOMPLETE_STATUSES: Final[frozenset[str]] = frozenset({"METHOD_GATED"})
SUPPORTED_HIDDEN_ALPHA_STATUSES: Final[frozenset[str]] = frozenset({"STRUCTURAL_NOT_AUTHORIZED"})

#: What a context is FOR. Derived from which factory ran, never supplied.
SCIENTIFIC_PROFILE: Final = "SCIENTIFIC_FROZEN"
FIXTURE_PROFILE: Final = "FIXTURE_ONLY_NOT_SCIENTIFIC"

#: The material values `s08.recovery.v1` IS. A document claiming this version while changing
#: any of them is describing a different method [AUTH: 00 §25; 01 §17].
PINNED_SCIENTIFIC_VALUES: Final[dict[str, dict[str, Any]]] = {
    "s08.recovery.v1": {
        "c2_n_iters": 1000,
        "linear_residual_rank": 32,
        "conditioning_rank_tolerance": 1e-06,
        "arithmetic_dtype": "float32",
        "diagnostic_dtype": "float64",
        "c2_rank_scheduler": "DISABLED_FIXED_STRUCTURAL_RANK",
        "c1_pass_criterion_ef": 1e-05,
        "partner_norm_match_limit": 1.25,
    }
}

_NUMPY_DTYPES: Final[dict[str, np.dtype[Any]]] = {
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}


class RecoveryContextError(ValueError):
    """The resolved recovery configuration cannot be executed by this code."""


class RecoveryExecutionContext:
    """Every material recovery setting, resolved once and bound to every result."""

    __slots__ = (
        "_arithmetic_dtype",
        "_c1_pass_criterion_ef",
        "_c2_method",
        "_c2_method_version",
        "_c2_n_iters",
        "_c2_rank_scheduler",
        "_conditioning_rank_tolerance",
        "_config_sha256",
        "_dare_incomplete_recovery_status",
        "_diagnostic_dtype",
        "_hidden_alpha_recovery_status",
        "_identity",
        "_linear_residual_rank",
        "_o3_recovery_authorization",
        "_partner_norm_match_limit",
        "_profile",
        "_recovery_version",
        "_svd_backend",
        "_svd_workspace_dtype",
    )

    # Bare annotations type the slots without creating class attributes.
    _recovery_version: str
    _arithmetic_dtype: str
    _diagnostic_dtype: str
    _svd_workspace_dtype: str
    _svd_backend: str
    _c2_method: str
    _c2_method_version: str
    _c2_n_iters: int
    _c2_rank_scheduler: str
    _linear_residual_rank: int
    _conditioning_rank_tolerance: float
    _c1_pass_criterion_ef: float
    _partner_norm_match_limit: float
    _o3_recovery_authorization: str
    _dare_incomplete_recovery_status: str
    _hidden_alpha_recovery_status: str
    _profile: str
    _config_sha256: str
    _identity: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "RecoveryExecutionContext is factory-issued; use"
            " load_recovery_execution_context() so the context hash is computed from the"
            " config actually consumed [AUTH: 01 §17]"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("RecoveryExecutionContext is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("RecoveryExecutionContext is immutable")

    # ---------------------------------------------------------------- projections
    @property
    def recovery_version(self) -> str:
        return self._recovery_version

    @property
    def arithmetic_dtype(self) -> str:
        return self._arithmetic_dtype

    @property
    def diagnostic_dtype(self) -> str:
        return self._diagnostic_dtype

    @property
    def svd_workspace_dtype(self) -> str:
        return self._svd_workspace_dtype

    @property
    def svd_backend(self) -> str:
        return self._svd_backend

    @property
    def c2_method(self) -> str:
        return self._c2_method

    @property
    def c2_method_version(self) -> str:
        return self._c2_method_version

    @property
    def c2_n_iters(self) -> int:
        return self._c2_n_iters

    @property
    def c2_rank_scheduler(self) -> str:
        return self._c2_rank_scheduler

    @property
    def linear_residual_rank(self) -> int:
        return self._linear_residual_rank

    @property
    def conditioning_rank_tolerance(self) -> float:
        return self._conditioning_rank_tolerance

    @property
    def c1_pass_criterion_ef(self) -> float:
        """00 §25 P0-C1. Frozen: no evaluator parameter can move it [AUTH: 00 §34B.1A]."""
        return self._c1_pass_criterion_ef

    @property
    def partner_norm_match_limit(self) -> float:
        """00 §25 P0-C2B.2. Frozen for the same reason."""
        return self._partner_norm_match_limit

    @property
    def profile(self) -> str:
        """SCIENTIFIC_FROZEN, or FIXTURE_ONLY_NOT_SCIENTIFIC for a planted-truth context."""
        return self._profile

    @property
    def is_scientific(self) -> bool:
        return self._profile == SCIENTIFIC_PROFILE

    @property
    def o3_recovery_authorization(self) -> str:
        return self._o3_recovery_authorization

    @property
    def dare_incomplete_recovery_status(self) -> str:
        return self._dare_incomplete_recovery_status

    @property
    def hidden_alpha_recovery_status(self) -> str:
        return self._hidden_alpha_recovery_status

    @property
    def config_sha256(self) -> str:
        return self._config_sha256

    @property
    def arithmetic(self) -> np.dtype[Any]:
        return _NUMPY_DTYPES[self._arithmetic_dtype]

    @property
    def diagnostic(self) -> np.dtype[Any]:
        return _NUMPY_DTYPES[self._diagnostic_dtype]

    @property
    def workspace(self) -> np.dtype[Any]:
        return _NUMPY_DTYPES[self._svd_workspace_dtype]

    @property
    def byte_order_code(self) -> str:
        return "<f4" if self._arithmetic_dtype == "float32" else "<f8"

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": CONTEXT_SCHEMA,
            "recovery_version": self._recovery_version,
            "arithmetic_dtype": self._arithmetic_dtype,
            "diagnostic_dtype": self._diagnostic_dtype,
            "svd_workspace_dtype": self._svd_workspace_dtype,
            "svd_backend": self._svd_backend,
            "c2_method": self._c2_method,
            "c2_method_version": self._c2_method_version,
            "c2_n_iters": self._c2_n_iters,
            "c2_rank_scheduler": self._c2_rank_scheduler,
            "linear_residual_rank": self._linear_residual_rank,
            "conditioning_numerical_rank_tolerance": self._conditioning_rank_tolerance,
            "c1_pass_criterion_ef": self._c1_pass_criterion_ef,
            "partner_norm_match_limit": self._partner_norm_match_limit,
            "profile": self._profile,
            "o3_recovery_authorization": self._o3_recovery_authorization,
            "dare_incomplete_recovery_status": self._dare_incomplete_recovery_status,
            "hidden_alpha_recovery_status": self._hidden_alpha_recovery_status,
            "config_sha256": self._config_sha256,
        }

    def identity(self) -> str:
        return self._identity

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"RecoveryExecutionContext({self._recovery_version}, {self._arithmetic_dtype},"
            f" {self._identity[:12]})"
        )


def _validate(values: dict[str, Any]) -> None:
    """Every reason this resolved configuration cannot be executed by this code."""
    problems: list[str] = []
    if values["recovery_version"] not in SUPPORTED_RECOVERY_VERSIONS:
        problems.append(
            f"recovery_version {values['recovery_version']!r} is not executable by this code"
            f" (supported: {sorted(SUPPORTED_RECOVERY_VERSIONS)})"
        )
    if values["c2_method"] not in SUPPORTED_C2_METHODS:
        problems.append(f"C2 method {values['c2_method']!r} is not implemented here")
    if values["c2_method_version"] not in SUPPORTED_C2_METHOD_VERSIONS:
        problems.append(
            f"C2 method version {values['c2_method_version']!r} is not implemented here"
        )
    if values["c2_rank_scheduler"] not in SUPPORTED_RANK_SCHEDULERS:
        problems.append(f"rank scheduler {values['c2_rank_scheduler']!r} is not implemented here")
    if values["svd_backend"] not in SUPPORTED_SVD_BACKENDS:
        problems.append(f"SVD backend {values['svd_backend']!r} is not implemented here")
    if values["arithmetic_dtype"] not in SUPPORTED_ARITHMETIC_DTYPES:
        problems.append(
            f"arithmetic dtype {values['arithmetic_dtype']!r} is not the frozen scientific"
            " artifact dtype float32 [AUTH: 01 §10; 00 §25 P0-C1]"
        )
    if values["diagnostic_dtype"] not in SUPPORTED_DIAGNOSTIC_DTYPES:
        problems.append(f"diagnostic dtype {values['diagnostic_dtype']!r} is not supported")
    if values["svd_workspace_dtype"] not in SUPPORTED_WORKSPACE_DTYPES:
        problems.append(f"SVD workspace dtype {values['svd_workspace_dtype']!r} is not supported")
    if values["c2_n_iters"] < 1:
        problems.append("the C2 iteration count must be at least one")
    if values["linear_residual_rank"] < 1:
        problems.append("the linear residual rank must be positive")
    if not 0.0 < values["conditioning_rank_tolerance"] <= 1.0:
        problems.append("the conditioning numerical-rank tolerance must lie in (0, 1]")
    if not 0.0 < values["c1_pass_criterion_ef"] < 1.0:
        problems.append("the C1 pass criterion must lie in (0, 1)")
    if values["partner_norm_match_limit"] < 1.0:
        problems.append("the partner norm-match limit must be at least 1.0")
    if values["o3_recovery_authorization"] not in SUPPORTED_O3_AUTHORIZATIONS:
        problems.append(
            f"O3 recovery authorisation {values['o3_recovery_authorization']!r} is not one this"
            " code can honour; the 00 §28B experiment is gated on an uncalibrated operator"
            " choice, so only CONDITIONAL_NOT_RUN is executable [AUTH: 00 §28B; 01 §17]"
        )
    if values["dare_incomplete_recovery_status"] not in SUPPORTED_DARE_INCOMPLETE_STATUSES:
        problems.append(
            f"DARE incomplete-lineage recovery status"
            f" {values['dare_incomplete_recovery_status']!r} claims an authorisation S08"
            " implements no method for [AUTH: 00 §24A.7]"
        )
    if values["hidden_alpha_recovery_status"] not in SUPPORTED_HIDDEN_ALPHA_STATUSES:
        problems.append(
            f"hidden-alpha recovery status {values['hidden_alpha_recovery_status']!r} claims an"
            " authorisation no spec grants and S08 implements no method for [AUTH: 00 §25]"
        )
    if problems:
        raise RecoveryContextError(
            "; ".join(problems)
            + " — refusing to execute one version while recording another [AUTH: 01 §17]"
        )


def _require_pinned_profile(values: dict[str, Any]) -> None:
    """A scientific context IS its frozen values, not merely a version string.

    `s08.recovery.v1` names one specific method. A document that claims that version while
    changing the iteration count, the residual rank, the numerical-rank tolerance, the
    artifact dtype or a frozen evaluation criterion is describing a different method, so it is
    refused instead of being hashed under the accepted name [AUTH: 00 §25, §34B.1A; 01 §17].
    """
    pinned = PINNED_SCIENTIFIC_VALUES.get(values["recovery_version"])
    if pinned is None:  # pragma: no cover - _validate already refused unknown versions
        raise RecoveryContextError(f"{values['recovery_version']!r} has no pinned profile")
    drift = [
        f"{key}={values[key]!r} (frozen: {expected!r})"
        for key, expected in pinned.items()
        if values[key] != expected
    ]
    if drift:
        raise RecoveryContextError(
            f"a scientific {values['recovery_version']} context cannot change "
            + "; ".join(drift)
            + " — that is a different method, not the accepted one. A planted-truth scenario"
            f" needing other values uses fixture_recovery_context(), whose results carry"
            f" {FIXTURE_PROFILE} [AUTH: 00 §25, §34B.1A; 01 §17]"
        )


def _canonical_snapshot(value: Any) -> Any:
    """One deep, plain-data copy, taken once.

    Every parsed setting and the config hash derive from the SAME snapshot, so a Mapping whose
    repeated reads return different values cannot make the executed configuration and the
    recorded `config_sha256` describe different documents [AUTH: 01 §16, §17].
    """
    if isinstance(value, Mapping):
        return {str(key): _canonical_snapshot(item) for key, item in value.items()}
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        return value
    return [_canonical_snapshot(item) for item in value]


def _parse(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Read every material setting out of the ONE snapshot."""
    from src.materials import material_integer, material_number, material_text
    from src.recovery.settings import recovery_document_settings

    settings = recovery_document_settings(snapshot)
    return {
        "recovery_version": material_text(settings, "recovery_version"),
        "arithmetic_dtype": material_text(settings, "arithmetic_dtype"),
        "diagnostic_dtype": material_text(settings, "diagnostic_dtype"),
        "svd_workspace_dtype": material_text(settings, "svd_workspace_dtype"),
        "svd_backend": material_text(settings, "svd_backend"),
        "c2_method": material_text(settings, "c2_method"),
        "c2_method_version": material_text(settings, "c2_method_version"),
        "c2_n_iters": material_integer(settings, "c2_n_iters"),
        "c2_rank_scheduler": material_text(settings, "c2_rank_scheduler"),
        "linear_residual_rank": material_integer(settings, "linear_residual_rank"),
        "conditioning_rank_tolerance": material_number(
            settings, "conditioning_numerical_rank_tolerance"
        ),
        "c1_pass_criterion_ef": material_number(settings, "c1_pass_criterion_ef"),
        "partner_norm_match_limit": material_number(settings, "partner_norm_match_limit"),
        "o3_recovery_authorization": material_text(settings, "o3_recovery_authorization"),
        "dare_incomplete_recovery_status": material_text(
            settings, "dare_incomplete_recovery_status"
        ),
        "hidden_alpha_recovery_status": material_text(settings, "hidden_alpha_recovery_status"),
    }


def _issue(
    values: dict[str, Any], snapshot: dict[str, Any], profile: str
) -> RecoveryExecutionContext:
    """Bind the parsed values, the profile and the hash of the SAME snapshot into a context."""
    config_sha256 = sha256_canonical(snapshot)
    payload: dict[str, JSONValue] = {
        "schema": CONTEXT_SCHEMA,
        **values,
        "profile": profile,
        "config_sha256": config_sha256,
    }
    context = object.__new__(RecoveryExecutionContext)
    for name, value in values.items():
        object.__setattr__(context, f"_{name}", value)
    object.__setattr__(context, "_profile", profile)
    object.__setattr__(context, "_config_sha256", config_sha256)
    object.__setattr__(context, "_identity", sha256_canonical(payload))
    return context


def load_recovery_execution_context(resolved_config: JSONDocument) -> RecoveryExecutionContext:
    """THE scientific factory. One snapshot in, one pinned-profile context out."""
    snapshot = _canonical_snapshot(resolved_config)
    values = _parse(snapshot)
    _validate(values)
    _require_pinned_profile(values)
    return _issue(values, snapshot, SCIENTIFIC_PROFILE)


def fixture_recovery_context(
    resolved_config: JSONDocument, **overrides: Any
) -> RecoveryExecutionContext:
    """A planted-truth context for synthetic scenarios and fixtures. NEVER scientific.

    00 §34B.1 dry runs and the S08 unit fixtures solve tiny matrices, where the frozen 1000
    sweeps and rank-32 residual model are meaningless. This factory exists so they need no
    second *scientific* configuration regime: it applies the requested overrides, skips the
    pinned-profile check, and stamps `FIXTURE_ONLY_NOT_SCIENTIFIC` on the context — which
    every result issued under it then carries in its provenance class and its identity. It is
    not a way to run the accepted method with different numbers.
    """
    snapshot = _canonical_snapshot(resolved_config)
    values = _parse(snapshot)
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise RecoveryContextError(f"{unknown} are not recovery settings")
    values.update(overrides)
    _validate(values)
    return _issue(
        values,
        {**snapshot, "fixture_overrides": _canonical_snapshot(overrides)},
        FIXTURE_PROFILE,
    )


def resolve_recovery_context(root: Path) -> RecoveryExecutionContext:
    """Resolve the repository's S08 config and issue its scientific context."""
    from src.recovery.settings import recovery_settings

    return load_recovery_execution_context(recovery_settings(root).document)
