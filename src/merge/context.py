"""The one resolved execution context every S07 operator runs under [AUTH: 01 §10, §17].

Opaque and factory-issued. `MergeExecutionContext.__init__` raises, `__setattr__` raises, and
the class is not a dataclass, so there is no `dataclasses.replace` path and no ordinary API
that can mint a context claiming an arbitrary config hash, version, dtype, scheme or backend.
The only way to obtain one is `load_merge_execution_context`, which reads the canonical
resolved merge config and hashes the exact snapshot it consumed.

Source declares which values this code can execute. That is a *capability* set, not a second
copy of the scientific truth: a config naming something outside it fails closed rather than
being silently downgraded to whatever the source happens to implement.

Precision is part of the context because it changes the bytes:

    arithmetic_dtype     float32   scientific merge artifacts   [01 §10; 00 §25 P0-C1]
    diagnostic_dtype     float64   derived scalar quantities    [01 §10]
    svd_workspace_dtype  float64   numerical workspace, cast back before hashing

float32 is the ONLY supported artifact arithmetic. A float64 scientific context is refused,
because a float64 result would not be the artifact the recovery baselines are defined on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np

from src.provenance.hashing import JSONDocument, JSONValue, sha256_canonical

CONTEXT_SCHEMA: Final = "s07.merge-execution-context.v2"

#: What this code can execute. Anything else fails closed, never downgrades.
SUPPORTED_OPERATOR_VERSIONS: Final[frozenset[str]] = frozenset({"s07.operators.v1"})
SUPPORTED_MASK_SCHEMES: Final[frozenset[str]] = frozenset({"s07.dare-mask.sha256-counter.v2"})
SUPPORTED_SVD_BACKENDS: Final[frozenset[str]] = frozenset(
    {"numpy.linalg.svd(full_matrices=False, dtype=float64)"}
)
#: 01 §10 / 00 §25: the scientific artifact dtype is float32 and nothing else.
SUPPORTED_ARITHMETIC_DTYPES: Final[frozenset[str]] = frozenset({"float32"})
SUPPORTED_DIAGNOSTIC_DTYPES: Final[frozenset[str]] = frozenset({"float64"})
SUPPORTED_WORKSPACE_DTYPES: Final[frozenset[str]] = frozenset({"float32", "float64"})

_NUMPY_DTYPES: Final[dict[str, np.dtype[Any]]] = {
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}


class ExecutionContextError(ValueError):
    """The resolved merge configuration cannot be executed by this code."""


class MergeExecutionContext:
    """Every material operator setting, resolved once and bound to every result.

    Opaque: no public constructor, no attribute assignment, not a dataclass.
    """

    __slots__ = (
        "_arithmetic_dtype",
        "_config_sha256",
        "_diagnostic_dtype",
        "_effective_rank_tolerance",
        "_identity",
        "_mask_scheme",
        "_operator_version",
        "_svd_backend",
        "_svd_workspace_dtype",
    )

    # Bare annotations, not assignments: they type the slots without creating class
    # attributes, which __slots__ would reject.
    _operator_version: str
    _mask_scheme: str
    _arithmetic_dtype: str
    _diagnostic_dtype: str
    _svd_backend: str
    _svd_workspace_dtype: str
    _effective_rank_tolerance: float
    _config_sha256: str
    _identity: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "MergeExecutionContext is factory-issued; use load_merge_execution_context() so"
            " the context hash is computed from the config actually consumed [AUTH: 01 §17]"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("MergeExecutionContext is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("MergeExecutionContext is immutable")

    # ---------------------------------------------------------------- projections
    @property
    def operator_version(self) -> str:
        return self._operator_version

    @property
    def mask_scheme(self) -> str:
        return self._mask_scheme

    @property
    def arithmetic_dtype(self) -> str:
        return self._arithmetic_dtype

    @property
    def diagnostic_dtype(self) -> str:
        return self._diagnostic_dtype

    @property
    def svd_backend(self) -> str:
        return self._svd_backend

    @property
    def svd_workspace_dtype(self) -> str:
        return self._svd_workspace_dtype

    @property
    def effective_rank_tolerance(self) -> float:
        return self._effective_rank_tolerance

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
        """The canonical little-endian code every content hash is taken over."""
        return "<f4" if self._arithmetic_dtype == "float32" else "<f8"

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": CONTEXT_SCHEMA,
            "operator_version": self._operator_version,
            "mask_scheme": self._mask_scheme,
            "arithmetic_dtype": self._arithmetic_dtype,
            "diagnostic_dtype": self._diagnostic_dtype,
            "svd_backend": self._svd_backend,
            "svd_workspace_dtype": self._svd_workspace_dtype,
            "effective_rank_tolerance": self._effective_rank_tolerance,
            "config_sha256": self._config_sha256,
        }

    def identity(self) -> str:
        return self._identity

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"MergeExecutionContext({self._operator_version}, {self._arithmetic_dtype},"
            f" {self._identity[:12]})"
        )


def _validate(
    *,
    operator_version: str,
    mask_scheme: str,
    arithmetic_dtype: str,
    diagnostic_dtype: str,
    svd_backend: str,
    svd_workspace_dtype: str,
    effective_rank_tolerance: float,
) -> None:
    """Every reason this resolved configuration cannot be executed by this code."""
    problems: list[str] = []
    if operator_version not in SUPPORTED_OPERATOR_VERSIONS:
        problems.append(
            f"operator_version {operator_version!r} is not executable by this code"
            f" (supported: {sorted(SUPPORTED_OPERATOR_VERSIONS)})"
        )
    if mask_scheme not in SUPPORTED_MASK_SCHEMES:
        problems.append(f"mask scheme {mask_scheme!r} is not implemented here")
    if svd_backend not in SUPPORTED_SVD_BACKENDS:
        problems.append(f"SVD backend {svd_backend!r} is not implemented here")
    if arithmetic_dtype not in SUPPORTED_ARITHMETIC_DTYPES:
        problems.append(
            f"arithmetic dtype {arithmetic_dtype!r} is not the frozen scientific artifact"
            " dtype float32 [AUTH: 01 §10; 00 §25 P0-C1]"
        )
    if diagnostic_dtype not in SUPPORTED_DIAGNOSTIC_DTYPES:
        problems.append(f"diagnostic dtype {diagnostic_dtype!r} is not supported")
    if svd_workspace_dtype not in SUPPORTED_WORKSPACE_DTYPES:
        problems.append(f"SVD workspace dtype {svd_workspace_dtype!r} is not supported")
    if not 0.0 < effective_rank_tolerance <= 1.0:
        problems.append("the effective-rank tolerance must lie in (0, 1]")
    if problems:
        raise ExecutionContextError(
            "; ".join(problems)
            + " — refusing to execute one version while recording another [AUTH: 01 §17]"
        )


def load_merge_execution_context(resolved_config: JSONDocument) -> MergeExecutionContext:
    """Issue the context from a canonical resolved merge-config snapshot.

    The config hash is computed here, over the exact snapshot consumed, so a result can never
    carry a config identity that does not describe the settings it ran under.
    """
    from src.materials import material_number, material_text
    from src.merge.settings import merge_document_settings

    settings = merge_document_settings(resolved_config)
    operator_version = material_text(settings, "operator_version")
    mask_scheme = material_text(settings, "dare_mask_scheme")
    arithmetic_dtype = material_text(settings, "arithmetic_dtype")
    diagnostic_dtype = material_text(settings, "diagnostic_dtype")
    svd_backend = material_text(settings, "svd_backend")
    svd_workspace_dtype = material_text(settings, "svd_workspace_dtype")
    tolerance = material_number(settings, "effective_rank_tolerance")
    _validate(
        operator_version=operator_version,
        mask_scheme=mask_scheme,
        arithmetic_dtype=arithmetic_dtype,
        diagnostic_dtype=diagnostic_dtype,
        svd_backend=svd_backend,
        svd_workspace_dtype=svd_workspace_dtype,
        effective_rank_tolerance=tolerance,
    )

    config_sha256 = sha256_canonical(dict(resolved_config))
    payload: dict[str, JSONValue] = {
        "schema": CONTEXT_SCHEMA,
        "operator_version": operator_version,
        "mask_scheme": mask_scheme,
        "arithmetic_dtype": arithmetic_dtype,
        "diagnostic_dtype": diagnostic_dtype,
        "svd_backend": svd_backend,
        "svd_workspace_dtype": svd_workspace_dtype,
        "effective_rank_tolerance": tolerance,
        "config_sha256": config_sha256,
    }

    context = object.__new__(MergeExecutionContext)
    object.__setattr__(context, "_operator_version", operator_version)
    object.__setattr__(context, "_mask_scheme", mask_scheme)
    object.__setattr__(context, "_arithmetic_dtype", arithmetic_dtype)
    object.__setattr__(context, "_diagnostic_dtype", diagnostic_dtype)
    object.__setattr__(context, "_svd_backend", svd_backend)
    object.__setattr__(context, "_svd_workspace_dtype", svd_workspace_dtype)
    object.__setattr__(context, "_effective_rank_tolerance", tolerance)
    object.__setattr__(context, "_config_sha256", config_sha256)
    object.__setattr__(context, "_identity", sha256_canonical(payload))
    return context


def resolve_merge_context(root: Path) -> MergeExecutionContext:
    """Resolve the repository's S07 config and issue its context."""
    from src.merge.settings import merge_settings

    return load_merge_execution_context(merge_settings(root).document)
