"""The induced-update boundary: opaque vectors over immutable bytes [00 §3.2, §22].

Merging happens on **induced updates** ΔW = BA, never on raw LoRA factors: the factorisation
is not unique, so a merge defined on factors would not be a merge of the objects the study
measures.

Three properties are structural rather than checked.

**Authoritative storage is immutable.** A task vector's tensors live as Python `bytes` plus a
shape. `__getitem__` returns a fresh read-only ndarray over those bytes, so item assignment
raises and `setflags(write=True)` raises; and even if a view could be made writable, it could
not reach the `bytes`, which Python cannot mutate. `content_identity()` hashes the bytes, so
it is truthful for the life of the object.

**Provenance is issuance, not a label or a hash.** There are two kinds of vector and they are
different classes:

    FixtureTaskVector    generated numbers. Permanently NON_EVIDENTIARY_FIXTURE.
    VerifiedTaskVector   issued ONLY by `verified_task_vector_from_adapter`, which validates
                         the accepted S06 manifest, re-hashes the artifact bytes, loads the
                         adapter and derives ΔW itself.

Neither has a usable public constructor, so there is no
`VerifiedTaskVector(tensors=..., receipt=...)` to hand forged tensors to alongside genuine
proof — the caller never supplies the tensors of a verified vector at all. There is no public
receipt object whose hash could be recomputed to manufacture standing.

Arithmetic dtype comes from the resolved `MergeExecutionContext` — float32 for scientific
artifacts [AUTH: 01 §10; 00 §25] — and is part of the content hash, because it is part of the
bytes.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from src.merge.context import MergeExecutionContext
from src.provenance.hashing import JSONDocument, JSONValue

Matrix = NDArray[Any]

UPDATE_SCHEMA: Final = "s07.task-vector.v3"

#: Generated numbers. Never attributes a scientific result [AUTH: 01 §14].
FIXTURE_INDUCED_UPDATE: Final = "NON_EVIDENTIARY_FIXTURE"
#: Derived through the S06 proof chain from a validated adapter artifact.
VERIFIED_INDUCED_UPDATE: Final = "VERIFIED_INDUCED_UPDATE"

#: Names that betray raw LoRA factors. A necessary check, never a sufficient one: renaming
#: defeats it, which is why the verified path derives ΔW itself instead of trusting a name.
_FACTOR_SUFFIX: Final = re.compile(r"\.lora_[AB]$|\.lora_[ab]\.weight$|^lora_[AB]$")


class TaskVectorError(ValueError):
    """A parameter object cannot serve as a merge input or output."""


class SurfaceMismatchError(TaskVectorError):
    """Two task vectors do not share an exactly compatible authorised parameter surface."""


class ProvenanceError(TaskVectorError):
    """An induced update cannot prove where it came from."""


# ----------------------------------------------------------------------------------------
# the opaque vector
# ----------------------------------------------------------------------------------------


class TaskVector:
    """One constituent's induced update. Opaque, immutable, and never directly constructible.

    Instances are always a `FixtureTaskVector` or a `VerifiedTaskVector`; `provenance_class`
    is derived from the class, so no string can move it.
    """

    __slots__ = ("_bytes", "_context", "_origin", "_shapes")

    _bytes: Mapping[str, bytes]
    _shapes: Mapping[str, tuple[int, int]]
    _context: MergeExecutionContext
    _origin: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            f"{type(self).__name__} is factory-issued; build a fixture with"
            " fixture_induced_update(), or a verified vector with"
            " verified_task_vector_from_adapter() [AUTH: 00 §22; 01 §14]"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a task vector is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a task vector is immutable")

    # ---------------------------------------------------------------- provenance
    @property
    def provenance_class(self) -> str:
        raise NotImplementedError  # pragma: no cover - both subclasses override

    @property
    def verified(self) -> bool:
        return self.provenance_class == VERIFIED_INDUCED_UPDATE

    @property
    def context(self) -> MergeExecutionContext:
        return self._context

    @property
    def origin(self) -> str:
        """A display label. Never part of any identity."""
        return self._origin

    # ---------------------------------------------------------------- canonical view
    @property
    def names(self) -> tuple[str, ...]:
        """The frozen canonical order. Nothing downstream iterates storage directly."""
        return tuple(sorted(self._bytes))

    @property
    def dtype(self) -> str:
        return self._context.arithmetic_dtype

    @property
    def shapes(self) -> dict[str, tuple[int, int]]:
        return {name: self._shapes[name] for name in self.names}

    def __getitem__(self, name: str) -> Matrix:
        """A fresh read-only ndarray over the immutable bytes.

        Item assignment raises, and `setflags(write=True)` raises because the buffer is a
        `bytes` object. No caller ever holds the authoritative storage.
        """
        if name not in self._bytes:
            raise TaskVectorError(f"{name!r} is not on this parameter surface")
        view = np.frombuffer(self._bytes[name], dtype=self._context.byte_order_code)
        return view.reshape(self._shapes[name])

    def copy_of(self, name: str) -> Matrix:
        """A writable copy, provably detached: it owns its own buffer."""
        return np.array(self[name], copy=True)

    def matrices(self) -> list[Matrix]:
        return [self[name] for name in self.names]

    # ---------------------------------------------------------------- identity
    def surface_identity(self) -> str:
        """What the vector covers: names, shapes and dtype. Not the values."""
        digest = hashlib.sha256(f"{UPDATE_SCHEMA}|surface".encode())
        for name in self.names:
            rows, columns = self._shapes[name]
            digest.update(f"|{name}:{rows}x{columns}:{self.dtype}".encode())
        return digest.hexdigest()

    def content_identity(self) -> str:
        """What the vector *is*: the authoritative bytes, in canonical order."""
        digest = hashlib.sha256(f"{UPDATE_SCHEMA}|content|{self.dtype}".encode())
        for name in self.names:
            digest.update(name.encode("utf-8"))
            digest.update(self._bytes[name])
        return digest.hexdigest()

    def frobenius_norm(self) -> float:
        """Accumulated in the diagnostic dtype: a scalar, not artifact bytes [01 §10]."""
        wide = self._context.diagnostic
        return float(np.sqrt(sum(float(np.sum(m.astype(wide) ** 2)) for m in self.matrices())))

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"{type(self).__name__}({self._origin!r}, {len(self._bytes)} tensors,"
            f" {self.dtype}, {self.content_identity()[:12]})"
        )


class FixtureTaskVector(TaskVector):
    """Generated numbers. Cannot become verified: that is a different class entirely."""

    __slots__ = ()

    @property
    def provenance_class(self) -> str:
        return FIXTURE_INDUCED_UPDATE


class VerifiedTaskVector(TaskVector):
    """An induced update derived from a validated S06 adapter artifact.

    The extra slots record what was proved. They are set by the issuing factory and are
    otherwise unreachable, so there is no supported way to attach genuine proof to tensors
    the factory did not derive.
    """

    __slots__ = ("_source",)

    _source: Mapping[str, JSONValue]

    @property
    def provenance_class(self) -> str:
        return VERIFIED_INDUCED_UPDATE

    def source_evidence(self) -> dict[str, JSONValue]:
        """A read-only projection of the S06 proof this vector was issued against."""
        return dict(self._source)


# ----------------------------------------------------------------------------------------
# issuance
# ----------------------------------------------------------------------------------------


def _canonical_bytes(
    tensors: Mapping[str, Matrix], context: MergeExecutionContext
) -> tuple[dict[str, bytes], dict[str, tuple[int, int]]]:
    """Validate, cast to the arithmetic dtype, and freeze into immutable bytes."""
    if not tensors:
        raise TaskVectorError("a task vector covering no tensor is not a parameter object")
    payload: dict[str, bytes] = {}
    shapes: dict[str, tuple[int, int]] = {}
    for raw_name, array in tensors.items():
        name = str(raw_name)
        if not name:
            raise TaskVectorError("every tensor needs a non-empty name")
        if _FACTOR_SUFFIX.search(name):
            raise TaskVectorError(
                f"{name!r} names a raw LoRA factor; merges operate on induced updates"
                " ΔW = BA, because the factorisation is not unique [AUTH: 00 §22]"
            )
        try:
            candidate = np.asarray(array, dtype=context.arithmetic)
        except (TypeError, ValueError) as exc:
            raise TaskVectorError(
                f"{name}: cannot be read as an induced update matrix ({exc}); merges take"
                " ΔW = BA, not raw LoRA factors [AUTH: 00 §22]"
            ) from exc
        if candidate.ndim != 2:
            raise TaskVectorError(f"{name}: an induced update must be a 2-D array")
        if not np.all(np.isfinite(candidate)):
            raise TaskVectorError(
                f"{name}: the induced update carries NaN or Inf; a non-finite merge input or"
                " output is not a parameter object [AUTH: 00 §22]"
            )
        payload[name] = np.ascontiguousarray(candidate, dtype=context.byte_order_code).tobytes("C")
        shapes[name] = (int(candidate.shape[0]), int(candidate.shape[1]))
    return payload, shapes


def _issue(
    cls: type[TaskVector],
    tensors: Mapping[str, Matrix],
    *,
    context: MergeExecutionContext,
    origin: str,
    source: Mapping[str, JSONValue] | None = None,
) -> Any:
    payload, shapes = _canonical_bytes(tensors, context)
    vector = object.__new__(cls)
    object.__setattr__(vector, "_bytes", payload)
    object.__setattr__(vector, "_shapes", shapes)
    object.__setattr__(vector, "_context", context)
    object.__setattr__(vector, "_origin", origin)
    if source is not None:
        object.__setattr__(vector, "_source", dict(source))
    return vector


def fixture_induced_update(
    tensors: Mapping[str, Matrix],
    *,
    context: MergeExecutionContext,
    origin: str = "fixture",
) -> FixtureTaskVector:
    """THE fixture constructor. Its result is permanently non-evidentiary.

    This is the only place caller-supplied matrices become a task vector, and what it produces
    can never acquire verified standing.
    """
    issued: FixtureTaskVector = _issue(FixtureTaskVector, tensors, context=context, origin=origin)
    return issued


def verified_task_vector_from_adapter(
    *, adapter_manifest: JSONDocument, root: Path, context: MergeExecutionContext
) -> VerifiedTaskVector:
    """THE production constructor. It derives ΔW itself; the caller supplies no tensors.

    The whole chain runs here, on accepted Block C interfaces used exactly as they stand:

    1. `validate_adapter_manifest` — the manifest is structurally valid for its role;
    2. `verify_adapter_artifact` — the named artifact's bytes are re-read and re-hashed
       against the recorded adapter and induced-update hashes;
    3. `load_adapter` — the validated artifact becomes a `LoraAdapter`;
    4. `induced_updates()` — ΔW = scaling·B·A is derived from the factors, here, not handed in;
    5. the tensors are cast to the frozen artifact dtype and frozen into immutable bytes.

    The cast is an S07 arithmetic-policy step and is recorded: the evidence keeps S06's
    float64 `induced_update_sha256` as the source proof, while the vector's own content
    identity is over the float32 bytes the merge consumes [AUTH: 01 §10].
    """
    from src.provenance.hashing import read_canonical_json
    from src.training.manifests import (
        AdapterManifestError,
        validate_adapter_manifest,
        verify_adapter_artifact,
    )
    from src.training.trainer import load_adapter

    problems = validate_adapter_manifest(adapter_manifest)
    if problems:
        raise ProvenanceError(f"the adapter manifest is invalid: {'; '.join(problems)}")
    try:
        verify_adapter_artifact(adapter_manifest, root)
    except AdapterManifestError as exc:
        raise ProvenanceError(f"the adapter artifact does not match its manifest: {exc}") from exc

    relative = str(adapter_manifest["adapter_artifact_path"])
    stored = read_canonical_json(root / relative)
    if not isinstance(stored, Mapping):  # pragma: no cover - verify_adapter_artifact caught it
        raise ProvenanceError(f"{relative}: the adapter artifact is not a JSON object")
    adapter = load_adapter(stored)
    if adapter.update_identity() != adapter_manifest["induced_update_sha256"]:  # pragma: no cover
        raise ProvenanceError("the induced update does not match its recorded identity")

    evidence: dict[str, JSONValue] = {
        "adapter_alias": str(adapter_manifest["adapter_alias"]),
        "run_id": str(adapter_manifest["run_id"]),
        "manifest_role": str(adapter_manifest["manifest_role"]),
        "adapter_artifact_path": relative,
        "adapter_file_sha256": str(adapter_manifest["adapter_file_sha256"]),
        "induced_update_sha256": str(adapter_manifest["induced_update_sha256"]),
        "execution_contract_sha256": str(adapter_manifest["execution_contract_sha256"]),
        "corpus_role": str(adapter_manifest["corpus_role"]),
        "model_revision": str(adapter_manifest["model_revision"]),
        "source_precision": str(adapter_manifest["precision"]),
    }
    issued: VerifiedTaskVector = _issue(
        VerifiedTaskVector,
        adapter.induced_updates(),
        context=context,
        origin=str(adapter_manifest["adapter_alias"]),
        source=evidence,
    )
    return issued


# ----------------------------------------------------------------------------------------
# surface algebra
# ----------------------------------------------------------------------------------------


def require_compatible(vectors: Sequence[TaskVector], *, what: str = "merge") -> str:
    """Every participant must share one exactly compatible surface. Returns its identity.

    Exact, not "compatible enough": a name present on one side and absent on the other is a
    refusal, never a zero fill, and a shape difference is a refusal, never a broadcast.
    """
    if len(vectors) < 2:
        raise SurfaceMismatchError(f"a {what} needs at least two task vectors")
    reference = vectors[0]
    for index, other in enumerate(vectors[1:], start=1):
        missing = sorted(set(reference.names) - set(other.names))
        extra = sorted(set(other.names) - set(reference.names))
        if missing:
            raise SurfaceMismatchError(
                f"{what}: participant {index} is missing {len(missing)} tensor(s) present in"
                f" participant 0, first {missing[0]!r}; a missing target matrix is not zero"
            )
        if extra:
            raise SurfaceMismatchError(
                f"{what}: participant {index} carries {len(extra)} tensor(s) absent from"
                f" participant 0, first {extra[0]!r}; the surfaces are different objects"
            )
        for name in reference.names:
            if reference.shapes[name] != other.shapes[name]:
                raise SurfaceMismatchError(
                    f"{what}: {name} is {reference.shapes[name]} in participant 0 and"
                    f" {other.shapes[name]} in participant {index}; shapes are never"
                    " broadcast into agreement"
                )
        if reference.dtype != other.dtype:
            raise SurfaceMismatchError(f"{what}: participants disagree about dtype")
        if reference.context.identity() != other.context.identity():
            raise SurfaceMismatchError(
                f"{what}: participants were resolved under different execution contexts"
            )
    return reference.surface_identity()


def combine(
    contributions: Sequence[tuple[float, TaskVector]],
    *,
    context: MergeExecutionContext,
    origin: str,
) -> FixtureTaskVector:
    """Σ c_i · τ_i, in the context's arithmetic dtype. The one place operator arithmetic runs.

    The result is a merged descendant, not a constituent, so it is fixture-class: only a
    constituent read out of a validated S06 artifact is verified. A descendant's provenance
    lives on its `MergeResult`, which records both inputs' identities and classes.
    """
    if not contributions:
        raise TaskVectorError("no contributions to combine")
    vectors = [vector for _, vector in contributions]
    if len(vectors) > 1:
        require_compatible(vectors)
    for coefficient, _ in contributions:
        if not np.isfinite(coefficient):
            raise TaskVectorError(f"coefficient {coefficient!r} is not finite")
    dtype = context.arithmetic
    reference = vectors[0]
    merged: dict[str, Matrix] = {}
    for name in reference.names:
        total = np.zeros(reference.shapes[name], dtype=dtype)
        for coefficient, vector in contributions:
            total = total + np.asarray(coefficient, dtype=dtype) * vector[name]
        merged[name] = np.asarray(total, dtype=dtype)
    return fixture_induced_update(merged, context=context, origin=origin)
