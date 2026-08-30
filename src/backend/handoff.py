"""Backend adapter -> accepted S06 artifact -> S07 verified task vector [AUTH: 00 §22; 01 §16].

There is deliberately no "trusted backend task vector" path. The backend writes the artifact
and manifest the already-accepted S06 contract defines, and S07's
`verified_task_vector_from_adapter` then **independently** re-validates the manifest, re-reads
and re-hashes the artifact bytes, reloads the adapter and derives ΔW itself. Nothing the
backend asserts is taken on trust downstream; if the backend and S07 ever disagree about the
induced update, the handoff fails rather than resolving in the backend's favour.

That also means this module cannot promote anything: an evidentiary manifest still needs an
immutable model revision, a research corpus role, a real run id, a READY backend status and
resolved material constants, all checked by S06's own validator [AUTH: 01 §16].
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from src.backend.adapters import AdapterError, PersistedAdapter
from src.merge.context import MergeExecutionContext
from src.merge.updates import TaskVector
from src.provenance.hashing import JSONDocument, JSONValue, write_canonical_json

HANDOFF_VERSION: Final = "backend.s06-handoff.v1"


class HandoffError(ValueError):
    """A persisted backend adapter cannot be expressed as an accepted S06 artifact."""


def adapter_artifact_document(adapter: PersistedAdapter) -> dict[str, JSONValue]:
    """The accepted S06 adapter-artifact shape, built from the real PEFT factors.

    Factors are emitted in float64 because that is the dtype S06's `load_adapter` reads and
    hashes in; the float32 cast that the merge stage applies is S07's arithmetic-policy step
    and belongs there, not here [AUTH: 01 §10].
    """
    factors: dict[str, JSONValue] = {}
    for name in adapter.target_parameters:
        a, b = adapter.factors[name]
        factors[name] = {
            "A": np.asarray(a, dtype=np.float64).tolist(),
            "B": np.asarray(b, dtype=np.float64).tolist(),
        }

    from src.training.trainer import LoraAdapter

    reconstructed = LoraAdapter(
        factors={
            name: (
                np.asarray(adapter.factors[name][0], dtype=np.float64),
                np.asarray(adapter.factors[name][1], dtype=np.float64),
            )
            for name in adapter.target_parameters
        },
        rank=adapter.specification.rank,
        scaling=adapter.specification.scaling,
    )
    if reconstructed.update_identity() != adapter.update_identity():  # pragma: no cover
        raise HandoffError(
            "the S06 adapter object and the backend adapter disagree about the induced update;"
            " the scaling semantics differ between them"
        )
    return {
        "rank": adapter.specification.rank,
        "scaling": adapter.specification.scaling,
        "adapter_identity": reconstructed.identity(),
        "factors": factors,
    }


def write_adapter_artifact(
    root: Path, relative_path: str, adapter: PersistedAdapter
) -> tuple[str, str]:
    """Write the S06 artifact and return `(adapter_file_sha256, induced_update_sha256)`.

    Both hashes are computed from the document that was actually written, so a manifest built
    from them describes the bytes on disk rather than an intention.
    """
    from src.provenance.hashing import read_canonical_json
    from src.training.trainer import adapter_file_hash, load_adapter

    document = adapter_artifact_document(adapter)
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(path, document)

    stored = read_canonical_json(path)
    if not isinstance(stored, dict):  # pragma: no cover - write_canonical_json wrote an object
        raise HandoffError(f"{relative_path}: the written artifact is not a JSON object")
    reloaded = load_adapter(stored)
    if reloaded.update_identity() != adapter.update_identity():
        raise HandoffError(
            f"{relative_path}: the artifact reloads to a different induced update than the"
            " backend adapter produced"
        )
    return adapter_file_hash(stored), reloaded.update_identity()


def verified_task_vector_from_backend_adapter(
    *, adapter_manifest: JSONDocument, root: Path, context: MergeExecutionContext
) -> TaskVector:
    """Route the backend's artifact through S07 unchanged.

    A one-line delegation on purpose: the point of the handoff is that the backend has no
    privileged path into S07, so this adds no validation, no shortcut and no trust.
    """
    from src.merge.updates import verified_task_vector_from_adapter

    if not isinstance(context, MergeExecutionContext):
        raise HandoffError("the S07 handoff needs a factory-issued merge execution context")
    return verified_task_vector_from_adapter(
        adapter_manifest=adapter_manifest, root=root, context=context
    )


def check_backend_and_s07_agree(
    *,
    adapter: PersistedAdapter,
    adapter_manifest: JSONDocument,
    root: Path,
    context: MergeExecutionContext,
) -> dict[str, JSONValue]:
    """Derive ΔW twice — backend and S07 — and require the two to agree.

    S07 casts to the frozen float32 artifact dtype, so agreement is checked at that precision;
    the float64 induced-update identity is compared exactly.
    """
    vector = verified_task_vector_from_backend_adapter(
        adapter_manifest=adapter_manifest, root=root, context=context
    )
    backend_updates = adapter.induced_updates()
    names = tuple(sorted(backend_updates))
    vector_names = tuple(vector.names)
    if names != vector_names:
        raise HandoffError(
            f"the backend adapted {names} but S07 derived {vector_names}; the parameter"
            " surfaces disagree"
        )
    for name in names:
        expected = np.asarray(backend_updates[name], dtype=np.float32)
        actual = np.asarray(vector[name], dtype=np.float32)
        if not np.array_equal(expected, actual):
            raise HandoffError(
                f"{name}: the backend and S07 derived different induced updates; the LoRA"
                " scaling semantics disagree between them [AUTH: 00 §22]"
            )
    if str(adapter_manifest["induced_update_sha256"]) != adapter.update_identity():
        raise HandoffError(  # pragma: no cover - write_adapter_artifact already checked
            "the manifest's induced_update_sha256 is not the backend adapter's"
        )
    return {
        "handoff_version": HANDOFF_VERSION,
        "parameter_surface_sha256": vector.surface_identity(),
        "induced_update_sha256": adapter.update_identity(),
        "provenance_class": vector.provenance_class,
        "n_target_parameters": len(names),
    }


def require_no_parallel_trusted_path(root: Path) -> None:
    """No backend module may issue a verified task vector itself [AUTH: 00 §22; 01 §16].

    `_issue` and the `VerifiedTaskVector` class are S07's; a backend that imported either
    could mint a verified vector from tensors nothing revalidated.
    """
    import ast

    for path in sorted((root / "src" / "backend").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "src.merge.updates":
                names = {alias.name for alias in node.names}
                forbidden = names & {"_issue", "VerifiedTaskVector", "_canonical_bytes"}
                if forbidden:
                    raise AdapterError(
                        f"{path.name} imports {sorted(forbidden)} from S07; the backend must"
                        " go through verified_task_vector_from_adapter, not around it"
                    )
