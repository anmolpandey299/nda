#!/usr/bin/env python3
"""S00-B environment capture — 01 §12 steps 2-9, run on the real H100 image.

Writes `manifests/environments/<ENVIRONMENT_LOCK_SHA256>.json` and nothing else. Publication
is transactional: a failed capture leaves no manifest at all, canonical or otherwise.

Authority for the two measurements this performs:

* `bf16_fp32_tolerance` — 01 §10 fixes BF16 for training and FP32 where rounding matters, and
  forbids an AMP/autocast default from silently determining scientific arithmetic; 01 §8C(11)
  requires the scorer to be deterministic within a frozen numerical tolerance. Plan §5.7
  defines the value as the BF16 vs FP32 discrepancy measured on the image over fixed inputs.
  We therefore MEASURE the discrepancy; we do not choose a threshold.
* `nondeterminism_sources` — 01 §30: "Where CUDA kernels are nondeterministic, document the
  exact source and measure run-to-run tolerance." Plan §5.7 measures it with repeated
  identical passes. We therefore capture the determinism-relevant runtime state AND the
  measured run-to-run delta.

The environment is intentionally runtime-only: there is no nvcc and no pip, by design. The
CUDA runtime is resolved from the frozen installed distribution metadata and the dependency
snapshot comes from the standard library, so capture needs no toolkit, no package manager,
no network and installs nothing.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflight import (  # noqa: E402
    TBD,
    environment_lock_sha256,
    validate_environment_manifest,
)

#: Distribution names that ship the CUDA runtime, newest naming first [AUTH: 01 §12].
CUDA_RUNTIME_DISTRIBUTIONS: tuple[str, ...] = (
    "nvidia-cuda-runtime",
    "nvidia-cuda-runtime-cu13",
    "nvidia-cuda-runtime-cu12",
)

#: Fixed measurement parameters. Deterministic inputs, so the measurement is reproducible.
MEASUREMENT_SEED = 20260821
MEASUREMENT_SHAPE: tuple[int, int] = (1024, 1024)
RUN_TO_RUN_REPEATS = 5

BAKED_IMAGE_METADATA = Path("/etc/pmm-image.json")
HOST_PREFIXES: tuple[str, ...] = ("/opt/homebrew", "/Users", "/usr/local/Cellar")


class CaptureError(RuntimeError):
    """Any condition that makes the environment unresolvable. Always fails closed."""


# --------------------------------------------------------------------------------------
# 1. CUDA runtime, without nvcc
# --------------------------------------------------------------------------------------


def cuda_runtime_version(
    version_of: Callable[[str], str] = importlib_metadata.version,
) -> tuple[str, str]:
    """Resolve the CUDA runtime from installed distribution metadata.

    The science image is runtime-only. `nvcc` belongs to the CUDA *development* toolkit and
    is deliberately absent, so its absence must never make the runtime unresolvable. The
    runtime that torch actually loads is the pinned `nvidia-cuda-runtime` wheel in uv.lock,
    which is the authoritative frozen value [AUTH: 01 §12].

    Returns (distribution_name, version).
    """
    for name in CUDA_RUNTIME_DISTRIBUTIONS:
        try:
            return name, version_of(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    raise CaptureError(
        "no CUDA runtime distribution installed; expected one of "
        + ", ".join(CUDA_RUNTIME_DISTRIBUTIONS)
    )


# --------------------------------------------------------------------------------------
# 2. dependency snapshot, without pip
# --------------------------------------------------------------------------------------


def _normalise(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(".", "-")


def dependency_snapshot(
    distributions: Callable[[], Iterable[Any]] = importlib_metadata.distributions,
) -> list[str]:
    """Every installed distribution as a deterministically sorted `name==version` list.

    `pip` is absent from the image by design, so `pip freeze` cannot be the mechanism. The
    standard library's importlib.metadata reads the same installed metadata with no package
    manager, no network and no installation [AUTH: 01 §12].
    """
    seen: dict[str, str] = {}
    for dist in distributions():
        raw_name = dist.metadata["Name"] if dist.metadata else None
        if not raw_name:
            continue
        name = _normalise(str(raw_name))
        version = str(dist.version)
        # A duplicated distribution is a broken environment, not something to average over.
        if name in seen and seen[name] != version:
            raise CaptureError(f"distribution {name!r} installed twice: {seen[name]}, {version}")
        seen[name] = version
    if not seen:
        raise CaptureError("no installed distributions found; the environment is not usable")
    return [f"{name}=={seen[name]}" for name in sorted(seen)]


# --------------------------------------------------------------------------------------
# 3. BF16 vs FP32 tolerance — measured, never chosen
# --------------------------------------------------------------------------------------


def measure_bf16_fp32_tolerance(torch: Any) -> dict[str, object]:
    """Measure the BF16 vs FP32 discrepancy of a fixed deterministic matmul on the GPU.

    Matmul is the dominant arithmetic of a forward scoring pass, so its BF16-vs-FP32 gap is
    the quantity 01 §10 and 01 §8C(11) constrain. The FP32 result is the reference; the BF16
    path is the compared value. Autocast is explicitly disabled so no AMP default can silently
    determine the arithmetic [AUTH: 01 §10].
    """
    if not torch.cuda.is_available():
        raise CaptureError("CUDA is unavailable; the BF16/FP32 tolerance cannot be measured")
    if not torch.cuda.is_bf16_supported():
        raise CaptureError("BF16 is unsupported on this device; capture fails closed")

    device = "cuda:0"
    torch.manual_seed(MEASUREMENT_SEED)
    torch.cuda.manual_seed_all(MEASUREMENT_SEED)
    rows, cols = MEASUREMENT_SHAPE
    a32 = torch.randn(rows, cols, device=device, dtype=torch.float32)
    b32 = torch.randn(cols, rows, device=device, dtype=torch.float32)

    with torch.autocast(device_type="cuda", enabled=False):
        reference = a32 @ b32
        compared = (a32.to(torch.bfloat16) @ b32.to(torch.bfloat16)).to(torch.float32)

    delta = (compared - reference).abs()
    denominator = reference.abs().clamp_min(torch.finfo(torch.float32).tiny)
    relative = delta / denominator
    return {
        "measurement": "bf16_vs_fp32_matmul",
        "authority": "01 §10; 01 §8C(11); plan §5.7",
        "device": device,
        "seed": MEASUREMENT_SEED,
        "shape": list(MEASUREMENT_SHAPE),
        "reference_dtype": "float32",
        "compared_dtype": "bfloat16",
        "autocast_enabled": False,
        "max_abs_error": float(delta.max().item()),
        "mean_abs_error": float(delta.mean().item()),
        "max_rel_error": float(relative.max().item()),
        "reference_max_abs": float(reference.abs().max().item()),
    }


# --------------------------------------------------------------------------------------
# 4. nondeterminism sources — state plus measured run-to-run tolerance
# --------------------------------------------------------------------------------------


def measure_run_to_run_delta(torch: Any, dtype_name: str, repeats: int) -> float:
    """Largest absolute difference across repeated identical passes [AUTH: 01 §30]."""
    device = "cuda:0"
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(MEASUREMENT_SEED)
    torch.cuda.manual_seed_all(MEASUREMENT_SEED)
    rows, cols = MEASUREMENT_SHAPE
    a = torch.randn(rows, cols, device=device, dtype=torch.float32).to(dtype)
    b = torch.randn(cols, rows, device=device, dtype=torch.float32).to(dtype)
    first = (a @ b).to(torch.float32)
    worst = 0.0
    for _ in range(repeats - 1):
        again = (a @ b).to(torch.float32)
        worst = max(worst, float((again - first).abs().max().item()))
    return worst


def capture_nondeterminism_sources(torch: Any) -> dict[str, object]:
    """Structured determinism state plus the measured run-to-run tolerance [AUTH: 01 §30].

    Recorded as structured fields, never as prose, so a change in any of them is mechanically
    visible in the diff of the environment manifest.
    """
    if not torch.cuda.is_available():
        raise CaptureError("CUDA is unavailable; nondeterminism sources cannot be measured")

    cudnn = torch.backends.cudnn
    sources: dict[str, object] = {
        "authority": "01 §30; plan §5.4, §5.7",
        "cudnn_version": cudnn.version(),
        "cudnn_enabled": bool(cudnn.enabled),
        "cudnn_deterministic": bool(cudnn.deterministic),
        "cudnn_benchmark": bool(cudnn.benchmark),
        "cudnn_allow_tf32": bool(cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "run_to_run": {
            "repeats": RUN_TO_RUN_REPEATS,
            "seed": MEASUREMENT_SEED,
            "shape": list(MEASUREMENT_SHAPE),
            "float32_max_abs_delta": measure_run_to_run_delta(torch, "float32", RUN_TO_RUN_REPEATS),
            "bfloat16_max_abs_delta": measure_run_to_run_delta(
                torch, "bfloat16", RUN_TO_RUN_REPEATS
            ),
        },
    }
    run = sources["run_to_run"]
    assert isinstance(run, dict)
    run["deterministic_observed"] = bool(
        run["float32_max_abs_delta"] == 0.0 and run["bfloat16_max_abs_delta"] == 0.0
    )
    return sources


def serialise_nondeterminism(sources: dict[str, object]) -> str:
    """Deterministic serialisation, so identical state hashes identically."""
    return json.dumps(sources, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------------------
# 5. hardware and image identity
# --------------------------------------------------------------------------------------


def _run(command: Sequence[str]) -> str:
    result = subprocess.run(list(command), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise CaptureError(f"{' '.join(command)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def capture_gpu(runner: Callable[[Sequence[str]], str] = _run) -> dict[str, object]:
    if shutil.which("nvidia-smi") is None:
        raise CaptureError("nvidia-smi is absent; this is not a GPU pod")
    query = ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]
    rows = [line.strip() for line in runner(query).splitlines() if line.strip()]
    if not rows:
        raise CaptureError("nvidia-smi reported no GPU")
    name, uuid, driver = (part.strip() for part in rows[0].split(","))
    return {
        "gpu_model": name,
        "gpu_uuid": uuid,
        "gpu_count": str(len(rows)),
        "cuda_driver": driver,
        "nvidia_smi_capture": runner(["nvidia-smi"]),
    }


def capture_image_identity(baked: Path = BAKED_IMAGE_METADATA) -> dict[str, object]:
    """Image identity is read from inside the image, never taken from the caller.

    A caller-supplied digest is recorded as unverified and forced to TBD, so it can never
    become the environment's identity [AUTH: 01 §12(8)(9), §16; 03 §8].
    """
    if baked.is_file():
        data = json.loads(baked.read_text(encoding="utf-8"))
        digest = str(data.get("image_digest", TBD))
        if not digest.startswith("sha256:"):
            raise CaptureError(f"baked image digest is malformed: {digest!r}")
        return {
            "docker_image_tag": str(data.get("image_ref", TBD)),
            "docker_image_digest": digest,
            "base_image_digest": str(data.get("base_image_digest", TBD)),
            "image_lane": str(data.get("lane", TBD)),
            "image_source_git_commit": str(data.get("source_git_commit", TBD)),
            "image_identity_source": "BAKED_INTO_IMAGE",
        }
    supplied = os.environ.get("DOCKER_IMAGE_DIGEST")
    source = "CALLER_SUPPLIED_UNVERIFIED" if supplied else "NONE"
    if supplied:
        print(
            "capture: image digest was caller-supplied and cannot be verified from inside the"
            " image; recording it as unresolved [AUTH: 01 §12(9); 03 §8]",
            file=sys.stderr,
        )
    return {
        "docker_image_tag": os.environ.get("DOCKER_IMAGE_TAG", TBD),
        "docker_image_digest": TBD,
        "base_image_digest": TBD,
        "image_lane": TBD,
        "image_source_git_commit": TBD,
        "image_identity_source": source,
    }


# --------------------------------------------------------------------------------------
# 6. build, validate, publish transactionally
# --------------------------------------------------------------------------------------


IMAGE_RECORD_REL = "manifests/environments/S00B_IMAGE_RECORD.json"


def image_record_from_environment(manifest: dict[str, object]) -> dict[str, object]:
    """Derive the image record from the identity baked into the running image.

    The build host must not write this into tracked source: a completed record for image A
    would be baked into the next build commit B, the image built from B would conflict with
    it, and escaping would need another record, commit and rebuild - endlessly. Materialising
    it here, from the image that is actually running, breaks that cycle while keeping the
    identity unforgeable, because it comes from /etc/pmm-image.json by way of capture
    [AUTH: 01 §12(8)(9), §16].
    """
    for field in ("docker_image_digest", "image_source_git_commit"):
        value = manifest.get(field)
        if not isinstance(value, str) or value == TBD or not value:
            raise CaptureError(f"cannot materialise the image record: {field} is {value!r}")
    return {
        "authority": "01 §12(8)(9), §16; plan §5.6",
        "lane": manifest.get("image_lane", TBD),
        "image_ref": manifest.get("docker_image_tag", TBD),
        "sealed_image_digest": manifest["docker_image_digest"],
        "source_git_commit": manifest["image_source_git_commit"],
        "base_image_digest": manifest.get("base_image_digest", TBD),
        "environment_lock_sha256": manifest.get("environment_lock_sha256", TBD),
        "identity_source": manifest.get("image_identity_source", TBD),
        "note": (
            "Materialised on the pod from the identity baked into the running image. Commit "
            "it with the other closure evidence; never carry it into a later build commit."
        ),
    }


def write_image_record(root: Path, manifest: dict[str, object]) -> Path:
    """Atomically materialise the canonical image record for the running image."""
    record = image_record_from_environment(manifest)
    target = root / IMAGE_RECORD_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def build_manifest(root: Path, torch: Any) -> dict[str, object]:
    lock = root / "uv.lock"
    if not lock.is_file():
        raise CaptureError("uv.lock is absent; there is no frozen environment to capture")
    runtime_dist, runtime_version = cuda_runtime_version()
    manifest: dict[str, object] = {
        "authority": "01 §12(2)-(9), §15, §16, §30, §32",
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda),
        "cuda_runtime": runtime_version,
        "cuda_runtime_distribution": runtime_dist,
        "dependency_versions": dependency_snapshot(),
        "bf16_fp32_tolerance": measure_bf16_fp32_tolerance(torch),
        "nondeterminism_sources": capture_nondeterminism_sources(torch),
        "capture_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    manifest.update(capture_gpu())
    manifest.update(capture_image_identity())
    return manifest


def finalise(manifest: dict[str, object]) -> dict[str, object]:
    """Validate, then let the identity fall out of the completed manifest.

    `environment_lock_sha256` is never special-cased: it resolves only once every component
    is present, and the stored value must equal the value recomputed from the manifest
    [AUTH: 01 §12(5)-(9), §15; plan §5.6].
    """
    missing = validate_environment_manifest(manifest | {"environment_lock_sha256": ""})
    if missing:
        raise CaptureError(f"manifest is schema-incomplete: {', '.join(sorted(missing))}")
    unresolved = sorted(k for k, v in manifest.items() if v == TBD)
    if unresolved:
        raise CaptureError(f"unresolved: {', '.join(unresolved)}")

    identity = environment_lock_sha256(manifest)
    if identity == TBD or len(identity) != 64 or identity != identity.lower():
        raise CaptureError("environment identity did not resolve to 64 lowercase hex")
    finalised = dict(manifest)
    finalised["environment_lock_sha256"] = identity
    if environment_lock_sha256(finalised) != identity:
        raise CaptureError("recomputed environment identity does not match the stored value")
    if validate_environment_manifest(finalised):
        raise CaptureError("finalised manifest failed schema validation")
    return finalised


def publish(root: Path, manifest: dict[str, object], out_dir: Path | None = None) -> Path:
    """Atomically publish exactly one `<64hex>.json`. Nothing is written before validation."""
    finalised = finalise(manifest)
    identity = str(finalised["environment_lock_sha256"])
    directory = out_dir or (root / "manifests" / "environments")
    directory.mkdir(parents=True, exist_ok=True)

    # A previous non-transactional failure could have left this behind; it is never valid.
    stale = directory / f"{TBD}.json"
    if stale.exists():
        stale.unlink()

    payload = json.dumps(finalised, indent=2, sort_keys=True) + "\n"
    handle, temporary = tempfile.mkstemp(dir=str(directory), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, directory / f"{identity}.json")
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    # The image record describes the image this capture ran inside, so it is materialised
    # here rather than carried in from a previous build [C -> B -> H model].
    write_image_record(root, finalised)
    return directory / f"{identity}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=None, help="override the output directory")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve() if args.out else None

    try:
        import torch
    except ImportError as exc:
        print(f"S00_ENV_CAPTURE = {TBD}", file=sys.stderr)
        print(f"torch is not importable: {exc}", file=sys.stderr)
        return 1

    try:
        published = publish(root, build_manifest(root, torch), out_dir)
    except CaptureError as exc:
        print(f"S00_ENV_CAPTURE = {TBD}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(
            "resolve on the RunPod H100 SXM image per 01 §12 steps 2-9, then rerun",
            file=sys.stderr,
        )
        return 1
    print(f"wrote {published}")
    print("S00_ENV_CAPTURE = COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
