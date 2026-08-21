"""S00-B environment capture, exercised against mocked H100 hardware.

The real pod is runtime-only: no nvcc, no pip. These tests prove capture works under exactly
those conditions, measures rather than assumes, and publishes transactionally.
"""

from __future__ import annotations

import ast
import json
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import capture_environment as cap
import numpy as np
import pytest
from preflight import TBD, environment_lock_sha256, validate_environment_manifest

# The exact distributions observed in the sealed image on the real H100.
OBSERVED_DISTRIBUTIONS: dict[str, str] = {
    "nvidia-cublas": "13.1.1.3",
    "nvidia-cuda-cupti": "13.0.85",
    "nvidia-cuda-nvrtc": "13.0.88",
    "nvidia-cuda-runtime": "13.0.96",
    "nvidia-cudnn-cu13": "9.20.0.48",
    "nvidia-cufft": "12.0.0.61",
    "nvidia-cufile": "1.15.1.6",
    "nvidia-curand": "10.4.0.35",
    "nvidia-cusolver": "12.0.4.66",
    "nvidia-cusparse": "12.6.3.3",
    "nvidia-cusparselt-cu13": "0.8.1",
    "nvidia-nccl-cu13": "2.29.7",
    "nvidia-nvjitlink": "13.3.33",
    "nvidia-nvshmem-cu13": "3.4.5",
    "nvidia-nvtx": "13.0.85",
    "accelerate": "1.14.0",
    "opacus": "1.6.0",
    "peft": "0.20.0",
    "tokenizers": "0.22.2",
    "torch": "2.13.0+cu130",
    "transformers": "5.15.1",
}


# ---------------------------------------------------------------- a minimal torch stand-in
def _to_bf16(array: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 precision, round-to-nearest-even on the low 16 bits."""
    bits = array.astype(np.float32).view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = (bits.astype(np.uint64) + 0x7FFF + lsb.astype(np.uint64)) & 0xFFFFFFFF
    return (rounded.astype(np.uint32) & 0xFFFF0000).view(np.float32)


class FakeTensor:
    def __init__(self, data: np.ndarray, dtype: str) -> None:
        self.data = data
        self.dtype = dtype

    def __matmul__(self, other: FakeTensor) -> FakeTensor:
        product = self.data.astype(np.float32) @ other.data.astype(np.float32)
        return FakeTensor(_to_bf16(product) if self.dtype == "bfloat16" else product, self.dtype)

    def __sub__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(self.data - other.data, "float32")

    def __truediv__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(self.data / other.data, "float32")

    def to(self, dtype: str) -> FakeTensor:
        return FakeTensor(_to_bf16(self.data) if dtype == "bfloat16" else self.data, dtype)

    def abs(self) -> FakeTensor:
        return FakeTensor(np.abs(self.data), self.dtype)

    def clamp_min(self, floor: float) -> FakeTensor:
        return FakeTensor(np.maximum(self.data, floor), self.dtype)

    def max(self) -> FakeTensor:
        return FakeTensor(np.array(self.data.max()), self.dtype)

    def mean(self) -> FakeTensor:
        return FakeTensor(np.array(self.data.mean()), self.dtype)

    def item(self) -> float:
        return float(self.data)


class _Cudnn:
    version_value = 92000
    enabled = True
    deterministic = False
    benchmark = False
    allow_tf32 = False

    def version(self) -> int:
        return self.version_value


class _Autocast:
    def __init__(self, **_: Any) -> None: ...
    def __enter__(self) -> _Autocast:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class FakeTorch:
    """Only the surface capture_environment actually touches."""

    float32 = "float32"
    bfloat16 = "bfloat16"
    __version__ = "2.13.0+cu130"

    def __init__(self, *, cuda_available: bool = True, bf16: bool = True) -> None:
        self._rng = np.random.default_rng(0)
        self.version = type("V", (), {"cuda": "13.0"})()
        self.cuda = type(
            "C",
            (),
            {
                "is_available": lambda _self: cuda_available,
                "is_bf16_supported": lambda _self: bf16,
                "manual_seed_all": lambda _self, seed: None,
            },
        )()
        cudnn = _Cudnn()
        self.backends = type(
            "B",
            (),
            {
                "cudnn": cudnn,
                "cuda": type("CB", (), {"matmul": type("M", (), {"allow_tf32": False})()})(),
            },
        )()

    def manual_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def randn(self, rows: int, cols: int, device: str = "", dtype: str = "float32") -> FakeTensor:
        return FakeTensor(self._rng.standard_normal((rows, cols)).astype(np.float32), dtype)

    def finfo(self, _dtype: str) -> Any:
        return np.finfo(np.float32)

    def autocast(self, **kwargs: Any) -> _Autocast:
        return _Autocast(**kwargs)

    def get_float32_matmul_precision(self) -> str:
        return "highest"

    def are_deterministic_algorithms_enabled(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _small_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measurement shape is a parameter; a small one keeps the unit lane fast."""
    monkeypatch.setattr(cap, "MEASUREMENT_SHAPE", (64, 64))


def _fake_distributions(mapping: dict[str, str]) -> Any:
    class Dist:
        def __init__(self, name: str, version: str) -> None:
            self.metadata = {"Name": name}
            self.version = version

    return lambda: [Dist(name, version) for name, version in mapping.items()]


def _executable_source(path: Path) -> str:
    """Module source with comments and docstrings removed.

    Prose explaining why nvcc and pip are *not* used must not be mistaken for using them.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                body.pop(0)
    return ast.unparse(tree)


# ============================================================== A, B — CUDA runtime, no nvcc
def test_a_runtime_only_image_resolves_cuda_runtime_without_nvcc() -> None:
    def version_of(name: str) -> str:
        if name == "nvidia-cuda-runtime":
            return "13.0.96"
        raise importlib_metadata.PackageNotFoundError(name)

    assert cap.cuda_runtime_version(version_of) == ("nvidia-cuda-runtime", "13.0.96")


def test_a_capture_never_invokes_nvcc(repo_root: Path) -> None:
    code = _executable_source(repo_root / "scripts" / "capture_environment.py")
    assert "nvcc" not in code, "nvcc belongs to the dev toolkit and is absent by design"


@pytest.mark.parametrize("name", cap.CUDA_RUNTIME_DISTRIBUTIONS)
def test_a_any_known_runtime_distribution_name_resolves(name: str) -> None:
    def version_of(candidate: str) -> str:
        if candidate == name:
            return "13.0.96"
        raise importlib_metadata.PackageNotFoundError(candidate)

    assert cap.cuda_runtime_version(version_of) == (name, "13.0.96")


def test_b_missing_cuda_runtime_distribution_fails_closed() -> None:
    def version_of(name: str) -> str:
        raise importlib_metadata.PackageNotFoundError(name)

    with pytest.raises(cap.CaptureError, match="no CUDA runtime distribution"):
        cap.cuda_runtime_version(version_of)


# ============================================================== C, D, E — dependency capture
def test_c_dependency_capture_needs_no_pip(repo_root: Path) -> None:
    snapshot = cap.dependency_snapshot(_fake_distributions(OBSERVED_DISTRIBUTIONS))
    assert snapshot
    code = _executable_source(repo_root / "scripts" / "capture_environment.py")
    assert "pip" not in code, "pip is absent from the image by design"
    assert "importlib_metadata" in code


def test_d_dependency_snapshot_is_deterministically_ordered() -> None:
    shuffled = dict(reversed(list(OBSERVED_DISTRIBUTIONS.items())))
    first = cap.dependency_snapshot(_fake_distributions(OBSERVED_DISTRIBUTIONS))
    second = cap.dependency_snapshot(_fake_distributions(shuffled))
    assert first == second == sorted(first)


def test_e_snapshot_includes_science_and_cuda_distributions() -> None:
    snapshot = cap.dependency_snapshot(_fake_distributions(OBSERVED_DISTRIBUTIONS))
    assert "torch==2.13.0+cu130" in snapshot
    assert "transformers==5.15.1" in snapshot
    assert "opacus==1.6.0" in snapshot
    assert "nvidia-cuda-runtime==13.0.96" in snapshot
    assert sum(1 for entry in snapshot if entry.startswith("nvidia-")) == 15


def test_e_duplicate_distribution_fails_closed() -> None:
    class Dist:
        def __init__(self, name: str, version: str) -> None:
            self.metadata = {"Name": name}
            self.version = version

    def distributions() -> list[Dist]:
        return [Dist("torch", "2.13.0+cu130"), Dist("torch", "2.4.1+cu124")]

    with pytest.raises(cap.CaptureError, match="installed twice"):
        cap.dependency_snapshot(distributions)


def test_c_empty_environment_fails_closed() -> None:
    with pytest.raises(cap.CaptureError, match="no installed distributions"):
        cap.dependency_snapshot(lambda: [])


# ============================================================== F, G — BF16 tolerance
def test_f_bf16_tolerance_is_measured_not_tbd() -> None:
    measured = cap.measure_bf16_fp32_tolerance(FakeTorch())
    assert TBD not in {str(value) for value in measured.values()}
    assert measured["measurement"] == "bf16_vs_fp32_matmul"
    assert measured["autocast_enabled"] is False
    assert measured["reference_dtype"] == "float32"
    assert measured["compared_dtype"] == "bfloat16"
    errors: dict[str, float] = {}
    for key in ("max_abs_error", "mean_abs_error", "max_rel_error"):
        value = measured[key]
        assert isinstance(value, float) and value >= 0.0, key
        errors[key] = value
    assert errors["max_abs_error"] > 0.0, "a real BF16 cast must lose precision"
    assert errors["mean_abs_error"] <= errors["max_abs_error"]


def test_f_measurement_is_reproducible_from_the_fixed_seed() -> None:
    first = cap.measure_bf16_fp32_tolerance(FakeTorch())
    second = cap.measure_bf16_fp32_tolerance(FakeTorch())
    assert first["max_abs_error"] == second["max_abs_error"]
    assert first["seed"] == cap.MEASUREMENT_SEED


def test_g_bf16_unsupported_fails_closed() -> None:
    with pytest.raises(cap.CaptureError, match="BF16 is unsupported"):
        cap.measure_bf16_fp32_tolerance(FakeTorch(bf16=False))


def test_g_no_cuda_fails_closed() -> None:
    with pytest.raises(cap.CaptureError, match="CUDA is unavailable"):
        cap.measure_bf16_fp32_tolerance(FakeTorch(cuda_available=False))


def test_g_tolerance_is_never_caller_supplied(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    monkeypatch.setenv("BF16_FP32_TOLERANCE", "0.0")
    measured = cap.measure_bf16_fp32_tolerance(FakeTorch())
    max_abs = measured["max_abs_error"]
    assert isinstance(max_abs, float) and max_abs > 0.0
    code = _executable_source(repo_root / "scripts" / "capture_environment.py")
    assert "BF16_FP32_TOLERANCE" not in code


# ============================================================== H, I — nondeterminism
def test_h_nondeterminism_sources_are_captured_not_tbd() -> None:
    sources = cap.capture_nondeterminism_sources(FakeTorch())
    assert TBD not in {str(value) for value in sources.values()}
    assert sources["cudnn_version"] == 92000
    assert sources["cudnn_deterministic"] is False
    assert sources["cudnn_benchmark"] is False
    assert sources["cuda_matmul_allow_tf32"] is False
    assert sources["float32_matmul_precision"] == "highest"
    assert sources["cublas_workspace_config"] is None
    assert sources["deterministic_algorithms"] is False
    run = sources["run_to_run"]
    assert isinstance(run, dict)
    assert run["repeats"] == cap.RUN_TO_RUN_REPEATS
    assert run["float32_max_abs_delta"] == 0.0
    assert run["deterministic_observed"] is True


def test_h_no_cuda_fails_closed() -> None:
    with pytest.raises(cap.CaptureError, match="CUDA is unavailable"):
        cap.capture_nondeterminism_sources(FakeTorch(cuda_available=False))


def test_i_nondeterminism_serialisation_is_deterministic() -> None:
    first = cap.capture_nondeterminism_sources(FakeTorch())
    second = cap.capture_nondeterminism_sources(FakeTorch())
    assert cap.serialise_nondeterminism(first) == cap.serialise_nondeterminism(second)
    reordered = dict(reversed(list(first.items())))
    assert cap.serialise_nondeterminism(reordered) == cap.serialise_nondeterminism(first)


def test_i_serialisation_is_structured_not_prose() -> None:
    payload = cap.serialise_nondeterminism(cap.capture_nondeterminism_sources(FakeTorch()))
    assert json.loads(payload)["cudnn_version"] == 92000


# ============================================================== J..N — transactional publish
def _complete_manifest() -> dict[str, object]:
    manifest: dict[str, object] = {
        "authority": "01 §12(2)-(9), §15, §16, §30, §32",
        "uv_lock_sha256": "b" * 64,
        "python_version": "3.13.15",
        "torch_version": "2.13.0+cu130",
        "torch_cuda_build": "13.0",
        "cuda_runtime": "13.0.96",
        "cuda_runtime_distribution": "nvidia-cuda-runtime",
        "dependency_versions": cap.dependency_snapshot(_fake_distributions(OBSERVED_DISTRIBUTIONS)),
        "bf16_fp32_tolerance": cap.measure_bf16_fp32_tolerance(FakeTorch()),
        "nondeterminism_sources": cap.capture_nondeterminism_sources(FakeTorch()),
        "capture_timestamp_utc": "2026-08-21T00:00:00+00:00",
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "gpu_uuid": "GPU-fixture",
        "gpu_count": "1",
        "cuda_driver": "580.126.09",
        "nvidia_smi_capture": "fixture",
        "docker_image_tag": "registry/repo",
        "docker_image_digest": "sha256:" + "c" * 64,
        "base_image_digest": "sha256:" + "d" * 64,
        "image_lane": "science",
        "image_source_git_commit": "e" * 40,
        "image_identity_source": "BAKED_INTO_IMAGE",
    }
    return manifest


def test_l_successful_capture_publishes_exactly_one_manifest(tmp_path: Path) -> None:
    published = cap.publish(tmp_path, _complete_manifest(), out_dir=tmp_path / "env")
    files = sorted((tmp_path / "env").glob("*.json"))
    assert files == [published]
    assert len(published.stem) == 64 and published.stem == published.stem.lower()
    int(published.stem, 16)
    assert not list((tmp_path / "env").glob("*.tmp"))


def test_m_environment_identity_recomputes_identically(tmp_path: Path) -> None:
    published = cap.publish(tmp_path, _complete_manifest(), out_dir=tmp_path / "env")
    stored = json.loads(published.read_text(encoding="utf-8"))
    assert stored["environment_lock_sha256"] == published.stem
    assert environment_lock_sha256(stored) == stored["environment_lock_sha256"]
    assert validate_environment_manifest(stored) == []


@pytest.mark.parametrize(
    "field", ["cuda_runtime", "cuda_driver", "torch_version", "gpu_model", "uv_lock_sha256"]
)
def test_j_failed_capture_writes_no_tbd_manifest(tmp_path: Path, field: str) -> None:
    manifest = _complete_manifest()
    manifest[field] = TBD
    out = tmp_path / "env"
    with pytest.raises(cap.CaptureError):
        cap.publish(tmp_path, manifest, out_dir=out)
    assert not (out / f"{TBD}.json").exists()


@pytest.mark.parametrize("field", ["cuda_runtime", "bf16_fp32_tolerance", "gpu_uuid"])
def test_k_failed_capture_writes_no_canonical_manifest(tmp_path: Path, field: str) -> None:
    manifest = _complete_manifest()
    manifest.pop(field)
    out = tmp_path / "env"
    with pytest.raises(cap.CaptureError):
        cap.publish(tmp_path, manifest, out_dir=out)
    assert not out.exists() or list(out.glob("*.json")) == []


def test_j_stale_tbd_manifest_is_removed_on_success(tmp_path: Path) -> None:
    """The earlier non-transactional implementation left this behind on the real pod."""
    out = tmp_path / "env"
    out.mkdir()
    stale = out / f"{TBD}.json"
    stale.write_text("{}", encoding="utf-8")
    cap.publish(tmp_path, _complete_manifest(), out_dir=out)
    assert not stale.exists()
    assert len(list(out.glob("*.json"))) == 1


def test_k_no_temporary_file_survives_a_failure(tmp_path: Path) -> None:
    manifest = _complete_manifest()
    manifest["gpu_model"] = TBD
    out = tmp_path / "env"
    out.mkdir()
    with pytest.raises(cap.CaptureError):
        cap.publish(tmp_path, manifest, out_dir=out)
    assert list(out.iterdir()) == []


# ============================================================== N — anti-forgery preserved
def test_n_baked_identity_is_read_from_inside_the_image(tmp_path: Path) -> None:
    baked = tmp_path / "pmm-image.json"
    baked.write_text(
        json.dumps(
            {
                "image_ref": "registry/repo",
                "image_digest": "sha256:" + "a" * 64,
                "base_image_digest": "sha256:" + "b" * 64,
                "lane": "science",
                "source_git_commit": "c" * 40,
            }
        ),
        encoding="utf-8",
    )
    identity = cap.capture_image_identity(baked)
    assert identity["docker_image_digest"] == "sha256:" + "a" * 64
    assert identity["image_identity_source"] == "BAKED_INTO_IMAGE"
    assert identity["image_lane"] == "science"


def test_n_caller_supplied_digest_is_never_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCKER_IMAGE_DIGEST", "sha256:" + "f" * 64)
    identity = cap.capture_image_identity(tmp_path / "absent.json")
    assert identity["docker_image_digest"] == TBD
    assert identity["image_identity_source"] == "CALLER_SUPPLIED_UNVERIFIED"


def test_n_malformed_baked_digest_fails_closed(tmp_path: Path) -> None:
    baked = tmp_path / "pmm-image.json"
    baked.write_text(json.dumps({"image_digest": "not-a-digest"}), encoding="utf-8")
    with pytest.raises(cap.CaptureError, match="malformed"):
        cap.capture_image_identity(baked)


def test_n_forged_identity_cannot_produce_a_manifest(tmp_path: Path) -> None:
    manifest = _complete_manifest()
    manifest["docker_image_digest"] = TBD
    with pytest.raises(cap.CaptureError):
        cap.publish(tmp_path, manifest, out_dir=tmp_path / "env")
    assert not (tmp_path / "env").exists() or list((tmp_path / "env").glob("*.json")) == []
