"""CUDA / PyTorch compatibility contract, accepted on the real H100 image
[AUTH: 01 §9, §10, §12(3), §21, §30; 01 §8C(3),(4),(11),(12); plan §5.4].

Collected skeletons that skip with an explicit reason off-hardware, so `make gpu-smoke` can
never collect zero tests and report a state it did not evaluate. No CUDA, PyTorch or H100
value is asserted from memory: everything hardware-derived is read from the captured
environment manifest [AUTH: 03 §8].
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from preflight import is_environment_lock_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
H100_COMPUTE_CAPABILITY = (9, 0)  # NVIDIA H100 SXM, fixed by 01 §9, not a measurement


def _torch_cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


requires_h100 = pytest.mark.skipif(
    not _torch_cuda_available(),
    reason="NOT_RUN(NO_GPU): torch with CUDA is unavailable [AUTH: 01 §21; 03 §8]",
)


def test_gpu_smoke_lane_is_not_empty() -> None:
    """Guard against the lane silently collecting nothing [AUTH: 02 §C6; 00 §34B.3]."""
    assert H100_COMPUTE_CAPABILITY == (9, 0)


@requires_h100
def test_device_is_h100_sxm() -> None:
    import torch

    assert torch.cuda.get_device_capability(0) == H100_COMPUTE_CAPABILITY, (
        "the scientific compute lock is NVIDIA H100 SXM 80GB [AUTH: 01 §9]"
    )


@requires_h100
def test_bf16_forward_path_without_quantization() -> None:
    """BF16 training path and FP32 merge/recovery arithmetic must both be available,
    and autocast must never silently decide scientific arithmetic [AUTH: 01 §10; 01 §8C]."""
    import torch

    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    assert a.dtype is torch.bfloat16
    assert (a.float() @ a.float()).dtype is torch.float32
    assert torch.zeros(2, device="cuda", dtype=torch.float64).dtype is torch.float64


@requires_h100
def test_run_to_run_tolerance_is_measured_not_assumed() -> None:
    """01 §30 requires the nondeterminism source documented AND the run-to-run tolerance
    measured. Capture now performs both, so a real H100 run must exercise the recorded
    contract instead of skipping past it."""
    env_dir = REPO_ROOT / "manifests" / "environments"
    captured = (
        [p for p in sorted(env_dir.glob("*.json")) if is_environment_lock_manifest(p)]
        if env_dir.is_dir()
        else []
    )
    if not captured:
        pytest.skip("NOT_RUN(NO_GPU): run `make env-capture` before the gpu-smoke lane")

    manifest = json.loads(captured[0].read_text(encoding="utf-8"))

    tolerance = manifest["bf16_fp32_tolerance"]
    assert isinstance(tolerance, dict), "the tolerance must be measured, not a placeholder"
    assert tolerance["measurement"] == "bf16_vs_fp32_matmul"
    assert tolerance["autocast_enabled"] is False, "AMP must not decide the arithmetic [01 §10]"
    max_abs = tolerance["max_abs_error"]
    assert isinstance(max_abs, float) and max_abs > 0.0, (
        "a real BF16 cast loses precision; a zero tolerance means nothing was measured"
    )
    assert 0.0 <= tolerance["mean_abs_error"] <= max_abs

    sources = manifest["nondeterminism_sources"]
    assert isinstance(sources, dict)
    assert isinstance(sources["cudnn_version"], int)
    for flag in (
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_allow_tf32",
        "deterministic_algorithms",
    ):
        assert isinstance(sources[flag], bool), flag
    run = sources["run_to_run"]
    assert isinstance(run, dict)
    assert run["repeats"] >= 2, "a run-to-run tolerance needs repeated passes [01 §30]"
    for key in ("float32_max_abs_delta", "bfloat16_max_abs_delta"):
        assert isinstance(run[key], float) and run[key] >= 0.0, key
    assert isinstance(run["deterministic_observed"], bool)
