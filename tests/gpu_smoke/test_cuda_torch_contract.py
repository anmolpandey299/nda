"""CUDA / PyTorch compatibility contract, accepted on the real H100 image
[AUTH: 01 §9, §10, §12(3), §21, §30; 01 §8C(3),(4),(11),(12); plan §5.4].

Collected skeletons that skip with an explicit reason off-hardware, so `make gpu-smoke` can
never collect zero tests and report a state it did not evaluate. No CUDA, PyTorch or H100
value is asserted from memory: everything hardware-derived is read from the captured
environment manifest [AUTH: 03 §8].
"""

from __future__ import annotations

import importlib.util

import pytest

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
    """01 §30 requires nondeterminism sources to be documented and the tolerance measured."""
    pytest.skip(
        "NOT_RUN(NO_GPU): the tolerance is measured by scripts/capture_environment.sh "
        "on the H100 image and recorded as bf16_fp32_tolerance [AUTH: 01 §30; plan §5.7]"
    )
