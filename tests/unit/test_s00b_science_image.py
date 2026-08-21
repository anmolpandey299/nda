"""S00-B: the frozen science image specification.

The stock RunPod container is not the science environment and must never be accepted as one.
Nothing here asserts a measurement that still requires execution on the H100.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
from preflight import TBD, is_environment_lock_manifest

SCIENCE_TORCH = "2.13.0"
SCIENCE_CUDA_BUILD = "cu130"
SCIENCE_TORCH_PINNED = f"{SCIENCE_TORCH}+{SCIENCE_CUDA_BUILD}"
OBSERVED_DRIVER = "580.126.09"
STOCK_PYTHON = "3.11.10"
STOCK_TORCH = "2.4.1+cu124"

#: Values that only real execution can produce [AUTH: 03 §8; 00 §0.2.4; 01 §30].
STILL_UNMEASURED = (
    "bf16_fp32_tolerance",
    "nondeterminism_sources",
    "batch_size",
    "throughput_sequences_per_second",
    "peak_vram",
    "projected_gpu_hours",
)


@pytest.fixture(scope="module")
def pyproject(repo_root: Path) -> dict[str, object]:
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ selection
def test_requires_python_is_unchanged(pyproject: dict[str, object]) -> None:
    project = pyproject["project"]
    assert isinstance(project, dict)
    assert project["requires-python"] == ">=3.13,<3.14"


def test_science_extra_pins_the_selected_torch(pyproject: dict[str, object]) -> None:
    project = pyproject["project"]
    assert isinstance(project, dict)
    extras = project["optional-dependencies"]
    assert isinstance(extras, dict)
    science = extras["science"]
    assert isinstance(science, list)
    torch_req = next(r for r in science if r.startswith("torch"))
    assert f"torch=={SCIENCE_TORCH}" in torch_req
    assert "sys_platform == 'linux'" in torch_req
    assert "platform_machine == 'x86_64'" in torch_req
    for package in ("transformers", "peft", "accelerate", "opacus"):
        assert any(r.startswith(package) for r in science), package


def test_torch_comes_from_the_cuda_130_index(pyproject: dict[str, object]) -> None:
    """Driver 580.126.09 reports CUDA 13.0, so the cu130 build is the exact match and no
    minor-version compatibility fallback is relied on [AUTH: 01 §9, §12]."""
    tool = pyproject["tool"]
    assert isinstance(tool, dict)
    uv = tool["uv"]
    assert isinstance(uv, dict)
    assert uv["sources"]["torch"]["index"] == "pytorch-cu130"
    index = next(i for i in uv["index"] if i["name"] == "pytorch-cu130")
    assert index["url"].endswith(f"/whl/{SCIENCE_CUDA_BUILD}")
    assert index["explicit"] is True


def test_lock_pins_the_exact_torch_and_cuda_build(repo_root: Path) -> None:
    lock = (repo_root / "uv.lock").read_text(encoding="utf-8")
    match = re.search(r'name = "torch"\nversion = "([^"]+)"', lock)
    assert match, "torch is absent from the environment lock"
    assert match.group(1) == SCIENCE_TORCH_PINNED


def test_cuda_runtime_is_locked_not_inherited(repo_root: Path) -> None:
    """The cu130 wheel pulls pinned nvidia-* wheels, so the runtime is hash-pinned in the
    lock and the image needs no mutable NVIDIA base [AUTH: 01 §12]."""
    lock = (repo_root / "uv.lock").read_text(encoding="utf-8")
    nvidia = set(re.findall(r'name = "(nvidia-[a-z0-9-]+)"', lock))
    assert len(nvidia) >= 10, sorted(nvidia)
    assert any(n.startswith("nvidia-cudnn") for n in nvidia)
    assert any(n.startswith("nvidia-nccl") for n in nvidia)


def test_cpu_dev_toolchain_is_still_pinned(repo_root: Path) -> None:
    """Adding the science set must not unpin the CPU/dev lane."""
    lock = (repo_root / "uv.lock").read_text(encoding="utf-8")
    for tool in ("ruff", "mypy", "pytest"):
        assert re.search(rf'name = "{tool}"\nversion = "\d', lock), tool


def test_stock_pod_torch_is_not_reused(repo_root: Path) -> None:
    lock = (repo_root / "uv.lock").read_text(encoding="utf-8")
    assert STOCK_TORCH not in lock
    assert '"2.4.1' not in lock


# ------------------------------------------------------------------ image
def test_science_and_sealed_stages_exist(repo_root: Path) -> None:
    text = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    assert "AS science\n" in text
    assert "AS science-sealed" in text


def test_every_base_is_digest_pinned(repo_root: Path) -> None:
    text = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("FROM "):
            ref = line.split()[1]
            assert ref.startswith("${") or "@sha256:" in ref, line
            assert ":latest" not in ref, line


def test_sealed_stage_bakes_the_image_identity(repo_root: Path) -> None:
    text = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    sealed = text.split("AS science-sealed", 1)[1]
    assert "/etc/pmm-image.json" in sealed
    assert '"lane":"science"' in sealed.replace(" ", "")
    assert "image_ref" in sealed and "image_digest" in sealed


def test_science_image_installs_only_from_the_frozen_lock(repo_root: Path) -> None:
    text = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    science = text.split("AS science\n", 1)[1].split("AS science-sealed", 1)[0]
    assert "uv sync --frozen" in science
    assert "--extra science" in science
    assert "pip install" not in science


def test_capture_still_reads_identity_from_inside_the_image(repo_root: Path) -> None:
    script = (repo_root / "scripts" / "capture_environment.sh").read_text(encoding="utf-8")
    assert "/etc/pmm-image.json" in script
    assert "CALLER_SUPPLIED_UNVERIFIED" in script
    assert "IMAGE_LANE" in script


# ------------------------------------------------------------------ workflow
def test_build_script_runs_outside_the_pod_and_needs_no_docker_in_docker(
    repo_root: Path,
) -> None:
    script = (repo_root / "scripts" / "build_science_image.sh").read_text(encoding="utf-8")
    assert "docker is required on the BUILD host" in script
    assert "--target science" in script and "--target science-sealed" in script
    assert "RepoDigests" in script, "the pushed digest must be read, not assumed"


def test_bootstrap_checks_every_required_property(repo_root: Path) -> None:
    script = (repo_root / "scripts" / "bootstrap_runpod_s00b.sh").read_text(encoding="utf-8")
    for probe in (
        "REQUIRED_COMMIT",
        "git status --porcelain",
        "nvidia-smi",
        "EXPECTED_PYTHON_MAJOR_MINOR",
        "EXPECTED_TORCH",
        "torch.cuda.is_available",
        "/etc/pmm-image.json",
        "make env-capture",
        "make gpu-smoke",
        "make preflight",
        "make bundle-verify",
    ):
        assert probe in script, probe
    assert script.count("fail ") >= 10, "every check must fail closed"
    assert f'EXPECTED_TORCH="{SCIENCE_TORCH_PINNED}"' in script


# ------------------------------------------------------------------ honesty
def test_hardware_probe_is_recorded_and_rejected(repo_root: Path) -> None:
    probe = json.loads(
        (repo_root / "manifests/environments/S00B_HARDWARE_PROBE.json").read_text("utf-8")
    )
    assert probe["accepted_as_environment_lock"] is False
    assert probe["observed"]["nvidia_driver"] == OBSERVED_DRIVER
    assert probe["observed"]["stock_container_python"] == STOCK_PYTHON
    assert probe["observed"]["stock_torch"] == STOCK_TORCH
    assert probe["observed"]["docker_available_in_pod"] is False
    assert len(probe["rejected_because"]) >= 4


def test_probe_is_not_mistaken_for_an_environment_lock(repo_root: Path) -> None:
    path = repo_root / "manifests/environments/S00B_HARDWARE_PROBE.json"
    assert not is_environment_lock_manifest(path)
    assert is_environment_lock_manifest(Path(f"{'a' * 64}.json"))


def test_measurements_requiring_hardware_are_still_unmeasured(repo_root: Path) -> None:
    """No throughput, VRAM or BF16 tolerance may be invented [AUTH: 03 §8; 00 §0.2.4]."""
    readiness = json.loads(
        (repo_root / "artifacts/p0_pre/P0_PRE_READINESS.json").read_text("utf-8")
    )
    assert readiness["environment_lock_sha256"] == TBD
    assert readiness["P0_PRE_READY"] is False
    probe = json.loads(
        (repo_root / "manifests/environments/S00B_HARDWARE_PROBE.json").read_text("utf-8")
    )
    assert set(STILL_UNMEASURED) <= set(probe["still_unmeasured"])
