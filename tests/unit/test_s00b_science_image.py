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
from check_repo_invariants import (
    VENV_GUARD_MARKER,
    _dockerfile_stages,
    check_build_context,
)
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
TARGET_PLATFORM = "linux/amd64"


@pytest.fixture(scope="module")
def build_script(repo_root: Path) -> str:
    return (repo_root / "scripts" / "build_science_image.sh").read_text(encoding="utf-8")


def _build_passes(script: str) -> list[str]:
    """The two `docker buildx build` invocations, as text blocks."""
    parts = script.split("docker buildx build")[1:]
    return [p.split("\n\n", 1)[0] for p in parts]


def test_build_script_runs_outside_the_pod_and_needs_no_docker_in_docker(
    build_script: str,
) -> None:
    assert "docker is required on the BUILD host" in build_script
    assert "--target science \\" in build_script
    assert "--target science-sealed \\" in build_script
    assert "containerimage.digest" in build_script, (
        "the pushed digest must come from buildx metadata, not the local image store"
    )


def test_both_passes_build_for_linux_amd64(build_script: str) -> None:
    """RunPod rejected an arm64-only push with 'no matching manifest for linux/amd64'.
    A plain `docker build` on Apple Silicon produces arm64, so the platform is explicit."""
    passes = _build_passes(build_script)
    assert len(passes) == 2, f"expected two buildx passes, found {len(passes)}"
    for index, block in enumerate(passes, start=1):
        assert '--platform "$TARGET_PLATFORM"' in block, f"pass {index} has no --platform"
        assert "--push" in block, f"pass {index} does not push from buildx"
    assert f'TARGET_PLATFORM="{TARGET_PLATFORM}"' in build_script


def test_pass_one_and_two_target_the_right_stages(build_script: str) -> None:
    first, second = _build_passes(build_script)
    assert "--target science \\" in first and "science-sealed" not in first
    assert "--target science-sealed \\" in second


def test_buildx_is_required_and_the_local_store_is_not_trusted(build_script: str) -> None:
    assert "docker buildx version" in build_script
    assert "docker build " not in build_script.replace("docker buildx build ", "")
    assert "RepoDigests" not in build_script
    assert "\ndocker push" not in build_script


def test_pushed_manifest_is_verified_to_contain_linux_amd64(build_script: str) -> None:
    """Fail closed if the registry artifact has no linux/amd64 entry."""
    assert "require_amd64" in build_script
    assert build_script.count("require_amd64 ") >= 2, "both passes must be verified"
    assert "imagetools inspect" in build_script
    assert "contains no " in build_script and "RunPod H100 would reject it" in build_script


def test_arm64_is_not_built(build_script: str) -> None:
    """The scientific execution target is RunPod H100 linux/amd64 only. arm64 may be named in
    prose explaining the failure, but never in a --platform argument."""
    code = "\n".join(
        line for line in build_script.splitlines() if not line.lstrip().startswith("#")
    )
    platform_args = re.findall(r"--platform\s+(\S+)", code)
    assert platform_args, "no --platform argument found"
    assert all(arg == '"$TARGET_PLATFORM"' for arg in platform_args), platform_args
    assert f'TARGET_PLATFORM="{TARGET_PLATFORM}"' in build_script
    assert "arm64" not in code


def test_two_pass_sealed_design_is_preserved(build_script: str) -> None:
    assert "SEALED_PARENT_DIGEST=${DIGEST}" in build_script
    assert "SOURCE_GIT_COMMIT=${COMMIT}" in build_script
    assert "sealed_image_digest" in build_script
    assert "refusing to build from a dirty tree" in build_script


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


# ------------------------------------------------------------------ host-venv contamination
DOCKERIGNORE_REQUIRED = (
    ".venv",
    "**/.venv",
    "__pycache__",
    "**/__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
)


@pytest.fixture(scope="module")
def dockerfile(repo_root: Path) -> str:
    return (repo_root / "Dockerfile").read_text(encoding="utf-8")


def _patterns(repo_root: Path) -> set[str]:
    text = (repo_root / ".dockerignore").read_text(encoding="utf-8")
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_dockerignore_excludes_the_host_venv(repo_root: Path) -> None:
    """The defect: no .dockerignore, so `COPY . .` overwrote the image's Linux .venv with the
    host's macOS one and /repo/.venv/bin/python dangled at /opt/homebrew."""
    assert (repo_root / ".dockerignore").is_file()
    patterns = _patterns(repo_root)
    for required in DOCKERIGNORE_REQUIRED:
        assert required in patterns, f".dockerignore does not exclude {required!r}"


def test_dockerignore_keeps_git(repo_root: Path) -> None:
    """The image ships /repo as a pinned worktree so the bootstrap can verify the commit and
    the clean tree inside the pod [AUTH: 01 §35(3), §36]."""
    assert ".git" not in _patterns(repo_root)


def test_every_full_context_copy_revalidates_the_image_venv(dockerfile: str) -> None:
    stages = _dockerfile_stages(dockerfile)
    checked = 0
    for name, body in stages:
        copy_at = next((i for i, line in enumerate(body) if line.strip() == "COPY . ."), None)
        if copy_at is None:
            continue
        checked += 1
        assert any(VENV_GUARD_MARKER in line for line in body[copy_at:]), (
            f"stage {name}: `COPY . .` has no post-COPY interpreter check"
        )
    assert checked == 2, f"expected the cpu-dev and science lanes, found {checked}"


def test_guard_proves_a_linux_interpreter_and_the_locked_torch(dockerfile: str) -> None:
    science = dockerfile.split("AS science\n", 1)[1].split("AS science-sealed", 1)[0]
    assert "sys.platform == 'linux'" in science
    assert "sys.version_info[:2] == (3, 13)" in science
    assert f"torch.__version__ == '{SCIENCE_TORCH_PINNED}'" in science
    assert "/opt/homebrew/*" in science, "the observed host-symlink case must be named"

    cpu = dockerfile.split("AS cpu-dev\n", 1)[1].split("AS science\n", 1)[0]
    assert "sys.platform == 'linux'" in cpu
    assert "sys.version_info[:2] == (3, 13)" in cpu


def test_frozen_uv_sync_remains_authoritative(dockerfile: str) -> None:
    """Dependencies are created by uv from the frozen lock, never installed by hand."""
    assert "uv sync --frozen --extra cpu-dev --extra science --no-install-project" in dockerfile
    assert "uv sync --frozen --extra cpu-dev --no-install-project" in dockerfile
    code = "\n".join(line for line in dockerfile.splitlines() if not line.lstrip().startswith("#"))
    assert "pip install" not in code


def test_missing_dockerignore_is_a_violation(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("FROM x@sha256:0 AS a\nCOPY . .\n", encoding="utf-8")
    violations = check_build_context(tmp_path)
    assert violations and violations[0].invariant == "I16"


def test_dockerignore_without_venv_is_a_violation(tmp_path: Path) -> None:
    (tmp_path / ".dockerignore").write_text("__pycache__\n", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text("FROM x@sha256:0 AS a\n", encoding="utf-8")
    assert any(".venv" in v.reason for v in check_build_context(tmp_path))


def test_unguarded_full_context_copy_is_a_violation(tmp_path: Path) -> None:
    (tmp_path / ".dockerignore").write_text(
        "\n".join(DOCKERIGNORE_REQUIRED) + "\n", encoding="utf-8"
    )
    (tmp_path / "Dockerfile").write_text(
        "FROM x@sha256:0 AS a\nRUN uv sync --frozen\nCOPY . .\n", encoding="utf-8"
    )
    violations = check_build_context(tmp_path)
    assert any("post-COPY" in v.reason or "still" in v.reason for v in violations), violations


def test_excluding_git_is_a_violation(tmp_path: Path) -> None:
    (tmp_path / ".dockerignore").write_text(
        "\n".join((*DOCKERIGNORE_REQUIRED, ".git")) + "\n", encoding="utf-8"
    )
    (tmp_path / "Dockerfile").write_text("FROM x@sha256:0 AS a\n", encoding="utf-8")
    assert any(".git" in v.reason for v in check_build_context(tmp_path))


def test_platform_repair_is_still_intact(build_script: str) -> None:
    """This repair must not have regressed the linux/amd64 fix."""
    passes = _build_passes(build_script)
    assert len(passes) == 2
    assert all('--platform "$TARGET_PLATFORM"' in p and "--push" in p for p in passes)
    assert f'TARGET_PLATFORM="{TARGET_PLATFORM}"' in build_script


def test_every_invariant_function_is_registered() -> None:
    """check_image_pins was defined but never added to CHECKS, so the digest-pin invariant
    silently never ran. No invariant may be defined and left unwired."""
    import check_repo_invariants as chk

    defined = {
        name for name in dir(chk) if name.startswith("check_") and callable(getattr(chk, name))
    }
    registered = {fn.__name__ for fn in chk.CHECKS}
    assert defined - registered == set(), f"unwired invariants: {sorted(defined - registered)}"
