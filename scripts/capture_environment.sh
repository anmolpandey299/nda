#!/usr/bin/env bash
# Environment capture — 01 §12 steps 2-9, run on the real H100 image.
#
# Writes manifests/environments/<env_id>.json where env_id = ENVIRONMENT_LOCK_SHA256, the
# SHA256 over the COMPLETE frozen execution environment: uv.lock, Docker image digest, CUDA
# runtime and driver, torch version and CUDA build, Python version, GPU model
# [AUTH: 01 §12(5)-(9), §15, §16, §30, §32; plan §5.6].
#
# Any component this machine cannot resolve is written as the literal TBD_REQUIRES_HARDWARE
# and the script exits non-zero: an unresolved environment must never look captured
# [AUTH: 03 §8].
set -euo pipefail

TBD="TBD_REQUIRES_HARDWARE"
OUT_DIR="manifests/environments"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    *) echo "usage: capture_environment.sh [--out DIR]" >&2; exit 2 ;;
  esac
done
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p "$OUT_DIR"

py() { [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3; }
PY="$(py)"

# step 5 — hash uv.lock
UV_LOCK_SHA256="$( [ -f uv.lock ] && shasum -a 256 uv.lock | awk '{print $1}' || echo "$TBD" )"
# step 6 — nvidia-smi verbatim
if command -v nvidia-smi >/dev/null 2>&1; then
  NVIDIA_SMI="$(nvidia-smi 2>&1 || true)"
  GPU_MODEL="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)"
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  CUDA_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
  NVIDIA_SMI="$TBD"; GPU_MODEL="$TBD"; GPU_UUID="$TBD"; GPU_COUNT="$TBD"; CUDA_DRIVER="$TBD"
fi
# step 7 — CUDA runtime and the torch build actually installed
CUDA_RUNTIME="$( command -v nvcc >/dev/null 2>&1 \
  && nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1 || echo "$TBD" )"
TORCH_VERSION="$($PY -c 'import torch;print(torch.__version__)' 2>/dev/null || echo "$TBD")"
TORCH_CUDA_BUILD="$($PY -c 'import torch;print(torch.version.cuda)' 2>/dev/null || echo "$TBD")"
PYTHON_VERSION="$($PY -c 'import platform;print(platform.python_version())')"
# steps 8-9 — image tag and digest are supplied by the build, never guessed
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-$TBD}"
DOCKER_IMAGE_DIGEST="${DOCKER_IMAGE_DIGEST:-$TBD}"
BF16_FP32_TOLERANCE="${BF16_FP32_TOLERANCE:-$TBD}"
NONDETERMINISM_SOURCES="${NONDETERMINISM_SOURCES:-$TBD}"
DEPS="$($PY -m pip freeze 2>/dev/null | tr '\n' ';' || echo "$TBD")"

TMP="$(mktemp)"
UV_LOCK_SHA256="$UV_LOCK_SHA256" NVIDIA_SMI="$NVIDIA_SMI" GPU_MODEL="$GPU_MODEL" \
GPU_UUID="$GPU_UUID" GPU_COUNT="$GPU_COUNT" CUDA_DRIVER="$CUDA_DRIVER" \
CUDA_RUNTIME="$CUDA_RUNTIME" TORCH_VERSION="$TORCH_VERSION" \
TORCH_CUDA_BUILD="$TORCH_CUDA_BUILD" PYTHON_VERSION="$PYTHON_VERSION" \
DOCKER_IMAGE_TAG="$DOCKER_IMAGE_TAG" DOCKER_IMAGE_DIGEST="$DOCKER_IMAGE_DIGEST" \
BF16_FP32_TOLERANCE="$BF16_FP32_TOLERANCE" NONDETERMINISM_SOURCES="$NONDETERMINISM_SOURCES" \
DEPS="$DEPS" OUT_DIR="$OUT_DIR" "$PY" - <<'PYEOF' > "$TMP"
import datetime, json, os, sys
sys.path.insert(0, "scripts")
from preflight import environment_lock_sha256, validate_environment_manifest

m = {
    "uv_lock_sha256": os.environ["UV_LOCK_SHA256"],
    "docker_image_digest": os.environ["DOCKER_IMAGE_DIGEST"],
    "docker_image_tag": os.environ["DOCKER_IMAGE_TAG"],
    "cuda_runtime": os.environ["CUDA_RUNTIME"],
    "cuda_driver": os.environ["CUDA_DRIVER"],
    "torch_version": os.environ["TORCH_VERSION"],
    "torch_cuda_build": os.environ["TORCH_CUDA_BUILD"],
    "python_version": os.environ["PYTHON_VERSION"],
    "gpu_model": os.environ["GPU_MODEL"],
    "gpu_uuid": os.environ["GPU_UUID"],
    "gpu_count": os.environ["GPU_COUNT"],
    "nvidia_smi_capture": os.environ["NVIDIA_SMI"],
    "dependency_versions": os.environ["DEPS"],
    "bf16_fp32_tolerance": os.environ["BF16_FP32_TOLERANCE"],
    "nondeterminism_sources": os.environ["NONDETERMINISM_SOURCES"],
    "capture_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "authority": "01 §12(2)-(9), §15, §16, §30, §32",
}
m["environment_lock_sha256"] = environment_lock_sha256(m)
missing = validate_environment_manifest(m)
unresolved = sorted(k for k, v in m.items() if v == "TBD_REQUIRES_HARDWARE")
print(json.dumps({"manifest": m, "missing": missing, "unresolved": unresolved}))
PYEOF

ENV_ID="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["manifest"]["environment_lock_sha256"])' "$TMP")"
"$PY" -c 'import json,sys;json.dump(json.load(open(sys.argv[1]))["manifest"],open(sys.argv[2],"w"),indent=2,sort_keys=True);open(sys.argv[2],"a").write("\n")' \
  "$TMP" "$OUT_DIR/$ENV_ID.json"
UNRESOLVED="$("$PY" -c 'import json,sys;print(" ".join(json.load(open(sys.argv[1]))["unresolved"]))' "$TMP")"
rm -f "$TMP"

echo "wrote $OUT_DIR/$ENV_ID.json"
if [ -n "$UNRESOLVED" ]; then
  echo "S00_ENV_CAPTURE = TBD_REQUIRES_HARDWARE" >&2
  echo "unresolved: $UNRESOLVED" >&2
  echo "resolve on the RunPod H100 SXM image per 01 §12 steps 2-9, then rerun" >&2
  exit 1
fi
echo "S00_ENV_CAPTURE = COMPLETE"
