#!/usr/bin/env bash
# S00-B acceptance on the RunPod H100, inside the sealed science image.
#
#   bash scripts/bootstrap_runpod_s00b.sh <BUILD_COMMIT>
#
# COMMIT SEMANTICS. Three different commits are involved and they are not interchangeable:
#
#   DESCRIBED_COMMIT          the scientific candidate the acceptance bundle describes
#   BUILD_COMMIT              the bundle commit; it CONTAINS the bundle, so the image must be
#                             built from it, otherwise the image ships no acceptance bundle
#   IMAGE_SOURCE_GIT_COMMIT   baked into /etc/pmm-image.json at build time == BUILD_COMMIT
#   BOOTSTRAP_REQUIRED_COMMIT == BUILD_COMMIT, i.e. the argument to this script
#
# Never bootstrap the scientific candidate: that commit predates its own bundle.
#
# Verification only: it starts no training, downloads no model and measures nothing it does
# not observe. Every check FAILS CLOSED [AUTH: 01 §12(2)-(9), §21; 03 §8].
set -euo pipefail

BUILD_COMMIT="${1:?usage: bootstrap_runpod_s00b.sh <BUILD_COMMIT>}"
REQUIRED_COMMIT="$BUILD_COMMIT"
EXPECTED_PYTHON_MAJOR_MINOR="3.13"
EXPECTED_TORCH="2.13.0+cu130"
EXPECTED_CAPABILITY="(9, 0)"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

fail() { echo "S00B_BOOTSTRAP = FAIL: $*" >&2; exit 1; }
ok()   { echo "  ok    $*"; }

PY="$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )"

echo "== 1. exact repository commit =="
HEAD="$(git rev-parse HEAD)"
case "$REQUIRED_COMMIT" in
  ????????????????????????????????????????) : ;;
  *) fail "BUILD_COMMIT must be a full 40-hex SHA obtained from git rev-parse" ;;
esac
git cat-file -e "${REQUIRED_COMMIT}^{commit}" 2>/dev/null \
  || fail "BUILD_COMMIT $REQUIRED_COMMIT is not a commit in this repository"
[ "$HEAD" = "$REQUIRED_COMMIT" ] || fail "HEAD $HEAD != BUILD_COMMIT $REQUIRED_COMMIT"
ok "HEAD = $HEAD"

echo "== 2. clean tree =="
DIRTY="$(git status --porcelain -uall)"
[ -z "$DIRTY" ] || fail "working tree is dirty:
$DIRTY"
ok "no staged, unstaged or untracked changes"

echo "== 3. H100 present =="
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi absent; this is not a GPU pod"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$GPU_NAME" in *H100*) ok "GPU = $GPU_NAME" ;; *) fail "GPU is '$GPU_NAME', not an H100 [AUTH: 01 §9]" ;; esac

echo "== 4. expected Python =="
PYVER="$($PY -c 'import platform;print(platform.python_version())')"
case "$PYVER" in
  ${EXPECTED_PYTHON_MAJOR_MINOR}.*) ok "python = $PYVER" ;;
  *) fail "python $PYVER does not satisfy the frozen requires-python >=3.13,<3.14" ;;
esac

echo "== 5. expected torch build =="
TORCH="$($PY -c 'import torch;print(torch.__version__)' 2>/dev/null)" \
  || fail "torch is not importable; this is not the sealed science image"
[ "$TORCH" = "$EXPECTED_TORCH" ] || fail "torch $TORCH != frozen $EXPECTED_TORCH"
ok "torch = $TORCH"

echo "== 6. torch CUDA availability and device capability =="
$PY -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
  || fail "torch.cuda.is_available() is false"
CAP="$($PY -c 'import torch;print(torch.cuda.get_device_capability(0))')"
[ "$CAP" = "$EXPECTED_CAPABILITY" ] || fail "compute capability $CAP != $EXPECTED_CAPABILITY"
ok "cuda available, capability $CAP, torch.version.cuda=$($PY -c 'import torch;print(torch.version.cuda)')"

echo "== 7. sealed image metadata =="
BAKED=/etc/pmm-image.json
[ -r "$BAKED" ] || fail "$BAKED absent; the stock RunPod container is not the frozen image"
LANE="$($PY -c 'import json,sys;print(json.load(open(sys.argv[1])).get("lane",""))' "$BAKED")"
[ "$LANE" = "science" ] || fail "image lane is '$LANE', not 'science'"
IMG_DIGEST="$($PY -c 'import json,sys;print(json.load(open(sys.argv[1])).get("image_digest",""))' "$BAKED")"
case "$IMG_DIGEST" in
  sha256:????????????????????????????????????????????????????????????????) ok "image digest = $IMG_DIGEST" ;;
  *) fail "image digest '$IMG_DIGEST' is missing or malformed" ;;
esac
IMG_COMMIT="$($PY -c 'import json,sys;print(json.load(open(sys.argv[1])).get("source_git_commit",""))' "$BAKED")"
[ "$IMG_COMMIT" = "$REQUIRED_COMMIT" ] \
  || fail "image was built from $IMG_COMMIT, not the BUILD_COMMIT $REQUIRED_COMMIT"
ok "image built from the build commit"

echo "== 8. environment capture (01 §12 steps 2-9) =="
make env-capture || fail "env-capture left components unresolved"

echo "== 9. gpu smoke =="
make gpu-smoke || fail "gpu-smoke failed"

echo "== 10. ordered gate =="
make preflight || fail "preflight failed"

echo "== 11. bundle verification =="
# Source artifacts are hash-bound; runtime evidence is verified by re-derivation, so
# producing the environment manifest and the readiness file during this run cannot
# invalidate the bundle that authorised the run [AUTH: 01 §16; 02 §C6].
make bundle-verify || fail "source drift: the bundle does not describe the current code"

echo "== 12. S00-B closure record =="
make closure || fail "closure incomplete; the environment identity did not resolve"

cat <<SUMMARY

S00B_BOOTSTRAP = PASS
  commit   $HEAD
  image    $IMG_DIGEST
  gpu      $GPU_NAME
  torch    $TORCH
  closure  stage_acceptance/S00/12_S00B_CLOSURE.json

Commit the environment manifest, the readiness file and the closure record, then regenerate
the bundle so the closure state is the described one.
Throughput, peak VRAM and the BF16 tolerance are measured by their own runs and stay
TBD_REQUIRES_HARDWARE until then [AUTH: 00 §0.2.4; 01 §30; 03 §8].
SUMMARY
