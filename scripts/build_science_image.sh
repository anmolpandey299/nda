#!/usr/bin/env bash
# Build and seal the S00-B science image OUTSIDE the RunPod pod [AUTH: 01 §12(8)(9)].
#
# Docker is not available inside a stock RunPod pod, so the image is built locally or in CI,
# pushed to a registry, and RunPod is launched with the resulting immutable digest. Nothing
# here runs on RunPod and nothing requires Docker-in-Docker.
#
#   bash scripts/build_science_image.sh <registry/repo> [tag]
#
# TARGET PLATFORM IS linux/amd64, ALWAYS AND EXPLICITLY.
# The scientific execution target is a RunPod H100 on linux/amd64. A plain `docker build` on
# an Apple Silicon host produces a linux/arm64 image, which RunPod rejects with
# "no matching manifest for linux/amd64 in the manifest list entries". Every pass therefore
# builds with buildx under an explicit --platform and pushes straight from buildx, so the
# local image store is never the source of truth. linux/arm64 is deliberately NOT built.
#
# Two passes, because an image cannot contain its own digest:
#   pass 1  build the `science` stage, push, read the pushed digest D
#   pass 2  build `science-sealed` FROM that digest, bake D into /etc/pmm-image.json, push
set -euo pipefail

TARGET_PLATFORM="linux/amd64"
IMAGE_REPO="${1:?usage: build_science_image.sh <registry/repo> [tag]}"
TAG="${2:-s00b}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

command -v docker >/dev/null 2>&1 || { echo "docker is required on the BUILD host" >&2; exit 1; }
docker buildx version >/dev/null 2>&1 \
  || { echo "docker buildx is required to build for $TARGET_PLATFORM" >&2; exit 1; }

if [ -n "$(git status --porcelain -uall)" ]; then
  echo "refusing to build from a dirty tree [AUTH: 01 §35(3), §36]" >&2
  exit 1
fi
# BUILD_COMMIT: the bundle commit. The image must be built from the commit that CONTAINS the
# acceptance bundle, not from the scientific candidate the bundle describes - otherwise the
# image ships no bundle and bundle verification inside the pod is impossible.
COMMIT="$(git rev-parse HEAD)"

# The default `docker` driver cannot always push cross-platform builds; a docker-container
# builder can, and creating one is idempotent.
BUILDER="pmm-linux-amd64"
if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
  docker buildx create --name "$BUILDER" --driver docker-container >/dev/null
fi
docker buildx inspect --bootstrap "$BUILDER" >/dev/null

META_DIR="$(mktemp -d)"
trap 'rm -rf "$META_DIR"' EXIT

# Read the digest of what was actually pushed, from buildx metadata — never from the local
# image store, which may hold a different architecture.
pushed_digest() {
  python3 -c '
import json, sys
meta = json.load(open(sys.argv[1]))
digest = meta.get("containerimage.digest", "")
if not digest.startswith("sha256:"):
    sys.exit("buildx did not report a pushed image digest")
print(digest)
' "$1"
}

# Fail closed unless the pushed artifact really contains a linux/amd64 image.
require_amd64() {
  local ref="$1"
  docker buildx imagetools inspect "$ref" --format '{{json .}}' > "$META_DIR/inspect.json" \
    || { echo "cannot inspect $ref in the registry" >&2; exit 1; }
  python3 -c '
import json, sys

want_os, want_arch = "linux", "amd64"
data = json.load(open(sys.argv[1]))


def hit(os_, arch):
    return os_ == want_os and arch == want_arch


found = False
manifest = data.get("manifest") or data.get("Manifest") or {}
for entry in manifest.get("manifests", []) or []:
    platform = entry.get("platform") or {}
    # attestation manifests report unknown/unknown; ignore them rather than fail
    if hit(platform.get("os"), platform.get("architecture")):
        found = True

image = data.get("image") or data.get("Image") or {}
if isinstance(image, dict):
    if hit(image.get("os"), image.get("architecture")):
        found = True
    for key, value in image.items():
        if key == f"{want_os}/{want_arch}":
            found = True
        elif isinstance(value, dict) and hit(value.get("os"), value.get("architecture")):
            found = True

if not found:
    sys.exit(
        f"pushed artifact {sys.argv[2]} contains no {want_os}/{want_arch} manifest; "
        "RunPod H100 would reject it"
    )
print(f"   verified {want_os}/{want_arch} present in {sys.argv[2]}")
' "$META_DIR/inspect.json" "$ref"
}

echo "== pass 1: build and push the unsealed science stage ($TARGET_PLATFORM) =="
docker buildx build \
  --builder "$BUILDER" \
  --platform "$TARGET_PLATFORM" \
  --target science \
  --tag "${IMAGE_REPO}:${TAG}-unsealed" \
  --metadata-file "$META_DIR/pass1.json" \
  --push \
  .
DIGEST="$(pushed_digest "$META_DIR/pass1.json")"
echo "   parent digest = $DIGEST"
require_amd64 "${IMAGE_REPO}@${DIGEST}"

echo "== pass 2: seal the identity into the image ($TARGET_PLATFORM) =="
docker buildx build \
  --builder "$BUILDER" \
  --platform "$TARGET_PLATFORM" \
  --target science-sealed \
  --build-arg "SEALED_PARENT_REF=${IMAGE_REPO}" \
  --build-arg "SEALED_PARENT_DIGEST=${DIGEST}" \
  --build-arg "SOURCE_GIT_COMMIT=${COMMIT}" \
  --tag "${IMAGE_REPO}:${TAG}" \
  --metadata-file "$META_DIR/pass2.json" \
  --push \
  .
SEALED_DIGEST="$(pushed_digest "$META_DIR/pass2.json")"
require_amd64 "${IMAGE_REPO}@${SEALED_DIGEST}"

RECORD="manifests/environments/S00B_IMAGE_RECORD.json"
python3 - "$RECORD" "$IMAGE_REPO" "$TAG" "$DIGEST" "$SEALED_DIGEST" "$COMMIT" \
         "$TARGET_PLATFORM" <<'PYEOF'
import json, sys
record, repo, tag, parent, sealed, commit, platform = sys.argv[1:8]
json.dump({
    "authority": "01 §12(8)(9), §16; plan §5.6",
    "lane": "science",
    "target_platform": platform,
    "image_ref": repo,
    "image_tag": tag,
    "unsealed_parent_digest": parent,
    "sealed_image_digest": sealed,
    "source_git_commit": commit,
    "base_image_ref": "python:3.13-slim",
    "base_image_digest":
        "sha256:ffb752e139c0a19692a43af8d8523b274222dd68eebad5d583b45c2201c6e30a",
    "note": "Launch RunPod with sealed_image_digest, never with the tag.",
}, open(record, "w"), indent=2, sort_keys=True)
open(record, "a").write("\n")
PYEOF

cat <<SUMMARY

science image ready
  target platform    : ${TARGET_PLATFORM}
  launch RunPod with : ${IMAGE_REPO}@${SEALED_DIGEST}
  BUILD_COMMIT       : ${COMMIT}
  bootstrap with     : bash scripts/bootstrap_runpod_s00b.sh ${COMMIT}
  recorded in        : ${RECORD}
  built from commit  : ${COMMIT}
Never launch by tag; the digest is the identity [AUTH: 01 §12(9)].
SUMMARY
