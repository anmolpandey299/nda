#!/usr/bin/env bash
# Build and seal the S00-B science image OUTSIDE the RunPod pod [AUTH: 01 §12(8)(9)].
#
# Docker is not available inside a stock RunPod pod, so the image is built locally or in CI,
# pushed to a registry, and RunPod is launched with the resulting immutable digest. Nothing
# here runs on RunPod and nothing requires Docker-in-Docker.
#
#   bash scripts/build_science_image.sh <registry/repo> [tag]
#
# Two passes, because an image cannot contain its own digest:
#   pass 1  build the `science` stage, push, read digest D
#   pass 2  build `science-sealed` FROM that digest, bake D into /etc/pmm-image.json, push
set -euo pipefail

IMAGE_REPO="${1:?usage: build_science_image.sh <registry/repo> [tag]}"
TAG="${2:-s00b}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

command -v docker >/dev/null 2>&1 || { echo "docker is required on the BUILD host" >&2; exit 1; }

if [ -n "$(git status --porcelain -uall)" ]; then
  echo "refusing to build from a dirty tree [AUTH: 01 §35(3), §36]" >&2
  exit 1
fi
COMMIT="$(git rev-parse HEAD)"

echo "== pass 1: build and push the unsealed science stage =="
docker build --target science -t "${IMAGE_REPO}:${TAG}-unsealed" .
docker push "${IMAGE_REPO}:${TAG}-unsealed"
DIGEST="$(docker inspect --format='{{index .RepoDigests 0}}' "${IMAGE_REPO}:${TAG}-unsealed" \
          | sed 's/.*@//')"
case "$DIGEST" in
  sha256:*) : ;;
  *) echo "could not obtain an immutable digest for pass 1" >&2; exit 1 ;;
esac
echo "   parent digest = $DIGEST"

echo "== pass 2: seal the identity into the image =="
docker build --target science-sealed \
  --build-arg "SEALED_PARENT_REF=${IMAGE_REPO}" \
  --build-arg "SEALED_PARENT_DIGEST=${DIGEST}" \
  --build-arg "SOURCE_GIT_COMMIT=${COMMIT}" \
  -t "${IMAGE_REPO}:${TAG}" .
docker push "${IMAGE_REPO}:${TAG}"
SEALED_DIGEST="$(docker inspect --format='{{index .RepoDigests 0}}' "${IMAGE_REPO}:${TAG}" \
                 | sed 's/.*@//')"

RECORD="manifests/environments/S00B_IMAGE_RECORD.json"
python3 - "$RECORD" "$IMAGE_REPO" "$TAG" "$DIGEST" "$SEALED_DIGEST" "$COMMIT" <<'PYEOF'
import json, sys
record, repo, tag, parent, sealed, commit = sys.argv[1:7]
json.dump({
    "authority": "01 §12(8)(9), §16; plan §5.6",
    "lane": "science",
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
  launch RunPod with : ${IMAGE_REPO}@${SEALED_DIGEST}
  recorded in        : ${RECORD}
  built from commit  : ${COMMIT}
Never launch by tag; the digest is the identity [AUTH: 01 §12(9)].
SUMMARY
