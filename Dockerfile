# Execution surface [AUTH: 01 §11, §12(8)-(9), §39 S00].
#
# Base images are pinned by immutable digest, never by a mutable tag: a bare `python:3.13-slim` and a
# floating uv tag both move, so a tag-pinned build is not the reproducible environment 01 §12
# requires. Digests below were resolved from the registries at S00-A.
#
# Two lanes, one file:
#   * cpu-dev  — buildable now, installs strictly from the committed S00-A uv.lock.
#   * science  — the H100 lane. Its base is resolved at S00-B on the real RunPod image and is
#                TBD_REQUIRES_HARDWARE until then, so the stage fails loudly rather than
#                silently building against a guessed CUDA base [AUTH: 03 §8].
#
# No `pip install -U` ever runs inside an experiment [AUTH: 01 §12].

# uv is pinned to the exact version that produced uv.lock.
# Global build arguments used by later FROM instructions.
# Valid defaults are required so Docker can parse every stage during pass 1.
# scripts/build_science_image.sh overrides these with the immutable pass-1 science image.
ARG SEALED_PARENT_REF=python:3.13-slim
ARG SEALED_PARENT_DIGEST=sha256:ffb752e139c0a19692a43af8d8523b274222dd68eebad5d583b45c2201c6e30a

FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uvbin

# ---------------------------------------------------------------- cpu-dev lane (S00-A)
FROM python:3.13-slim@sha256:ffb752e139c0a19692a43af8d8523b274222dd68eebad5d583b45c2201c6e30a AS cpu-dev
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_FROZEN=1
COPY --from=uvbin /uv /usr/local/bin/uv
WORKDIR /repo

# Bake the image identity so capture_environment.sh reads it from inside the image instead of
# trusting a caller-supplied value [AUTH: 01 §12(8)(9), §16].
ARG IMAGE_REF="python:3.13-slim"
ARG IMAGE_DIGEST="sha256:ffb752e139c0a19692a43af8d8523b274222dd68eebad5d583b45c2201c6e30a"
RUN printf '{"image_ref":"%s","image_digest":"%s","lane":"cpu-dev"}\n' \
      "$IMAGE_REF" "$IMAGE_DIGEST" > /etc/pmm-image.json

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra cpu-dev --no-install-project
COPY . .

# Fail closed if a host virtualenv reached the image and shadowed the one uv built. Without
# this, `COPY . .` silently replaced the Linux .venv with a macOS one and the defect only
# surfaced on the H100 [AUTH: 01 §12, §46].
RUN set -eu; \
    if [ -L /repo/.venv/bin/python ]; then \
      target="$(readlink /repo/.venv/bin/python)"; \
      case "$target" in \
        /opt/homebrew/*|/Users/*|/usr/local/Cellar/*) \
          echo "host virtualenv leaked into the image: .venv/bin/python -> $target" >&2; \
          exit 1 ;; \
      esac; \
    fi; \
    test -x /repo/.venv/bin/python \
      || { echo "/repo/.venv/bin/python is missing or not executable" >&2; exit 1; }; \
    /repo/.venv/bin/python -c "\
import sys; \
assert sys.platform == 'linux', sys.platform; \
assert sys.version_info[:2] == (3, 13), sys.version; \
import pytest, ruff; \
print('cpu-dev environment OK', sys.version.split()[0])"
CMD ["make", "preflight"]

# ---------------------------------------------------------------- science lane (S00-B)
# Built OUTSIDE the RunPod pod (locally or in CI), pushed to a registry, then launched on
# RunPod BY DIGEST as a custom container image. Docker is not available inside a stock pod,
# so nothing here requires Docker-in-Docker [AUTH: 01 §12(8)(9), §9].
#
# The base is the same digest-pinned CPython 3.13 image as the CPU lane: torch's cu130 build
# pulls 15 pinned nvidia-* wheels, so the CUDA runtime is locked in uv.lock rather than
# inherited from a mutable NVIDIA base image. Only the host driver is needed, which RunPod
# provides (observed: 580.126.09, CUDA 13.0).
FROM python:3.13-slim@sha256:ffb752e139c0a19692a43af8d8523b274222dd68eebad5d583b45c2201c6e30a AS science
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_FROZEN=1
COPY --from=uvbin /uv /usr/local/bin/uv
WORKDIR /repo
RUN apt-get update && apt-get install -y --no-install-recommends git make ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# --frozen never re-resolves: a drifted lock fails the build [AUTH: 01 §12, §46].
RUN uv sync --frozen --extra cpu-dev --extra science --no-install-project
COPY . .

# Same fail-closed guard, plus proof that the frozen science environment is the one that
# survived the copy. No GPU is required to run this during the image build [AUTH: 01 §12].
RUN set -eu; \
    if [ -L /repo/.venv/bin/python ]; then \
      target="$(readlink /repo/.venv/bin/python)"; \
      case "$target" in \
        /opt/homebrew/*|/Users/*|/usr/local/Cellar/*) \
          echo "host virtualenv leaked into the image: .venv/bin/python -> $target" >&2; \
          exit 1 ;; \
      esac; \
    fi; \
    test -x /repo/.venv/bin/python \
      || { echo "/repo/.venv/bin/python is missing or not executable" >&2; exit 1; }; \
    /repo/.venv/bin/python -c "\
import sys; \
assert sys.platform == 'linux', sys.platform; \
assert sys.version_info[:2] == (3, 13), sys.version; \
import torch; \
assert torch.__version__ == '2.13.0+cu130', torch.__version__; \
print('science environment OK', sys.version.split()[0], torch.__version__)"

# ------------------------------------------------------- sealed science image (pass 2)
# An image cannot contain its own digest, so identity is sealed in a second pass:
#   1. build+push the `science` stage            -> immutable digest D
#   2. build+push this stage FROM that digest    -> bakes D into /etc/pmm-image.json
# The sealed image is one metadata layer on top of an immutable parent, so the identity it
# reports is verifiable from the registry. capture_environment.sh reads it from inside the
# image and never trusts an environment variable [AUTH: 01 §12(8)(9), §16; 03 §8].
FROM ${SEALED_PARENT_REF}@${SEALED_PARENT_DIGEST} AS science-sealed
ARG SEALED_PARENT_REF
ARG SEALED_PARENT_DIGEST
ARG BASE_IMAGE_REF="python:3.13-slim"
ARG BASE_IMAGE_DIGEST="sha256:ffb752e139c0a19692a43af8d8523b274222dd68eebad5d583b45c2201c6e30a"
ARG SOURCE_GIT_COMMIT=TBD_REQUIRES_HARDWARE
RUN printf '{"image_ref":"%s","image_digest":"%s","base_image_ref":"%s","base_image_digest":"%s","source_git_commit":"%s","lane":"science"}\n' \
      "$SEALED_PARENT_REF" "$SEALED_PARENT_DIGEST" \
      "$BASE_IMAGE_REF" "$BASE_IMAGE_DIGEST" "$SOURCE_GIT_COMMIT" > /etc/pmm-image.json
