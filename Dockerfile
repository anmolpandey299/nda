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
CMD ["make", "preflight"]

# ---------------------------------------------------------------- science lane (S00-B)
# Resolve on the RunPod H100 SXM image per 01 §12 steps 2-9, then build with:
#   --build-arg SCIENCE_BASE_IMAGE=<repo>@sha256:<digest>
#   --build-arg IMAGE_DIGEST=sha256:<digest>
# A tag without a digest is rejected by capture_environment.sh.
ARG SCIENCE_BASE_IMAGE=TBD_REQUIRES_HARDWARE
FROM ${SCIENCE_BASE_IMAGE} AS science
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_FROZEN=1
COPY --from=uvbin /uv /usr/local/bin/uv
WORKDIR /repo
ARG SCIENCE_BASE_IMAGE
ARG IMAGE_DIGEST=TBD_REQUIRES_HARDWARE
RUN printf '{"image_ref":"%s","image_digest":"%s","lane":"science"}\n' \
      "$SCIENCE_BASE_IMAGE" "$IMAGE_DIGEST" > /etc/pmm-image.json
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# The CUDA-coupled dependency set is added to uv.lock at S00-B; until then this fails.
RUN uv sync --frozen --no-install-project
COPY . .
