# Execution surface [AUTH: 01 §11, §12(8)-(9), §39 S00].
#
# Two lanes, one file:
#   * cpu-dev  — buildable now, installs strictly from the committed S00-A uv.lock.
#   * science  — the H100 lane. Its base image is resolved at S00-B on the real RunPod
#                image and is TBD_REQUIRES_HARDWARE until then, so this stage fails loudly
#                rather than silently building against a guessed CUDA base [AUTH: 03 §8].
#
# No `pip install -U` ever runs inside an experiment [AUTH: 01 §12].

# ---------------------------------------------------------------- cpu-dev lane (S00-A)
FROM python:3.13-slim AS cpu-dev
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_FROZEN=1
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /repo
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra cpu-dev --no-install-project
COPY . .
CMD ["make", "preflight"]

# ---------------------------------------------------------------- science lane (S00-B)
# Resolve on the RunPod H100 SXM image per 01 §12 steps 2-9, then set:
#   --build-arg SCIENCE_BASE_IMAGE=<image@sha256:...>
# and record the tag and digest in manifests/environments/<env_id>.json.
ARG SCIENCE_BASE_IMAGE=TBD_REQUIRES_HARDWARE
FROM ${SCIENCE_BASE_IMAGE} AS science
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_FROZEN=1
WORKDIR /repo
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# The CUDA-coupled dependency set is added to uv.lock at S00-B; until then this fails.
RUN uv sync --frozen --no-install-project
COPY . .
