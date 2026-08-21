# Reproduce S00 from a clean machine

```bash
git clone <repo> && cd privacy-model-merging
git checkout stage/s00

curl -LsSf https://astral.sh/uv/install.sh | sh     # or: pipx install uv
uv sync --extra cpu-dev                             # resolves ONLY from the committed uv.lock

bash scripts/install_git_hooks.sh                   # convenience guard [AUTH: 01 §39 S00]
make preflight                                      # steps 0-8, plan §10.1
```

Expected terminal state:

```text
preflight PASS
artifacts/p0_pre/P0_PRE_READINESS.json
  BACKEND_INTEGRATED = false
  SUITE_SCOPE        = STATISTICAL_STACK_ONLY
  P0_PRE_READY       = false
  environment_lock_sha256 = TBD_REQUIRES_HARDWARE
```

`make synthetic`, `make backend-contract` and `make gpu-smoke` report `NOT_RUN(<reason>)`.
An empty lane never reports `PASS` [AUTH: 02 §C6; 00 §34B.3].

## Reproducing S00-B (requires an H100)

```bash
# on the RunPod H100 SXM image, per 01 §12 steps 2-9
uv sync                                             # adds the CUDA-coupled set, relocks
DOCKER_IMAGE_TAG=... DOCKER_IMAGE_DIGEST=... make env-capture
make gpu-smoke
```

`make env-capture` exits non-zero while any component is unresolved, so an unresolved
environment can never look captured [AUTH: 03 §8].

## Determinism notes

The CPU/dev lane pins every tool through `uv.lock`, so `make lint` and `make typecheck`
cannot flip against a byte-identical tree. The scientific lane's determinism is measured, not
assumed: run-to-run tolerance and nondeterministic kernel sources are recorded at S00-B
[AUTH: 01 §30].
