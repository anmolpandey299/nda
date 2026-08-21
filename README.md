# privacy-model-merging

Reproducible repository for the study **Privacy Leakage / Recoverability in Ordinary
Model-Merge Release Families**.

## Authority

This repository implements, and never overrides, the four documents in [`specs/`](specs/):

| Document | Role |
|---|---|
| `00_MEASUREMENT_SPEC_v1.9_FINAL_CLOSED.md` | scientific measurement constitution |
| `01_EXECUTION_STACK_LOCK_v2.md` | engineering / execution authority |
| `02_PRE_EXECUTION_IMPLEMENTATION_CORRECTIONS.md` | binding implementation clarifications |
| `03_REVIEW_GOVERNANCE_LOCK.md` | binding review process |

Their SHA256 values are frozen in [`specs/SPEC_HASHES.json`](specs/SPEC_HASHES.json).

## Current stage

```text
STAGE                 = S00  (repository bootstrap)
BACKEND_INTEGRATED    = FALSE
SUITE_SCOPE           = STATISTICAL_STACK_ONLY
P0_PRE_READY          = FALSE
S00_ENV_CAPTURE       = TBD_REQUIRES_HARDWARE
```

No research model has been run. No scientific result exists. See
[`artifacts/p0_pre/P0_PRE_READINESS.json`](artifacts/p0_pre/P0_PRE_READINESS.json).

## Quick start (CPU/dev lane)

```bash
uv sync --extra cpu-dev     # reproducible from the committed uv.lock
make preflight              # ordered code-quality gate + readiness evaluation
```

`make preflight` is the whole CPU gate. It never runs the GPU or backend lanes.

## Layout

`specs/` authority copies · `configs/` all scientific constants · `src/` implementation
(one package per stage) · `tests/` six lanes · `manifests/` model/data/environment/run
provenance · `results/` result tables · `artifacts/` runs, cache, P0-PRE gate artifacts ·
`reviews/` per-stage blind reviews · `stage_acceptance/` per-stage acceptance bundles ·
`logs/` per-run logs · `scripts/` invariant checker, environment capture, hook installer.

Operational detail for contributors and agents is in [`CLAUDE.md`](CLAUDE.md).
The accepted S00 architecture is [`stage_acceptance/S00/01_PLAN.md`](stage_acceptance/S00/01_PLAN.md).
