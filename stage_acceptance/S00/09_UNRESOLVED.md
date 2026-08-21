# Unresolved findings — S00

## Open

### U-01 — S00-B environment capture has not run
```text
severity   BLOCKING FOR STAGE ACCEPTANCE (not a review finding)
state      ENVIRONMENT_LOCK_SHA256 = TBD_REQUIRES_HARDWARE
```
No RunPod H100 SXM was available. Every §5.7 register value remains
`TBD_REQUIRES_HARDWARE`, `manifests/environments/<env_id>.json` does not exist, and
`make env-capture` exits non-zero by design. Consequence, already encoded: with no
environment lock there is no valid `RUN_ID`, so any result would be `NON_EVIDENTIARY`
[AUTH: 01 §15, §16]. Plan §19 requires S00-B evidence for acceptance and plan §17 F1 states
that F1 is not an acceptance branch. CPU-lane S01-S04 development may proceed on the
committed S00-A lock.

Closure: run `make env-capture` on the H100 image; `tests/gpu_smoke/` accepts it.

### U-02 — S00-CBR-005, deferred at review round 2
```text
ticket        S00-T01
severity      MINOR / UNJUSTIFIED_SCOPE
target stage  S01
record        reviews/S00/deferred.md
```
`.gitignore` was cited to `01 §11`, whose tree does not name it. The file now carries its own
authority note (`01 §27`, `§36`) and ignores only working-tree noise plus the regenerable
content-addressed cache; nothing evidentiary is ignored. Final disposition remains S01.

## Closed during implementation

| ID | Defect found while implementing | Fix |
|---|---|---|
| IMPL-01 | I7 and the provider check matched their own detector source | patterns built from parts; provider check narrowed to real integration idioms in `src/` and `configs/` |
| IMPL-02 | `pre-commit` hard-failed on an unborn HEAD, blocking the root commit | `git symbolic-ref --short -q` |
| IMPL-03 | `pre-commit` hard-failed in a clone without a toolchain | ruff step skipped with a warning; the authority gate is `make preflight` in CI |

## Deviations from the accepted plan

| Item | Nature | Why |
|---|---|---|
| `scripts/preflight.py` | one executable file beyond the three enumerated in plan §3 `scripts/` | Plan §10.1 mandates preflight steps 7-8 (evidence verification, readiness authorship) and plan §11.2 requires them to be the only writer of the readiness flag. That logic needs one importable, testable home. Flagged for the implementation reviewer rather than added silently [AUTH: 03 §4, §14]. |
| `software_gate` block in `P0_PRE_READINESS.json` | field beyond the plan §11 JSON sample | Plan §11.1 requires the 00 §35 / §34C.1 conjunction alongside the five §C6 evidence keys. The sample shows only the latter; both are now represented. |
| `02_CHANGED_FILES.txt` | slot beyond the plan §45 ten-file list | `01 §26` item 7 requires a changed-file list; §8.3 maps it to `02_DIFF.patch`, which is a patch, not a list. |
| `artifacts/p0_pre/evidence/` | subdirectory | Step 7 must read evidence records from somewhere that is not the file step 8 writes. |
