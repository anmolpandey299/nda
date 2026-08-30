# S09 Acceptance — Experiment Registry and Launcher

## 1. Identity and scope
```text
STAGE_ID = S09 · STAGE_NAME = EXPERIMENT_REGISTRY_ORCHESTRATOR · STATUS = ACCEPTED
BRANCH = stage/s09-registry-orchestrator
BASE_ACCEPTED_STATE           = stage-pre-s09-backend-v1
PRE_S09_ACCEPTANCE_COMMIT     = 7345497de0872a95cc394e659065299a04efb696

implemented      the P1 cell registry as version-controlled data · the P0 §35 stage machine ·
                 the registered coefficient schedules · a pure operator-gate resolver ·
                 cell-status resolution · a three-mode launcher (LIST / PLAN / EXECUTE)
not implemented  S10 benchmark · real model compatibility · model downloads · training ·
                 merge or recovery execution · privacy scoring · P0/P1 outcomes
authority        00 §8-§12, §24, §24A, §25, §34C.1A, §35, §36, §42 ·
                 01 §15-§19, §27, §29-§36, §39-§43 · 02 §C6-§C8
```

S09 encodes the registry and resolves it. **The launcher does not invent cells**: a caller names
a registered cell id, and the scientific combination comes from the row.

## 2. The registry
```text
19 calibration rows (00 §36.1) + 10 representative-DP rows (00 §36.2) = 29 total
4 STRUCTURAL_NA calibration rows: C04 · C07 · C13 · C16
```
Every row is compared field by field against an independent transcription of the spec's own
table, so drift in either the config or the spec shows up rather than resolving quietly. The
status vocabulary is 00 §36's closed eight-value enum; `ACTIVE`, `ENABLED`, `TODO` and `AUTO`
are rejected at load.

Absent by construction and asserted absent: hidden-alpha lineage, k = 8, TIES, SLERP, a
quantization operator, a lossy duplicate of C09W, and incomplete-lineage lossy DP cells.

## 3. Coefficient schedules
```text
fixed            alpha_i = 0.50 for every one of a cell's k descendants
k2-reference     [0.35, 0.65]
spread-matched   [0.35, 0.45, 0.55, 0.65]
wide-spread      [0.25, 0.40, 0.60, 0.75]
```

### The fixed coefficient is a pre-result design completion
00 §36 registers the fixed-alpha lineage but **assigns it no numeric coefficient**, and no
accepted config supplied one — `configs/merge/operators.json` states that the registered
coefficient schedule is S09's. `alpha = 0.50` is therefore completed here rather than taken
from the closed measurement spec, and the config records that provenance explicitly as
`PRE_RESULT_DESIGN_COMPLETION`.

```text
FIXED_ALPHA = 0.50 · PRE_RESULT_DESIGN_COMPLETION · NOT from the closed measurement spec
```

Rationale: symmetric equal-parent weighting; a neutral baseline; privileges neither the
protected constituent nor the partner; avoids the coefficient amplification an extreme alpha
would introduce into the rescaling identity; selected **before any real scientific outcome was
opened**; and required to make the already-registered fixed-schedule cells executable.

The three varying schedules are unchanged from 00 §25 / §36.3. A registry row names a schedule;
it carries no coefficients of its own, and no caller can supply any — `resolve_schedule` and
`schedule_for_cell` have no alpha parameter, and `--alpha` at the command line is refused.
Changing the coefficient moves the schedule document, the registry identity, the plan identity
and the resolved-config identity together.

## 4. Gates and resolution
The operator resolver is pure, reads exactly the two gate statuses, and is checked by AST to
contain no notion of effect size, recovery quality, preference or availability:
```text
DARE PASS              -> DARE
DARE FAIL + O3 PASS    -> SVD_TRUNC_MERGE
DARE FAIL + O3 FAIL    -> OPERATOR_AXIS_INSUFFICIENT
otherwise              -> UNRESOLVED
```
A frozen `P1_PRIMARY_LOSSY_OPERATOR` cannot be replaced, and a document freezing an operator
its own gates do not select is refused.

Gate state is never self-asserted: there is no `--dare-pass`, `--p0-c1-pass` or `--allow-na`
option, checked against the parser's registered options rather than its prose. A gate record
must bind its source artifact, that artifact's SHA256, the deciding RUN_ID(s) and the registry
identity; fixture documents carry `NON_EVIDENTIARY_FIXTURE`.

Resolution only narrows. `STRUCTURAL_NA` is permanent under every gate state. `LOSSY` rows
resolve to the one selected operator or stay gated. C14/C15/C17/C18 are `METHOD_GATED` under
DARE and become `RUN_REPRODUCTION` only under O3 **and** a passing P0-C4. Varying-alpha cells
wait on P0-C2. A DP row needs the exact `{101, 202, 303}` set — `{101, 202, 404}` is three
seeds and is not the DP set.

## 5. The P0 stage machine
The 00 §35 sequence is data, not branching: 13 stages in the exact registered order, with
dependencies, gates and calendar classes. `P0-D2` requires `P0-D` to have **failed** — running
the fallback because it is available would select an operator by availability. `P0-D3` and
`P0-C4` require O3 to be the selected operator. A failed `P0-C1` blocks the recovery pipeline.

```text
SOFTWARE_SLIP = TRUE · CALENDAR_COMPRESSED_MVRS = ACTIVE
P1_LAUNCH_DEADLINE = 2026-09-03T18:00:00+01:00
P0-C2 · P0-D · P0-E = POST_MVRS_EXPANSION, deferred and still registered
```
The deadline is planning metadata: wall clock alone never executes or skips a cell, and no gate
is weakened for calendar reasons. Every empirical gate is UNRESOLVED — no P0 outcome has been
opened.

## 6. The launcher
`LIST` reads the registry. `PLAN` resolves one cell and is side-effect free — it claims no
attempt, writes no manifest and carries no wall-clock field, so repeating it is bitwise
identical. `EXECUTE` claims exactly one attempt through the **accepted S01 lifecycle**:
`begin_run`, `experiment_id`, `run_id` and `RunAttempt.finalize` are Block A's, used as they
stand. There is no `S09RunManifest`, no second RUN_ID algorithm and no attempt counter of S09's
own.

One invocation is one attempt — no queue, scheduler, daemon, DAG walker or next-cell recursion,
checked by AST. Commands are argument arrays; there is no shell. Terminal states are 01 §36's.
A rerun receives a new RUN_ID and a failed attempt is never rewritten.

```text
verified: C01 alpha [0.5] · C02 [0.5, 0.5] · C03 [0.5, 0.5, 0.5, 0.5] -> RUN_CONTROL, runnable
          C05 [0.5, 0.5] · C06 [0.5, 0.5, 0.5, 0.5] -> RUN_REPRODUCTION once gates permit
```

An evidentiary run requires complete provenance: an exact model revision, a resolved
environment identity, a valid data manifest and a clean production tree. Nothing is filled in
with zeros, `UNKNOWN`, `latest` or a fixture value. Evidentiary behaviour is exercised in a
disposable committed fixture repository, because this branch is necessarily dirty.

The backend compatibility command requires an externally issued RUN_ID and mints none; S09 is
the layer that supplies it, and the evidence binds it exactly — a tampered RUN_ID fails
verification.

`EXECUTE` registers no scientific task yet and says so: the registered task set arrives with
S10 and the production backend. Inventing one would be inventing an experiment.

## 7. Test gate and protected state
```text
UNIT = 1485 passed · INTEGRATION = 593 passed · SYNTHETIC = 182 passed
FORMAT = PASS (198 files) · LINT = PASS · TYPECHECK = PASS (187) · INVARIANTS = 0 violations

S00_UNCHANGED = TRUE · BLOCK_A_PROVENANCE_UNCHANGED = TRUE
BLOCK_B_SCIENCE_UNCHANGED = TRUE · BLOCK_C_SCIENCE_UNCHANGED = TRUE
S07_PRODUCTION_UNCHANGED = TRUE · S08_PRODUCTION_UNCHANGED = TRUE
PRE_S09_BACKEND_PRODUCTION_UNCHANGED = TRUE
pyproject.toml unchanged · uv.lock unchanged
```
S09 is entirely additive: no accepted file was modified.

```text
REAL_DATA = FALSE · REAL_MODEL = FALSE · TRAINING = FALSE · H100 = FALSE
PRIVACY_OUTCOMES = FALSE · REAL_MODEL_COMPATIBILITY = NOT_RUN
BACKEND_INTEGRATED = FALSE · P0_PRE_READY = FALSE · S10_IMPLEMENTED = FALSE
```

## 8. Verdict
```text
S09_STATUS=ACCEPTED
UNRESOLVED_BLOCKER=NONE
UNRESOLVED_MAJOR=NONE
CALIBRATION_ROWS=19
DP_ROWS=10
TOTAL_ROWS=29
FIXED_ALPHA=0.50
FIXED_ALPHA_PROVENANCE=PRE_RESULT_DESIGN_COMPLETION
K2_REFERENCE=[0.35, 0.65]
SPREAD_MATCHED=[0.35, 0.45, 0.55, 0.65]
WIDE_SPREAD=[0.25, 0.40, 0.60, 0.75]
HIDDEN_ALPHA_CELL=ABSENT
K8_CELL=ABSENT
TIES_CELL=ABSENT
EXTRA_DP_CELLS=ABSENT
S01_RUN_ID_LIFECYCLE_REUSED=TRUE
LAUNCHER_INVENTS_CELLS=FALSE
REAL_MODEL=FALSE
TRAINING=FALSE
H100=FALSE
PRIVACY_OUTCOMES=FALSE
NEXT_STAGE=S10_H100_BENCHMARK
```
