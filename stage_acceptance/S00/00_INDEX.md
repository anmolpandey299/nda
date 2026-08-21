# Stage Acceptance Bundle — S00

Layout is `01 §45`; all twenty `01 §26` fields are present across these slots, mapped in
plan §8.3 [AUTH: 01 §26, §45]. Regenerate with `make bundle`; check with `make bundle-verify`.

| 01 §26 field | Slot |
|---|---|
| 1 STAGE_ID | this file |
| 2 bounded objective | this file |
| 3 controlling spec sections | this file |
| 4 base git commit | this file |
| 5 head git commit | this file |
| 6 git diff / patch | `02_DIFF.patch` |
| 7 changed-file list | `02_CHANGED_FILES.txt` |
| 8 resolved config(s) | `03A_RESOLVED_CONFIGS/` |
| 9-12 test logs | `04_TEST_OUTPUTS/` |
| 13 artifact manifest + SHA256 | `05_ARTIFACT_MANIFEST.json` |
| 14 implementer report | `06_IMPLEMENTER_REPORT.md` |
| 15 Claude reviewer report | `07_CLAUDE_REVIEW.md` |
| 16-17 independent reviewer report(s) | `08_CODEX_REVIEW.md` |
| 18 unresolved findings | `09_UNRESOLVED.md` |
| 19 real-scientific-data inspection | `11_DATA_INSPECTION_STATEMENT.md` |
| 20 requested verdict | this file |
| reproduction | `10_REPRODUCE.md` |

## 1. STAGE_ID

```text
STAGE_ID = S00
```

## 2. Bounded objective

Create the reproducible repository shell that every later stage writes into, such that from
S01 onward it is structurally impossible to produce a result without provenance, to run two
scoring / cross-fit / fixed-FPR paths, to let a notebook become the execution path, to keep a
cache valid across a scoring, backend or environment change, or to read a green synthetic
suite as backend readiness. No scientific code, no research-model execution
[AUTH: 01 §39 S00; plan §2.1].

## 3. Controlling spec sections

```text
01 §11 §12 §16 §17 §18 §22 §27 §28 §33 §35 §36 §39 S00 §45 §46
02 §C1 §C2 §C4 §C5 §C6 §C7 §C8
03 §4 §8 §9 §10 §13 §14
00 §34A.1-.5 §34B.2 §34B.3 §34C.1 §35
```

## 4. Base git commit

```text
base git commit = 4b825dc642cb6eb9a060e54bf8d69288fbee4904
```

S00 is the root commit of the repository; the base is git's empty tree.

## 5. Head git commit

```text
head git commit = bc5da88675c0e5f70fb5fee8d94ed83c8385f83a
```

`make bundle-verify` fails if any path outside `stage_acceptance/S00/` or
`reviews/S00/` differs between this commit and HEAD, so the bundle can never describe
stale code.

## 20. Requested verdict

```text
requested verdict = ACCEPT
```

Requested on the evidence in this bundle, subject to the open item in `09_UNRESOLVED.md`:
S00-B environment capture has not run, so `ENVIRONMENT_LOCK_SHA256 = TBD_REQUIRES_HARDWARE`
and plan §19 makes S00-B evidence an acceptance criterion. F1 is explicitly not an acceptance
branch [AUTH: plan §17 F1, §19].
