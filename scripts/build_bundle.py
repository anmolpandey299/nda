#!/usr/bin/env python3
"""Generate and verify a Stage Acceptance Bundle's derived artifacts.

The bundle must describe the current code exactly [AUTH: 01 §26, §45]. A committed file
cannot contain the SHA256 of the commit that contains it, so the bundle records the commit it
*describes* and `--verify` proves there is **zero code drift** between that commit and HEAD:
every path that differs must live inside the stage's own bundle or review directory.

CLI
    python scripts/build_bundle.py --root . --stage S00 [--verify]
Exit
    0 = generated, or verified consistent; 1 = drift or mismatch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflight import (  # noqa: E402
    EVIDENCE_LANES_REL,
    READINESS_PRODUCER,
    READINESS_REL,
    TBD,
    compute_readiness,
    environment_lock_sha256,
    is_environment_lock_manifest,
    validate_lane_evidence,
)

IMAGE_RECORD_REL = "manifests/environments/S00B_IMAGE_RECORD.json"
GPU_LANE = "gpu_smoke"

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
HEAD_SHA_RE = re.compile(r"^head git commit\s*=\s*([0-9a-f]{40})$", re.MULTILINE)
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SELF_REFERENTIAL = (
    "00_INDEX.md",
    "02_DIFF.patch",
    "02_CHANGED_FILES.txt",
    "05_ARTIFACT_MANIFEST.json",
)

#: Artifacts that EXECUTION produces, not artifacts that define the source.
#:
#: `P0_PRE_READINESS.json` is written by preflight step 8 [plan §11] and the environment
#: manifest is written by capture on the H100 [plan §5.6]. Both are expected to change the
#: moment S00-B runs for real, so hash-binding them to a pre-hardware bundle would make a
#: correct bootstrap structurally incapable of passing. They keep their provenance — they are
#: listed, hashed at bundle time, and verified by RE-DERIVATION rather than by a frozen hash,
#: which is a stronger check than the hash ever was [AUTH: 01 §16; 02 §C6].
RUNTIME_EVIDENCE_EXACT: tuple[str, ...] = (
    "artifacts/p0_pre/P0_PRE_READINESS.json",
    "manifests/environments/S00B_IMAGE_RECORD.json",
)
RUNTIME_EVIDENCE_PREFIXES: tuple[str, ...] = ("artifacts/p0_pre/evidence/",)
ENV_MANIFEST_DIR = "manifests/environments/"
SHA256_NAME = re.compile(r"^[0-9a-f]{64}\.json$")


def is_runtime_evidence(rel: str) -> bool:
    """True for artifacts produced by execution rather than committed as source."""
    if rel in RUNTIME_EVIDENCE_EXACT:
        return True
    if any(rel.startswith(prefix) for prefix in RUNTIME_EVIDENCE_PREFIXES):
        return True
    if rel.startswith(ENV_MANIFEST_DIR):
        return bool(SHA256_NAME.match(rel[len(ENV_MANIFEST_DIR) :]))
    return False


def git(root: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diff_text(root: Path, commit: str) -> str:
    return git(root, "diff", EMPTY_TREE, commit)


def changed_files_text(root: Path, commit: str) -> str:
    return git(root, "diff", "--name-only", EMPTY_TREE, commit)


def code_drift(root: Path, described: str, stage: str) -> list[str]:
    """Source paths that changed between the described commit and HEAD, plus any NEW
    untracked source file.

    An untracked `scripts/*.py` would otherwise evade verification entirely: it is in no
    diff, and the manifest only hashes what it already listed. The stage's own bundle and
    review directories are excluded because they carry the bundle, and runtime evidence is
    excluded because producing it is the point of S00-B.
    """
    allowed = (f"stage_acceptance/{stage}/", f"reviews/{stage}/")

    def relevant(path: str) -> bool:
        return not path.startswith(allowed) and not is_runtime_evidence(path)

    head = git(root, "rev-parse", "HEAD").strip()
    drifted: set[str] = set()
    if described != head:
        drifted.update(
            p for p in git(root, "diff", "--name-only", described, head).split() if relevant(p)
        )
    for path in git(root, "ls-files", "--others", "--exclude-standard").split():
        if relevant(path) and (root / path).is_file():
            drifted.add(f"{path} (untracked)")
    return sorted(drifted)


def artifact_manifest(root: Path, described: str, stage: str) -> dict[str, object]:
    self_paths = {f"stage_acceptance/{stage}/{n}" for n in SELF_REFERENTIAL}
    tracked = set(git(root, "ls-files").split())
    untracked = set(git(root, "ls-files", "--others", "--exclude-standard").split())
    files = sorted(f for f in tracked | untracked if (root / f).is_file() and f not in self_paths)
    source = [f for f in files if not is_runtime_evidence(f)]
    runtime = [f for f in files if is_runtime_evidence(f)]
    return {
        "stage": stage,
        "authority": "01 §16, §26(13), §45",
        "described_commit": described,
        "note": (
            "source_artifacts are hash-bound: any change is source drift. runtime_evidence is "
            "produced by execution (preflight step 8, H100 capture, the image build), so it is "
            "recorded with its hash at bundle time but verified by re-derivation, not by hash "
            "equality; otherwise a correct S00-B run would invalidate its own bundle."
        ),
        "self_referential_exclusions": sorted(self_paths),
        "source_artifact_count": len(source),
        "runtime_evidence_count": len(runtime),
        "source_artifacts": {f: sha256_file(root / f) for f in source},
        "runtime_evidence": {f: sha256_file(root / f) for f in runtime},
    }


def index_text(root: Path, described: str, stage: str) -> str:
    return f"""# Stage Acceptance Bundle — {stage}

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
STAGE_ID = {stage}
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
base git commit = {EMPTY_TREE}
```

{stage} is the root commit of the repository; the base is git's empty tree.

## 5. Head git commit

```text
head git commit = {described}
```

`make bundle-verify` fails if any path outside `stage_acceptance/{stage}/` or
`reviews/{stage}/` differs between this commit and HEAD, so the bundle can never describe
stale code.

## 20. Requested verdict

```text
requested verdict = ACCEPT
```

Requested on the evidence in this bundle, subject to the open item in `09_UNRESOLVED.md`:
S00-B environment capture has not run, so `ENVIRONMENT_LOCK_SHA256 = TBD_REQUIRES_HARDWARE`
and plan §19 makes S00-B evidence an acceptance criterion. F1 is explicitly not an acceptance
branch [AUTH: plan §17 F1, §19].
"""


def generate(root: Path, stage: str) -> str:
    described = git(root, "rev-parse", "HEAD").strip()
    bundle = root / "stage_acceptance" / stage
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "00_INDEX.md").write_text(index_text(root, described, stage), encoding="utf-8")
    (bundle / "02_DIFF.patch").write_text(diff_text(root, described), encoding="utf-8")
    (bundle / "02_CHANGED_FILES.txt").write_text(
        changed_files_text(root, described), encoding="utf-8"
    )
    (bundle / "05_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(artifact_manifest(root, described, stage), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return described


def _verify_runtime_evidence(root: Path, listed: dict[str, str]) -> list[str]:
    """Runtime evidence is verified by RE-DERIVATION, never by a frozen hash.

    A hash would be both too weak (a forged readiness file hashes fine if you re-hash it) and
    too strong (a correct S00-B run legitimately changes it). Re-deriving is what actually
    catches forgery [AUTH: 02 §C6; 01 §16].
    """
    problems: list[str] = []
    for rel in sorted(listed):
        if not (root / rel).is_file():
            problems.append(f"runtime evidence recorded but now absent: {rel}")

    readiness_path = root / READINESS_REL
    if readiness_path.is_file():
        try:
            stored = json.loads(readiness_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return problems + [f"{READINESS_REL} is not valid JSON ({exc.msg})"]
        producer = stored.get("computed_by")
        if producer != READINESS_PRODUCER:
            problems.append(
                f"{READINESS_REL} claims producer {producer!r}; only {READINESS_PRODUCER!r}"
                " may author it, so re-derivation cannot be steered by the stored value"
            )
            return problems
        recomputed = compute_readiness(root, computed_by=READINESS_PRODUCER)
        if stored != recomputed:
            differing = sorted(
                key
                for key in set(stored) | set(recomputed)
                if stored.get(key) != recomputed.get(key)
            )
            problems.append(
                f"{READINESS_REL} does not match its re-derivation; differing keys: "
                + ", ".join(differing)
            )

    env_dir = root / "manifests" / "environments"
    if env_dir.is_dir():
        for path in sorted(env_dir.glob("*.json")):
            if not is_environment_lock_manifest(path):
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            identity = environment_lock_sha256(manifest)
            if identity != manifest.get("environment_lock_sha256"):
                problems.append(
                    f"{path.name}: stored identity does not equal the recomputed identity"
                )
            elif identity != path.stem:
                problems.append(f"{path.name}: filename does not equal its identity")
    return problems


def verify_source(root: Path, stage: str) -> list[str]:
    """The IMMUTABLE half: described commit, source drift, diff, and source artifact hashes.

    Safe to run at any point in the ordered gate, because nothing it inspects is produced by
    execution. This is what preflight step 5 may use; the runtime half cannot be evaluated
    until step 8 has re-derived readiness for the current environment.
    """
    problems: list[str] = []
    bundle = root / "stage_acceptance" / stage
    index_path = bundle / "00_INDEX.md"
    manifest_path = bundle / "05_ARTIFACT_MANIFEST.json"
    if not index_path.is_file() or not manifest_path.is_file():
        return [f"{stage}: bundle is missing 00_INDEX.md or 05_ARTIFACT_MANIFEST.json"]

    match = HEAD_SHA_RE.search(index_path.read_text(encoding="utf-8"))
    if not match:
        return ["00_INDEX.md does not record an exact 40-hex head git commit"]
    described = match.group(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("described_commit") != described:
        problems.append(
            f"05_ARTIFACT_MANIFEST.json describes {manifest.get('described_commit')} "
            f"but 00_INDEX.md records {described}"
        )

    try:
        git(root, "cat-file", "-e", f"{described}^{{commit}}")
    except RuntimeError:
        return problems + [f"described commit {described} does not exist"]

    drift = code_drift(root, described, stage)
    if drift:
        problems.append(
            f"bundle is stale; source changed after {described[:12]}: "
            + ", ".join(drift[:6])
            + (" ..." if len(drift) > 6 else "")
        )

    expected_diff = diff_text(root, described)
    if (bundle / "02_DIFF.patch").read_text(encoding="utf-8") != expected_diff:
        problems.append(f"02_DIFF.patch does not equal `git diff <empty-tree> {described[:12]}`")
    expected_names = changed_files_text(root, described)
    if (bundle / "02_CHANGED_FILES.txt").read_text(encoding="utf-8") != expected_names:
        problems.append("02_CHANGED_FILES.txt does not equal the diff's --name-only output")

    source = manifest.get("source_artifacts")
    if not isinstance(source, dict) or not source:
        problems.append("05_ARTIFACT_MANIFEST.json lists no source_artifacts")
    else:
        for rel, digest in sorted(source.items()):
            path = root / rel
            if not path.is_file():
                problems.append(f"manifest lists a missing source artifact: {rel}")
            elif sha256_file(path) != digest:
                problems.append(f"source artifact hash mismatch: {rel}")
    return problems


CLOSURE_REL_TEMPLATE = "stage_acceptance/{stage}/12_S00B_CLOSURE.json"


def _selected_environment(root: Path) -> tuple[Path | None, dict[str, object]]:
    env_dir = root / "manifests" / "environments"
    manifests = (
        [p for p in sorted(env_dir.glob("*.json")) if is_environment_lock_manifest(p)]
        if env_dir.is_dir()
        else []
    )
    if not manifests:
        return None, {}
    return manifests[0], json.loads(manifests[0].read_text(encoding="utf-8"))


#: image_record_state is an OBSERVATION-TIME repository fact, not a property of the closure
#: record, and the official lifecycle changes it exactly once: committing the post-build image
#: record as the closure evidence commit H turns PENDING_COMMIT into CONSISTENT. Strict
#: equality across B -> H is therefore invalid. Exactly that one transition is permitted, and
#: only when every stable authority still agrees [F02 adjudication].
PERMITTED_IMAGE_STATE_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {("PENDING_COMMIT", "CONSISTENT")}
)


def _verify_image_state_continuity(
    root: Path,
    record: dict[str, object],
    env_manifest: dict[str, object],
    build_commit: str,
    current_state: str,
) -> list[str]:
    """Compare the stored image state with the recomputed one, allowing only the single
    expected post-evidence-commit transition."""
    stored_state = record.get("image_record_state")
    if stored_state == current_state:
        return []
    if (stored_state, current_state) not in PERMITTED_IMAGE_STATE_TRANSITIONS:
        return [
            f"closure image_record_state {stored_state!r} != the recomputed "
            f"{current_state!r}, and that transition is not part of the lifecycle"
        ]

    # PENDING_COMMIT -> CONSISTENT is only legitimate when the record that was committed
    # describes exactly the image the closure was made against.
    problems: list[str] = []
    record_path = root / IMAGE_RECORD_REL
    if not record_path.is_file():
        return ["closure claims the image record was committed, but it is absent"]
    try:
        committed = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{IMAGE_RECORD_REL} is not valid JSON ({exc.msg})"]

    baked_digest = env_manifest.get("docker_image_digest")
    baked_commit = env_manifest.get("image_source_git_commit")
    if committed.get("sealed_image_digest") != baked_digest:
        problems.append("the committed image record does not describe the running image")
    if committed.get("source_git_commit") != baked_commit:
        problems.append("the committed image record disagrees about the image source commit")
    if baked_commit != build_commit:
        problems.append("the baked image source commit is not the build commit")
    if record.get("sealed_image_digest") != baked_digest:
        problems.append("the closure record's sealed digest is not the running image's")
    if record.get("build_commit") != build_commit:
        problems.append("the closure record's build commit is not the derived build commit")
    return problems


def verify_closure_record(root: Path, stage: str) -> list[str]:
    """Re-verify an existing closure record. Never regenerate or mutate evidence.

    A closure record is a set of CLAIMS. Every summary field is recomputed from its
    authority - the bundle index, the Git graph, the validated environment manifest, the
    canonically validated lane evidence and the readiness re-derivation - so a forged
    summary cannot survive even when the artifact hashes happen to match
    [AUTH: 01 §16, §26(13); 02 §C6].
    """
    path = root / CLOSURE_REL_TEMPLATE.format(stage=stage)
    if not path.is_file():
        return []
    problems: list[str] = []
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"12_S00B_CLOSURE.json is not valid JSON ({exc.msg})"]
    if not isinstance(record, dict):
        return ["12_S00B_CLOSURE.json is not a JSON object"]

    head = git(root, "rev-parse", "HEAD").strip()
    authoritative_c = described_commit(root, stage)
    authoritative_b = bundle_build_commit(root, stage)

    # A / B — commit identities recomputed from their authorities, not read from the record.
    if record.get("science_described_commit") != authoritative_c:
        problems.append(
            "closure science_described_commit "
            f"{str(record.get('science_described_commit'))[:12]} != the bundle's "
            f"{authoritative_c[:12]}"
        )
    if record.get("build_commit") != authoritative_b:
        problems.append(
            f"closure build_commit {str(record.get('build_commit'))[:12]} != the commit that "
            f"contains the bundle {authoritative_b[:12]}"
        )

    # C — the C -> B -> H graph.
    if (
        authoritative_c
        and authoritative_b
        and not _is_ancestor(root, authoritative_c, authoritative_b)
    ):
        problems.append("the science commit is not an ancestor of the build commit")
    if authoritative_b and not _is_ancestor(root, authoritative_b, head):
        problems.append("the build commit is not an ancestor of the current closure HEAD")

    # F / G — environment and image identity from their authorities.
    manifest_path, env_manifest = _selected_environment(root)
    identity = environment_lock_sha256(env_manifest) if env_manifest else TBD
    if record.get("environment_lock_sha256") != identity:
        problems.append("closure environment identity is no longer the selected environment")
    if env_manifest:
        if record.get("sealed_image_digest") != env_manifest.get("docker_image_digest"):
            problems.append("closure sealed image digest is no longer current")
        if record.get("image_source_git_commit") != env_manifest.get("image_source_git_commit"):
            problems.append("closure image source commit is no longer current")
        state, image_problems = image_identity_state(root, env_manifest, authoritative_b)
        if state not in ACCEPTED_IMAGE_STATES:
            problems.extend(image_problems)
            problems.append(f"image identity state is {state}")
        else:
            problems.extend(
                _verify_image_state_continuity(root, record, env_manifest, authoritative_b, state)
            )

    if record.get("s00b_complete"):
        # D — readiness summary against the authoritative re-derivation.
        recomputed = compute_readiness(root, computed_by=READINESS_PRODUCER)
        expected_readiness = {
            key: recomputed.get(key)
            for key in ("BACKEND_INTEGRATED", "SUITE_SCOPE", "P0_PRE_READY")
        }
        if record.get("readiness") != expected_readiness:
            problems.append(
                f"closure readiness summary {record.get('readiness')} != the authoritative "
                f"re-derivation {expected_readiness}"
            )
        # E — GPU summary against the canonically validated lane evidence.
        gpu_record, gpu_problems = validate_lane_evidence(root, GPU_LANE)
        problems.extend(f"closure: {p}" for p in gpu_problems)
        if gpu_record is not None and record.get("gpu_smoke") != gpu_record["observed"]:
            problems.append("closure gpu_smoke summary does not match the validated lane evidence")

    # H — every hashed artifact still exists and still hashes to the recorded value.
    produced = record.get("produced_artifact_sha256")
    if not isinstance(produced, dict) or not produced:
        problems.append("closure record hashes no produced artifacts")
    else:
        for rel, digest in sorted(produced.items()):
            target = root / str(rel)
            if not target.is_file():
                problems.append(f"closure artifact is missing: {rel}")
            elif sha256_file(target) != digest:
                problems.append(f"closure artifact changed after closure: {rel}")
        for required_rel in (f"{EVIDENCE_LANES_REL}/{GPU_LANE}.json", READINESS_REL):
            if required_rel not in produced:
                label = "GPU lane evidence" if GPU_LANE in required_rel else "readiness"
                problems.append(f"closure record does not bind the {label} it consumed")
        if manifest_path is not None:
            env_rel = manifest_path.relative_to(root).as_posix()
            if env_rel not in produced:
                problems.append("closure record does not bind the environment manifest")
    return problems


def verify_runtime(root: Path, stage: str, *, check_closure: bool = True) -> list[str]:
    """The RUNTIME half: readiness re-derivation, environment identity, image identity,
    lane evidence and any existing closure record.

    Only meaningful once readiness has been written for the runtime evidence in play, i.e.
    after preflight step 8. Running it earlier is a circular ordering dependency.
    """
    manifest_path = root / "stage_acceptance" / stage / "05_ARTIFACT_MANIFEST.json"
    if not manifest_path.is_file():
        return [f"{stage}: bundle is missing 05_ARTIFACT_MANIFEST.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = manifest.get("runtime_evidence")
    if not isinstance(runtime, dict):
        return ["05_ARTIFACT_MANIFEST.json has no runtime_evidence section"]
    problems = _verify_runtime_evidence(root, runtime)

    # Lane evidence is validated whenever it exists; REQUIRING it is closure's job.
    if (root / EVIDENCE_LANES_REL / f"{GPU_LANE}.json").is_file():
        problems.extend(validate_lane_evidence(root, GPU_LANE)[1])

    # Image identity is checked here too, so tampering is caught by standalone verification
    # rather than only at closure.
    _, env_manifest = _selected_environment(root)
    if env_manifest:
        build_commit = bundle_build_commit(root, stage)
        state, image_problems = image_identity_state(root, env_manifest, build_commit)
        if state not in ACCEPTED_IMAGE_STATES:
            problems.extend(image_problems)
            problems.append(f"image identity state is {state}")

    if check_closure:
        problems.extend(verify_closure_record(root, stage))
    return problems


def verify(root: Path, stage: str) -> list[str]:
    """Full closure verification: immutable source AND runtime evidence.

    This is the FINAL gate. The bootstrap runs it after capture, the GPU lane and the
    complete preflight, never inside the ordered gate itself.
    """
    problems = verify_source(root, stage)
    if any("missing 00_INDEX.md" in p or "does not exist" in p for p in problems):
        return problems
    return problems + verify_runtime(root, stage)


DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
#: States in which S00-B may complete. Everything else fails closed.
ACCEPTED_IMAGE_STATES = frozenset({"CONSISTENT", "PENDING_COMMIT"})


def bundle_build_commit(root: Path, stage: str) -> str:
    """B — the commit that CONTAINS the acceptance bundle, derived from the Git graph.

    Three commit roles must stay distinct [F02]:

        C  science described commit   the candidate the bundle describes
        B  bundle/build commit        the commit the sealed image is permanently built from
        H  closure evidence commit    a later commit recording post-build H100 evidence

    B is the commit that introduced the current artifact manifest, so it survives H: the
    closure commit records evidence under other paths and never re-dates the bundle. Using
    HEAD here would make the image look stale the moment closure is committed, and would
    demand a rebuild that changes nothing.
    """
    manifest_rel = f"stage_acceptance/{stage}/05_ARTIFACT_MANIFEST.json"
    result = subprocess.run(
        ["git", "-C", str(root), "log", "-1", "--format=%H", "--", manifest_rel],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _is_ancestor(root: Path, older: str, newer: str) -> bool:
    if older == newer:
        return True
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", older, newer],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def described_commit(root: Path, stage: str) -> str:
    """C — read from the authoritative bundle index."""
    index_path = root / "stage_acceptance" / stage / "00_INDEX.md"
    if not index_path.is_file():
        return ""
    match = HEAD_SHA_RE.search(index_path.read_text(encoding="utf-8"))
    return match.group(1) if match else ""


def closure_artifact_paths(root: Path, stage: str) -> list[str]:
    """The exact artifacts a closure commit must preserve, derived from the repository.

    One source of truth for the bootstrap's printed instructions and the runbook, so the two
    cannot drift apart [F05]. A fresh clone containing these can run verify_source,
    verify_runtime and verify_closure_record.
    """
    manifest_path, _ = _selected_environment(root)
    paths = [
        manifest_path.relative_to(root).as_posix()
        if manifest_path
        else "manifests/environments/<ENVIRONMENT_LOCK_SHA256>.json",
        IMAGE_RECORD_REL,
        READINESS_REL,
        f"{EVIDENCE_LANES_REL}/{GPU_LANE}.json",
        CLOSURE_REL_TEMPLATE.format(stage=stage),
    ]
    return paths


def _is_commit(root: Path, sha: object) -> bool:
    if not isinstance(sha, str) or not FULL_SHA_RE.match(sha):
        return False
    result = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def image_identity_state(
    root: Path, env_manifest: dict[str, object], build_commit: str
) -> tuple[str, list[str]]:
    """Bind the sealed image to the commit it was built from, fail-closed.

    `S00B_IMAGE_RECORD.json` is written BY the build, so it cannot exist inside the commit
    that produced the image; requiring it to would be an endless build/edit/commit/rebuild
    loop. The authoritative identity is the one baked into /etc/pmm-image.json, which capture
    copied into the environment manifest from inside the image.

    States:
      CONSISTENT       a committed record describes exactly this image
      PENDING_COMMIT   the NARROW legitimate case: the baked identity is valid and from the
                       correct build commit, and no repository-side record exists yet. It is
                       committed with the closure evidence, which is why there is no loop.
      PENDING_REBUILD  the image was built from a different commit than the one being closed
      CONFLICTING      a record already exists and describes a DIFFERENT image
      TAMPERED         malformed or contradictory identity anywhere
    """
    problems: list[str] = []
    digest = env_manifest.get("docker_image_digest")
    source_commit = env_manifest.get("image_source_git_commit")

    if not isinstance(digest, str) or not DIGEST_RE.match(digest):
        problems.append(f"baked image digest is malformed: {digest!r}")
        return "TAMPERED", problems
    if not _is_commit(root, source_commit):
        problems.append(
            f"baked image source commit is malformed or is not a commit: {source_commit!r}"
        )
        return "TAMPERED", problems
    if source_commit != build_commit:
        problems.append(
            f"the image was built from {str(source_commit)[:12]} but closure is for "
            f"{build_commit[:12]}; rebuild from the build commit"
        )
        return "PENDING_REBUILD", problems

    record_path = root / IMAGE_RECORD_REL
    if not record_path.is_file():
        # The only legitimate pending state: this image exists, its identity is valid and
        # matches the build commit, and the post-build record is not committed yet.
        return "PENDING_COMMIT", problems
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"{IMAGE_RECORD_REL} is not valid JSON ({exc.msg})")
        return "TAMPERED", problems
    if not isinstance(record, dict):
        problems.append(f"{IMAGE_RECORD_REL} is not a JSON object")
        return "TAMPERED", problems

    record_digest = record.get("sealed_image_digest")
    record_commit = record.get("source_git_commit")
    if not isinstance(record_digest, str) or not DIGEST_RE.match(record_digest):
        problems.append(f"{IMAGE_RECORD_REL} digest is malformed: {record_digest!r}")
        return "TAMPERED", problems
    if not _is_commit(root, record_commit):
        problems.append(
            f"{IMAGE_RECORD_REL} source commit is malformed or is not a commit: {record_commit!r}"
        )
        return "TAMPERED", problems
    if record_digest != digest:
        problems.append(
            f"{IMAGE_RECORD_REL} describes {record_digest[:19]} but the running image is "
            f"{digest[:19]}; update the record, it is not a pending one"
        )
        return "CONFLICTING", problems
    if record_commit != source_commit:
        problems.append(
            f"{IMAGE_RECORD_REL} claims this image but a different source commit: "
            f"{record_commit!r} != {source_commit!r}"
        )
        return "TAMPERED", problems
    return "CONSISTENT", problems


def closure_problems(record: dict[str, object]) -> list[str]:
    """Typed accessor for the closure record's problem list."""
    problems = record.get("problems")
    return [str(p) for p in problems] if isinstance(problems, list) else []


def closure_record(root: Path, stage: str) -> dict[str, object]:
    """Evidence produced by an actual S00-B hardware run, built from VERIFIED values.

    Preserves the three commit roles: the bundle keeps describing C, the image stays bound to
    B, and the record may be committed later at H without implying a rebuild [F02].
    """
    problems: list[str] = []
    head = git(root, "rev-parse", "HEAD").strip()
    science_commit = described_commit(root, stage)
    build_commit = bundle_build_commit(root, stage)

    if not science_commit:
        problems.append("the bundle records no described commit")
    elif not _is_commit(root, science_commit):
        problems.append(f"the described commit {science_commit[:12]} is not a commit")
    if not build_commit:
        problems.append("the bundle has not been committed, so there is no build commit")
    elif not _is_commit(root, build_commit):
        problems.append(f"the build commit {build_commit[:12]} is not a commit")

    if science_commit and build_commit:
        if not _is_ancestor(root, science_commit, build_commit):
            problems.append(
                f"the science commit {science_commit[:12]} is not an ancestor of the build "
                f"commit {build_commit[:12]}"
            )
        if not _is_ancestor(root, build_commit, head):
            problems.append(
                f"the build commit {build_commit[:12]} is not an ancestor of HEAD {head[:12]}"
            )

    problems.extend(verify_source(root, stage))
    problems.extend(verify_runtime(root, stage, check_closure=False))

    manifest_path, env_manifest = _selected_environment(root)
    if manifest_path is None:
        problems.append("no environment manifest; run `make env-capture` on the H100")

    image_state, image_problems = (
        image_identity_state(root, env_manifest, build_commit)
        if env_manifest
        else ("PENDING_REBUILD", ["no environment manifest to bind the image to"])
    )
    if image_state not in ACCEPTED_IMAGE_STATES:
        problems.extend(image_problems)

    gpu_record, gpu_problems = validate_lane_evidence(root, GPU_LANE)
    problems.extend(gpu_problems)

    recomputed = compute_readiness(root, computed_by=READINESS_PRODUCER)
    readiness_path = root / READINESS_REL
    stored = (
        json.loads(readiness_path.read_text(encoding="utf-8")) if readiness_path.is_file() else {}
    )
    if stored != recomputed:
        problems.append(
            "stored readiness does not equal its authoritative re-derivation; closure will"
            " not trust it"
        )
    identity = str(recomputed["environment_lock_sha256"])
    if identity == TBD:
        problems.append("the environment identity has not resolved")

    gpu_evidence_path = root / EVIDENCE_LANES_REL / f"{GPU_LANE}.json"
    produced_paths = [
        *([manifest_path] if manifest_path else []),
        readiness_path,
        gpu_evidence_path,
    ]
    produced = {
        p.relative_to(root).as_posix(): sha256_file(p) for p in produced_paths if p.is_file()
    }

    complete = bool(not problems and gpu_record is not None and identity != TBD)
    observed = gpu_record["observed"] if gpu_record else {"outcome": "ABSENT"}
    return {
        "stage": stage,
        "authority": "01 §16, §26(13); 02 §C6; plan §19",
        "science_described_commit": science_commit,
        "build_commit": build_commit,
        "closure_head_commit": head,
        "image_source_git_commit": env_manifest.get("image_source_git_commit", TBD),
        "sealed_image_digest": env_manifest.get("docker_image_digest", TBD),
        "image_record_state": image_state,
        "environment_lock_sha256": identity,
        "environment_manifests": [manifest_path.name] if manifest_path else [],
        "gpu_smoke": observed,
        "readiness": {
            key: recomputed.get(key)
            for key in ("BACKEND_INTEGRATED", "SUITE_SCOPE", "P0_PRE_READY")
        },
        "produced_artifact_sha256": produced,
        "closure_commit_artifacts": closure_artifact_paths(root, stage),
        "problems": problems,
        "s00b_complete": complete,
        "note": (
            "The image stays bound to the build commit. Committing this record at a later "
            "HEAD records evidence; it never implies a rebuild. P0 evidence eligibility is "
            "separate: the GPU lane may be a genuine S00-B PASS while readiness still "
            "classifies it NON_EVIDENTIARY because no S01 RUN_ID exists."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--stage", default="S00")
    ap.add_argument(
        "--closure-artifacts",
        action="store_true",
        dest="closure_artifacts",
        help="print the exact artifacts a closure commit must preserve",
    )
    ap.add_argument(
        "--verify", action="store_true", help="full closure verification: source + runtime evidence"
    )
    ap.add_argument(
        "--verify-source",
        action="store_true",
        dest="verify_source",
        help="immutable half only; safe inside the ordered gate",
    )
    ap.add_argument(
        "--closure", action="store_true", help="write the S00-B hardware closure record"
    )
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    if args.verify or args.verify_source:
        source_only = args.verify_source and not args.verify
        problems = (verify_source if source_only else verify)(root, args.stage)
        label = "bundle-verify-source" if source_only else "bundle-verify"
        for p in problems:
            print(p, file=sys.stderr)
        print(f"{label}: {len(problems)} problem(s)", file=sys.stderr)
        return 1 if problems else 0

    if args.closure_artifacts:
        for rel in closure_artifact_paths(root, args.stage):
            print(rel)
        return 0

    if args.closure:
        try:
            record = closure_record(root, args.stage)
        except RuntimeError as exc:
            print("s00b_complete = False", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            return 1
        if not record["s00b_complete"]:
            for problem in closure_problems(record):
                print(problem, file=sys.stderr)
            # Same transactional rule as capture: an incomplete run leaves no artifact that
            # could later be mistaken for closure [AUTH: 02 §C6; plan §19].
            print("s00b_complete = False", file=sys.stderr)
            print(
                "no closure record written: the environment identity has not resolved; run"
                " `make env-capture` on the H100 first",
                file=sys.stderr,
            )
            return 1
        target = root / "stage_acceptance" / args.stage / "12_S00B_CLOSURE.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"closure record written to {target.relative_to(root)}")
        print("s00b_complete = True")
        return 0

    described = generate(root, args.stage)
    print(f"bundle regenerated for {args.stage} describing {described}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
