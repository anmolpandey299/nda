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
    READINESS_REL,
    TBD,
    compute_readiness,
    environment_lock_sha256,
    is_environment_lock_manifest,
)

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
HEAD_SHA_RE = re.compile(r"^head git commit\s*=\s*([0-9a-f]{40})$", re.MULTILINE)
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
    """Source paths that changed between the described commit and HEAD.

    The stage's own bundle and review directories are excluded because they carry the
    bundle itself, and runtime evidence is excluded because producing it is the point of
    S00-B. Everything else changing means the bundle is stale.
    """
    head = git(root, "rev-parse", "HEAD").strip()
    if described == head:
        return []
    names = git(root, "diff", "--name-only", described, head).split()
    allowed = (f"stage_acceptance/{stage}/", f"reviews/{stage}/")
    return sorted(p for p in names if not p.startswith(allowed) and not is_runtime_evidence(p))


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
        recomputed = compute_readiness(root, computed_by=str(stored.get("computed_by", "")))
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


def verify_runtime(root: Path, stage: str) -> list[str]:
    """The RUNTIME half: readiness must equal its re-derivation for the current environment.

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
    return _verify_runtime_evidence(root, runtime)


def verify(root: Path, stage: str) -> list[str]:
    """Full closure verification: immutable source AND runtime evidence.

    This is the FINAL gate. The bootstrap runs it after capture, the GPU lane and the
    complete preflight, never inside the ordered gate itself.
    """
    problems = verify_source(root, stage)
    if any("missing 00_INDEX.md" in p or "does not exist" in p for p in problems):
        return problems
    return problems + verify_runtime(root, stage)


def closure_record(root: Path, stage: str) -> dict[str, object]:
    """Evidence produced by an actual S00-B hardware run, recorded deterministically.

    The pre-hardware acceptance bundle cannot contain this: it does not exist until the H100
    has run. Emitting it explicitly is what closes S00-B, instead of comparing post-run state
    to a pre-run hash [AUTH: 01 §16, §26(13); 02 §C6; plan §19].
    """
    readiness_path = root / READINESS_REL
    readiness = (
        json.loads(readiness_path.read_text(encoding="utf-8")) if readiness_path.is_file() else {}
    )
    env_dir = root / "manifests" / "environments"
    manifests = (
        [p for p in sorted(env_dir.glob("*.json")) if is_environment_lock_manifest(p)]
        if env_dir.is_dir()
        else []
    )
    produced = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in [*manifests, readiness_path]
        if p.is_file()
    }
    identity = readiness.get("environment_lock_sha256", TBD)
    return {
        "stage": stage,
        "authority": "01 §16, §26(13); 02 §C6; plan §19",
        "described_commit": git(root, "rev-parse", "HEAD").strip(),
        "environment_lock_sha256": identity,
        "environment_manifests": [p.name for p in manifests],
        "readiness": {
            key: readiness.get(key) for key in ("BACKEND_INTEGRATED", "SUITE_SCOPE", "P0_PRE_READY")
        },
        "produced_artifact_sha256": produced,
        "s00b_complete": bool(identity != TBD and manifests),
        "note": (
            "Runtime evidence from one hardware run. Commit the environment manifest, the "
            "readiness file and this record, then regenerate the bundle so the closure state "
            "is the described one."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--stage", default="S00")
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

    if args.closure:
        try:
            record = closure_record(root, args.stage)
        except RuntimeError as exc:
            print("s00b_complete = False", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            return 1
        if not record["s00b_complete"]:
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
