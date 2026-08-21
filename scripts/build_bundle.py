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

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
HEAD_SHA_RE = re.compile(r"^head git commit\s*=\s*([0-9a-f]{40})$", re.MULTILINE)
SELF_REFERENTIAL = (
    "00_INDEX.md",
    "02_DIFF.patch",
    "02_CHANGED_FILES.txt",
    "05_ARTIFACT_MANIFEST.json",
)


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
    """Paths that changed between the described commit and HEAD, excluding the stage's own
    bundle and review directories. A non-empty list means the bundle is stale."""
    head = git(root, "rev-parse", "HEAD").strip()
    if described == head:
        return []
    names = git(root, "diff", "--name-only", described, head).split()
    allowed = (f"stage_acceptance/{stage}/", f"reviews/{stage}/")
    return sorted(p for p in names if not p.startswith(allowed))


def artifact_manifest(root: Path, described: str, stage: str) -> dict[str, object]:
    self_paths = {f"stage_acceptance/{stage}/{n}" for n in SELF_REFERENTIAL}
    tracked = set(git(root, "ls-files").split())
    untracked = set(git(root, "ls-files", "--others", "--exclude-standard").split())
    files = sorted(f for f in tracked | untracked if (root / f).is_file() and f not in self_paths)
    return {
        "stage": stage,
        "authority": "01 §16, §26(13), §45",
        "described_commit": described,
        "note": (
            "File hashes describe the tree at `described_commit`. A manifest cannot hash "
            "itself or the commit containing it, so the four self-referential bundle files "
            "are excluded and `build_bundle.py --verify` instead proves zero code drift "
            "between described_commit and HEAD."
        ),
        "self_referential_exclusions": sorted(self_paths),
        "environment_lock_sha256": "TBD_REQUIRES_HARDWARE",
        "file_count": len(files),
        "files": {f: sha256_file(root / f) for f in files},
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


def verify(root: Path, stage: str) -> list[str]:
    """Exact-value verification, not presence checking."""
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
            f"bundle is stale; code changed after {described[:12]}: "
            + ", ".join(drift[:6])
            + (" ..." if len(drift) > 6 else "")
        )

    expected_diff = diff_text(root, described)
    if (bundle / "02_DIFF.patch").read_text(encoding="utf-8") != expected_diff:
        problems.append(f"02_DIFF.patch does not equal `git diff <empty-tree> {described[:12]}`")
    expected_names = changed_files_text(root, described)
    if (bundle / "02_CHANGED_FILES.txt").read_text(encoding="utf-8") != expected_names:
        problems.append("02_CHANGED_FILES.txt does not equal the diff's --name-only output")

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        problems.append("05_ARTIFACT_MANIFEST.json lists no files")
    else:
        for rel, digest in sorted(files.items()):
            path = root / rel
            if not path.is_file():
                problems.append(f"manifest lists a missing artifact: {rel}")
            elif sha256_file(path) != digest:
                problems.append(f"manifest hash mismatch: {rel}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--stage", default="S00")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    if args.verify:
        problems = verify(root, args.stage)
        for p in problems:
            print(p, file=sys.stderr)
        print(f"bundle-verify: {len(problems)} problem(s)", file=sys.stderr)
        return 1 if problems else 0

    described = generate(root, args.stage)
    print(f"bundle regenerated for {args.stage} describing {described}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
