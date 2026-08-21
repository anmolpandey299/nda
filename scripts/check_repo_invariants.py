#!/usr/bin/env python3
"""Repository invariant checker — `make preflight` step 0.

Implements the structural halves of invariants I1-I14 of the accepted S00 architecture
(`stage_acceptance/S00/01_PLAN.md` §13).

CLI
    python scripts/check_repo_invariants.py [--root PATH] [--json]
Exit
    0 = all invariants hold; 1 = at least one violation
Out
    one line per violation: <INVARIANT_ID> <path> <reason>

[AUTH: 01 §15, §17, §18, §27; 02 §C1, §C2, §C6, §C7; 03 §9, §13; 00 §34A.2, §34A.5]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------------------
# Frozen vocabulary. These lists are authority, not preference.
# --------------------------------------------------------------------------------------

#: The eight RUN_ID inputs, verbatim and in order [AUTH: 01 §15].
RUN_ID_INPUTS: tuple[str, ...] = (
    "git_commit_sha",
    "spec_sha256",
    "execution_lock_sha256",
    "config_sha256",
    "model_revision",
    "data_manifest_sha256",
    "environment_lock_sha256",
    "training_seed",
)

#: Components of ENVIRONMENT_LOCK_SHA256 [AUTH: 01 §12(5)-(9), §15, §30, §32; plan §5.6].
ENVIRONMENT_LOCK_COMPONENTS: tuple[str, ...] = (
    "uv_lock_sha256",
    "docker_image_digest",
    "cuda_runtime",
    "cuda_driver",
    "torch_version",
    "torch_cuda_build",
    "python_version",
    "gpu_model",
)

#: Sole-owner API basenames; may exist only under src/scoring/ [AUTH: 00 §34A.2, §34A.5].
RESERVED_SINGLE_OWNER: tuple[str, ...] = ("engine", "crossfit", "roc", "cache", "pooled")
RESERVED_OWNER_DIR = "scoring"

#: Definitions that may not appear outside src/scoring/ [AUTH: 00 §34A.2(8); 02 §C2].
BANNED_OUTSIDE_SCORING = re.compile(
    r"(?i)^(?:.*_)?(?:tpr_at_fpr|fixed_fpr\w*|roc_curve|roc_interp\w*|interp_roc\w*|"
    r"make_folds|build_folds|assign_folds|fold_assignment|"
    r"calibrate_threshold|select_threshold|threshold_calibration|"
    r"cross_?fit\w*)$"
)
BOUNDARY_DIRS: tuple[str, ...] = ("analysis", "attacks")

#: Material experimental constants that must resolve from configs/** [AUTH: 01 §17].
MATERIAL_CONSTANTS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"^(?:[A-Z0-9_]*_)?TARGET_FPR$",
        r"^FPR_TARGET$",
        r"^MIN_?K(?:_FRACTION|_PERCENT|_FRAC)?$",
        r"^(?:MERGE_)?ALPHAS?$",
        r"^K$",
        r"^K_VALUES$",
        r"^NUM_PARTNERS$",
        r"^DARE_P$",
        r"^P_DARE$",
        r"^SVD_RANK$",
        r"^LORA_RANK$",
        r"^RANK$",
        r"^BOOTSTRAP_REPLICATES$",
        r"^N_BOOTSTRAP$",
        r"^BOOTSTRAP_N$",
        r"^SEEDS?$",
        r"^TRAINING_SEEDS?$",
        r"^RANDOM_SEED$",
    )
)

#: Prohibited estimator idiom [AUTH: 02 §C2, §C8].
PROHIBITED_ESTIMATOR = re.compile(r"(?i)max\s+tpr\s+at\s+repeated\s+fpr|max_tpr_at_repeated_fpr")

SPEC_FILES: tuple[str, ...] = (
    "00_MEASUREMENT_SPEC_v1.9_FINAL_CLOSED.md",
    "01_EXECUTION_STACK_LOCK_v2.md",
    "02_PRE_EXECUTION_IMPLEMENTATION_CORRECTIONS.md",
    "03_REVIEW_GOVERNANCE_LOCK.md",
)

#: Paths required by plan §3.
REQUIRED_PATHS: tuple[str, ...] = (
    "CLAUDE.md",
    "README.md",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "Dockerfile",
    ".gitignore",
    ".githooks/pre-commit",
    ".githooks/pre-push",
    ".github/workflows/ci.yml",
    "specs",
    "specs/SPEC_HASHES.json",
    "specs/deviations/SPEC_DEVIATIONS.md",
    "specs/deviations/SCORER_EXCEPTIONS.md",
    "configs/models",
    "configs/data",
    "configs/training",
    "configs/attacks",
    "configs/recovery",
    "configs/p0",
    "configs/p1",
    "src/__init__.py",
    "src/data",
    "src/models",
    "src/training",
    "src/dp",
    "src/merge",
    "src/recovery",
    "src/scoring",
    "src/attacks",
    "src/analysis",
    "src/provenance",
    "src/cli",
    "tests/unit",
    "tests/integration",
    "tests/synthetic",
    "tests/golden",
    "tests/backend_contract",
    "tests/gpu_smoke",
    "manifests/models",
    "manifests/data",
    "manifests/environments",
    "manifests/runs",
    "manifests/environments/AI_ENGINEERING_STACK_S00.json",
    "results/p0",
    "results/p1",
    "artifacts/p0_pre",
    "artifacts/runs",
    "artifacts/cache",
    "artifacts/p0_pre/P0_PRE_READINESS.json",
    "reviews/S00/scratch",
    "stage_acceptance/S00/01_PLAN.md",
    "logs",
    "scripts/check_repo_invariants.py",
    "scripts/capture_environment.sh",
    "scripts/install_git_hooks.sh",
    "scripts/preflight.py",
)

#: Paths that must NOT exist [AUTH: plan §3.1; 01 §18, §32].
FORBIDDEN_PATHS: tuple[str, ...] = ("notebooks", "serving", "STAGE_PLAN.md")

PLAN_LINE_CAP = 1200
TBD = "TBD_REQUIRES_HARDWARE"


@dataclass(frozen=True)
class Violation:
    invariant: str
    path: str
    reason: str

    def render(self) -> str:
        return f"{self.invariant} {self.path} {self.reason}"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def py_files(root: Path, sub: str) -> list[Path]:
    base = root / sub
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*.py") if ".venv" not in p.parts)


def _literal(node: ast.expr) -> bool:
    """True if the assigned value is a literal, i.e. a constant frozen in source."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        return _literal(node.operand)
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return all(_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(v is not None and _literal(v) for v in node.values)
    return False


def _module_scope_assignments(tree: ast.Module) -> list[tuple[str, ast.expr]]:
    out: list[tuple[str, ast.expr]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out.append((tgt.id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                out.append((node.target.id, node.value))
    return out


def _scorer_exception_paths(root: Path) -> set[str]:
    reg = root / "specs" / "deviations" / "SCORER_EXCEPTIONS.md"
    if not reg.is_file():
        return set()
    text = reg.read_text(encoding="utf-8")
    body = text.split("## Granted exceptions", 1)[-1] if "## Granted exceptions" in text else ""
    return set(re.findall(r"src/[\w/]+\.py", body))


# --------------------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------------------


def check_structure(root: Path) -> list[Violation]:
    """A1 / plan §3: required paths present, forbidden paths absent."""
    v = [
        Violation("I0", p, "required path missing")
        for p in REQUIRED_PATHS
        if not (root / p).exists()
    ]
    v += [
        Violation("I0", p, "forbidden path present [plan §3.1]")
        for p in FORBIDDEN_PATHS
        if (root / p).exists()
    ]
    return v


def check_single_owner(root: Path) -> list[Violation]:
    """I1 + I13: one scorer / cross-fit / fixed-FPR / cache path, enforced by API
    reservation AND an import boundary [AUTH: 00 §34A.2(8), §34A.5; 02 §C1, §C2]."""
    out: list[Violation] = []
    granted = _scorer_exception_paths(root)
    seen: dict[str, list[str]] = {}
    for f in py_files(root, "src"):
        rel = f.relative_to(root).as_posix()
        if f.stem in RESERVED_SINGLE_OWNER:
            seen.setdefault(f.stem, []).append(rel)
            if f.parent.name != RESERVED_OWNER_DIR and rel not in granted:
                out.append(
                    Violation(
                        "I1",
                        rel,
                        f"reserved module '{f.stem}' outside src/{RESERVED_OWNER_DIR}/"
                        " and no SCORER_EXCEPTION names it",
                    )
                )
    for stem, paths in seen.items():
        extra = [p for p in paths if p not in granted]
        if len(extra) > 1:
            out.append(
                Violation(
                    "I1",
                    ", ".join(sorted(extra)),
                    f"reserved module '{stem}' has {len(extra)} implementations",
                )
            )

    for sub in BOUNDARY_DIRS:
        for f in py_files(root, f"src/{sub}"):
            rel = f.relative_to(root).as_posix()
            if rel in granted:
                continue
            try:
                tree = ast.parse(f.read_text(encoding="utf-8"), filename=rel)
            except SyntaxError as exc:  # pragma: no cover - surfaced as a violation
                out.append(Violation("I1", rel, f"unparseable: {exc}"))
                continue
            names = [
                n.name
                for n in tree.body
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            ]
            names += [
                n for n, val in _module_scope_assignments(tree) if isinstance(val, ast.Lambda)
            ]
            for name in names:
                if BANNED_OUTSIDE_SCORING.match(name):
                    out.append(
                        Violation(
                            "I13",
                            rel,
                            f"'{name}' defines fold/threshold/fixed-FPR logic outside src/scoring/;"
                            " import it from src.scoring [AUTH: 00 §34A.2(8); 02 §C2]",
                        )
                    )
    return out


def check_prohibited_estimator(root: Path) -> list[Violation]:
    """A16: the scaffold's individual-record + max-at-repeated-FPR idiom stays out of src/."""
    return [
        Violation(
            "I13",
            f.relative_to(root).as_posix(),
            "prohibited 'max TPR at repeated FPR' estimator idiom [AUTH: 02 §C2, §C8]",
        )
        for f in py_files(root, "src")
        if PROHIBITED_ESTIMATOR.search(f.read_text(encoding="utf-8"))
    ]


def check_material_constants(root: Path) -> list[Violation]:
    """I14: 01 §17 constants resolve from configs/**, never from src/** module scope."""
    out: list[Violation] = []
    for f in py_files(root, "src"):
        rel = f.relative_to(root).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError:
            continue
        for name, value in _module_scope_assignments(tree):
            if _literal(value) and any(p.match(name) for p in MATERIAL_CONSTANTS):
                out.append(
                    Violation(
                        "I14",
                        rel,
                        f"material constant '{name}' is a source literal;"
                        " it must resolve from configs/** [AUTH: 01 §17]",
                    )
                )
    return out


def check_notebooks(root: Path) -> list[Violation]:
    """I2: no notebook can become an authoritative execution path [AUTH: 01 §18]."""
    out = (
        [
            Violation("I2", p.relative_to(root).as_posix(), "notebook under src/")
            for p in (root / "src").rglob("*.ipynb")
        ]
        if (root / "src").is_dir()
        else []
    )
    for rel in ["Makefile"]:
        f = root / rel
        if f.is_file() and ".ipynb" in f.read_text(encoding="utf-8"):
            out.append(Violation("I2", rel, "Makefile target references a notebook"))
    cfg = root / "configs"
    if cfg.is_dir():
        out += [
            Violation("I2", p.relative_to(root).as_posix(), "config references a notebook")
            for p in cfg.rglob("*")
            if p.is_file()
            and p.suffix in {".yaml", ".yml", ".json", ".toml"}
            and ".ipynb" in p.read_text(encoding="utf-8", errors="ignore")
        ]
    return out


def check_reviewer_isolation(root: Path) -> list[Violation]:
    """I3: reviewer output cannot become production code [AUTH: 03 §9]."""
    out: list[Violation] = []
    rev = root / "reviews"
    if rev.is_dir():
        out += [
            Violation(
                "I3",
                p.relative_to(root).as_posix(),
                "importable code under reviews/ outside scratch/",
            )
            for p in rev.rglob("*.py")
            if "scratch" not in p.relative_to(rev).parts
        ]
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        if 'packages = ["src"]' not in text:
            out.append(
                Violation("I3", "pyproject.toml", "packaging discovery is not restricted to src/")
            )
    return out


def check_specs(root: Path) -> list[Violation]:
    """I4: spec copies are the authority copies [AUTH: 01 §15]."""
    out: list[Violation] = []
    hashes_path = root / "specs" / "SPEC_HASHES.json"
    if not hashes_path.is_file():
        return [Violation("I4", "specs/SPEC_HASHES.json", "missing")]
    recorded = json.loads(hashes_path.read_text(encoding="utf-8"))
    for name in SPEC_FILES:
        copy = root / "specs" / name
        if not copy.is_file():
            out.append(Violation("I4", f"specs/{name}", "authority copy missing"))
            continue
        if name not in recorded:
            out.append(Violation("I4", f"specs/{name}", "absent from SPEC_HASHES.json"))
            continue
        actual = sha256_file(copy)
        if actual != recorded[name]["sha256"]:
            out.append(
                Violation(
                    "I4",
                    f"specs/{name}",
                    f"sha256 {actual[:12]} != recorded {recorded[name]['sha256'][:12]}",
                )
            )
        # divergence guard: a root-level original must be byte-identical to the copy
        original = root / name
        if original.is_file() and sha256_file(original) != actual:
            out.append(
                Violation("I4", name, "root-level authority document diverges from specs/ copy")
            )
    return out


def check_plan_cap(root: Path) -> list[Violation]:
    """I5: every stage plan is <= 1,200 lines [AUTH: 03 §13]."""
    plans = (
        sorted((root / "stage_acceptance").glob("*/01_PLAN.md"))
        if (root / "stage_acceptance").is_dir()
        else []
    )
    if not plans:
        return [Violation("I5", "stage_acceptance/*/01_PLAN.md", "glob is empty")]
    out: list[Violation] = []
    for p in plans:
        n = len(p.read_text(encoding="utf-8").splitlines())
        if n > PLAN_LINE_CAP:
            out.append(
                Violation(
                    "I5", p.relative_to(root).as_posix(), f"PLAN_TOO_LARGE: {n} > {PLAN_LINE_CAP}"
                )
            )
    return out


def check_readiness_authorship(root: Path) -> list[Violation]:
    """I7: only scripts/preflight.py may raise P0_PRE_READY [AUTH: 02 §C6]."""
    out: list[Violation] = []
    pat = re.compile("P0_PRE" + "_READY" + r'"?\s*[:=]\s*[Tt]rue')
    for f in list(py_files(root, "src")) + list(py_files(root, "scripts")):
        rel = f.relative_to(root).as_posix()
        if rel == "scripts/preflight.py":
            continue
        if pat.search(f.read_text(encoding="utf-8")):
            out.append(Violation("I7", rel, "raises the readiness flag outside preflight step 8"))
    return out


def check_ci(root: Path) -> list[Violation]:
    """I10: CI cannot masquerade as hardware evidence [AUTH: 01 §6.1, §49; 02 §C6]."""
    wf = root / ".github" / "workflows" / "ci.yml"
    if not wf.is_file():
        return [Violation("I10", ".github/workflows/ci.yml", "missing")]
    text = wf.read_text(encoding="utf-8")
    out: list[Violation] = []
    if "make backend-contract" in text:
        out.append(Violation("I10", "ci.yml", "CI invokes backend-contract [AUTH: 02 §C6]"))
    if "make gpu-smoke" in text:
        out.append(Violation("I10", "ci.yml", "CI invokes gpu-smoke [AUTH: 01 §6.1, §49]"))
    if re.search(r"(?i)runs-on:.*(gpu|cuda|h100)", text):
        out.append(Violation("I10", "ci.yml", "CI declares a GPU runner [AUTH: 01 §6.1, §49]"))
    if "make preflight" not in text:
        out.append(Violation("I10", "ci.yml", "CI does not invoke make preflight"))
    return out


def check_no_fabricated_hardware(root: Path) -> list[Violation]:
    """I6: hardware-derived fields are populated by S00-B or exactly TBD_REQUIRES_HARDWARE."""
    out: list[Violation] = []
    for manifest in (
        sorted((root / "manifests" / "environments").glob("*.json"))
        if (root / "manifests" / "environments").is_dir()
        else []
    ):
        obj = json.loads(manifest.read_text(encoding="utf-8"))
        for key in ENVIRONMENT_LOCK_COMPONENTS:
            if (
                key in obj
                and isinstance(obj[key], str)
                and obj[key].startswith("TBD")
                and obj[key] != TBD
            ):
                out.append(
                    Violation(
                        "I6",
                        manifest.relative_to(root).as_posix(),
                        f"'{key}' is a malformed placeholder: {obj[key]!r}",
                    )
                )
    return out


#: Actual integration, not prose mentions [AUTH: 01 §6.2, §24].
_OR = "open" + "router"
FORBIDDEN_PROVIDER = re.compile(
    r"(?i)(?:" + _OR + r"\.ai|" + _OR + r"_api_key|import\s+" + _OR + r"|from\s+" + _OR + r")"
)


def check_forbidden_provider(root: Path) -> list[Violation]:
    """Plan §3.1: the reserved API provider has no place in the engineering path.

    Credentials are reserved for scientific experiments only and no such output may select
    cells, seeds, thresholds or headline results [AUTH: 01 §6.2, §24, §38].
    """
    out: list[Violation] = []
    for sub in ("src", "configs"):
        base = root / sub
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if (
                p.is_file()
                and p.suffix in {".py", ".yaml", ".yml", ".json", ".toml", ".sh"}
                and FORBIDDEN_PROVIDER.search(p.read_text(encoding="utf-8", errors="ignore"))
            ):
                out.append(
                    Violation(
                        "I0",
                        p.relative_to(root).as_posix(),
                        "reserved-provider integration in the engineering path",
                    )
                )
    return out


def check_coverage_gate(root: Path) -> list[Violation]:
    """Plan §3.1: coverage percentage is not a correctness criterion [AUTH: 01 §33]."""
    f = root / "pyproject.toml"
    if f.is_file() and "fail_under" in f.read_text(encoding="utf-8"):
        return [Violation("I0", "pyproject.toml", "coverage threshold gate is not authorised")]
    return []


def hooks_path_warning(root: Path) -> str | None:
    """I9 is a convenience layer; a mismatch warns, it does not fail [AUTH: 03 §9; 01 §27]."""
    try:
        res = subprocess.run(
            ["git", "-C", str(root), "config", "--get", "core.hooksPath"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover
        return None
    if res.stdout.strip() != ".githooks":
        return (
            "WARN I9 core.hooksPath != .githooks — hooks are a convenience guard only; "
            "branch/tag protection and I15 are the authority mechanisms [AUTH: 03 §9]"
        )
    return None


CHECKS = (
    check_structure,
    check_single_owner,
    check_prohibited_estimator,
    check_material_constants,
    check_notebooks,
    check_reviewer_isolation,
    check_specs,
    check_plan_cap,
    check_readiness_authorship,
    check_ci,
    check_no_fabricated_hardware,
    check_forbidden_provider,
    check_coverage_gate,
)


def run_all(root: Path) -> list[Violation]:
    out: list[Violation] = []
    for check in CHECKS:
        out.extend(check(root))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="repository root to check")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    violations = run_all(root)
    warning = hooks_path_warning(root)

    if args.as_json:
        print(
            json.dumps(
                {
                    "root": str(root),
                    "violations": [v.__dict__ for v in violations],
                    "warnings": [warning] if warning else [],
                },
                indent=2,
            )
        )
    else:
        if warning:
            print(warning, file=sys.stderr)
        for v in violations:
            print(v.render())
        print(f"check_repo_invariants: {len(violations)} violation(s)", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
