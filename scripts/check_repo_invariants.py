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

#: Identifier vocabulary of a ROC implementation. A module that manipulates BOTH a
#: true-positive-rate and a false-positive-rate array is implementing the estimator,
#: whatever it calls itself [AUTH: 00 §34A.2(8); 02 §C1, §C2].
TPR_NAMES: frozenset[str] = frozenset({"tpr", "tprs", "true_positive_rate", "true_positive_rates"})
FPR_NAMES: frozenset[str] = frozenset(
    {"fpr", "fprs", "false_positive_rate", "false_positive_rates"}
)

#: Importing a ROC/threshold toolkit outside src/scoring/ is a second implementation path,
#: including under an alias [AUTH: 02 §C2].
ROC_TOOLKIT_MODULES: tuple[str, ...] = ("sklearn.metrics",)
ROC_TOOLKIT_SYMBOLS: frozenset[str] = frozenset(
    {"roc_curve", "roc_auc_score", "det_curve", "precision_recall_curve", "RocCurveDisplay"}
)

#: Minimum run-manifest fields; a result without them is NON_EVIDENTIARY [AUTH: 01 §16].
RUN_MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "run_id",
    "git_commit",
    "git_dirty",
    "spec_hash",
    "execution_lock_hash",
    "model_revision",
    "tokenizer_hash",
    "data_manifest_hash",
    "environment_lock_sha256",
    "gpu_model",
    "cuda",
    "pytorch",
    "precision",
    "seeds",
    "config",
    "wall_clock_start",
    "wall_clock_end",
    "exit_code",
    "artifact_paths",
    "artifact_hashes",
    "metrics_paths",
    "stdout_log_path",
    "stderr_log_path",
)

#: Material experimental constants that must resolve from configs/** [AUTH: 01 §17].
#: Matched case-insensitively: `target_fpr` and `TARGET_FPR` are the same defect.
MATERIAL_CONSTANTS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^(?:[a-z0-9_]*_)?target_fpr$",
        r"^fpr_target$",
        r"^min_?k(?:_fraction|_percent|_frac|_pct)?$",
        r"^(?:merge_)?alphas?$",
        r"^k$",
        r"^k_values?$",
        r"^num_partners$",
        r"^dare_p$",
        r"^p_dare$",
        r"^dare_drop_rate$",
        r"^svd_rank$",
        r"^lora_rank$",
        r"^rank$",
        r"^retained_ranks?$",
        r"^bootstrap_replicates?$",
        r"^n_bootstrap$",
        r"^bootstrap_n$",
        r"^seeds?$",
        r"^training_seeds?$",
        r"^random_seed$",
        r"^dp_seeds?$",
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
    ".dockerignore",
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
    "scripts/capture_environment.py",
    "scripts/install_git_hooks.sh",
    "scripts/preflight.py",
    "scripts/build_bundle.py",
    "scripts/make_review_worktree.sh",
    "scripts/build_science_image.sh",
    "scripts/bootstrap_runpod_s00b.sh",
    "tests/backend_contract/test_backend_contract.py",
    "tests/gpu_smoke/test_cuda_torch_contract.py",
    "tests/gpu_smoke/test_env_capture_contract.py",
    "manifests/environments/S00B_HARDWARE_PROBE.json",
)

#: Paths that must NOT exist [AUTH: plan §3.1; 01 §18, §32].
FORBIDDEN_PATHS: tuple[str, ...] = (
    "notebooks",
    "serving",
    "STAGE_PLAN.md",
    # A failed environment capture must leave nothing behind; this filename was produced by
    # the earlier non-transactional implementation [AUTH: 01 §12, §16].
    "manifests/environments/TBD_REQUIRES_HARDWARE.json",
)

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


def boundary_files(root: Path) -> list[Path]:
    """Every src/ module that is NOT the sole owner. src/scoring/ owns fold construction,
    calibration-threshold logic, fixed-FPR ROC and the tie-safe TPR@1%FPR estimator; every
    other package may consume those APIs and may not implement them
    [AUTH: 00 §34A.1-.2; 02 §C1, §C2]."""
    return [f for f in py_files(root, "src") if f.parent.name != RESERVED_OWNER_DIR]


def _identifiers(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
        elif isinstance(child, ast.arg):
            names.add(child.arg)
    return names


def _roc_toolkit_imports(tree: ast.Module) -> list[str]:
    """Any import of a ROC toolkit, including aliased forms [AUTH: 02 §C2]."""
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name == m or alias.name.startswith(m + ".") for m in ROC_TOOLKIT_MODULES
                ):
                    hits.append(
                        f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                full = f"{module}.{alias.name}" if module else alias.name
                if (
                    any(
                        full == m or full.startswith(m + ".") or module == m
                        for m in ROC_TOOLKIT_MODULES
                    )
                    or alias.name in ROC_TOOLKIT_SYMBOLS
                ):
                    hits.append(
                        f"from {module} import {alias.name}"
                        + (f" as {alias.asname}" if alias.asname else "")
                    )
    return hits


def _fpr_threshold_comparisons(node: ast.AST) -> bool:
    """An FPR operating point being selected outside src/scoring/.

    Catches the direct form `fpr <= 0.01` and the indirect one `[f <= 0.01 for f in fpr]`,
    where the compared operand is a loop variable: any scope that both names an FPR array and
    compares against a numeric literal is choosing an operating point [AUTH: 02 §C1, §C2].
    """
    names_fpr = any(
        (isinstance(c, ast.Name) and c.id.lower() in FPR_NAMES)
        or (isinstance(c, ast.Attribute) and c.attr.lower() in FPR_NAMES)
        or (isinstance(c, ast.arg) and c.arg.lower() in FPR_NAMES)
        for c in ast.walk(node)
    )
    if not names_fpr:
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Compare):
            operands = [child.left, *child.comparators]
            if any(
                isinstance(o, ast.Constant)
                and isinstance(o.value, int | float)
                and not isinstance(o.value, bool)
                for o in operands
            ):
                return True
    return False


def check_single_owner(root: Path) -> list[Violation]:
    """I1 + I13: one scorer / cross-fit / fixed-FPR / cache path, enforced by API
    reservation AND by dependency analysis, not by function naming alone
    [AUTH: 00 §34A.2(8), §34A.5; 02 §C1, §C2]."""
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

    for f in boundary_files(root):
        rel = f.relative_to(root).as_posix()
        if rel in granted:
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as exc:  # pragma: no cover - surfaced as a violation
            out.append(Violation("I1", rel, f"unparseable: {exc}"))
            continue

        for stmt in _roc_toolkit_imports(tree):
            out.append(
                Violation(
                    "I13",
                    rel,
                    f"ROC toolkit imported outside src/scoring/: '{stmt}';"
                    " consume the src.scoring API instead [AUTH: 02 §C2]",
                )
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if BANNED_OUTSIDE_SCORING.match(node.name):
                    out.append(
                        Violation(
                            "I13",
                            rel,
                            f"'{node.name}' defines fold/threshold/fixed-FPR logic outside"
                            " src/scoring/ [AUTH: 00 §34A.2(8); 02 §C2]",
                        )
                    )
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                names = {n.lower() for n in _identifiers(node)}
                if names & TPR_NAMES and names & FPR_NAMES:
                    out.append(
                        Violation(
                            "I13",
                            rel,
                            f"'{node.name}' manipulates both TPR and FPR arrays, i.e. it is a"
                            " second ROC implementation whatever it is named"
                            " [AUTH: 00 §34A.2(8); 02 §C2]",
                        )
                    )
                elif _fpr_threshold_comparisons(node):
                    out.append(
                        Violation(
                            "I13",
                            rel,
                            f"'{node.name}' selects an FPR operating point outside src/scoring/"
                            " [AUTH: 02 §C1, §C2]",
                        )
                    )

        module_body = ast.Module(
            body=[
                n
                for n in tree.body
                if not isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            ],
            type_ignores=[],
        )
        names = {n.lower() for n in _identifiers(module_body)}
        if names & TPR_NAMES and names & FPR_NAMES:
            out.append(
                Violation(
                    "I13",
                    rel,
                    "module scope manipulates both TPR and FPR arrays [AUTH: 00 §34A.2(8); 02 §C2]",
                )
            )
        elif _fpr_threshold_comparisons(module_body):
            out.append(
                Violation(
                    "I13", rel, "module scope selects an FPR operating point [AUTH: 02 §C1, §C2]"
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
    if "--frozen" not in text:
        out.append(
            Violation("I10", "ci.yml", "CI does not sync from the frozen lock [AUTH: 01 §12, §46]")
        )
    if "uv lock --check" not in text:
        out.append(
            Violation(
                "I10",
                "ci.yml",
                "CI does not fail on a pyproject/uv.lock mismatch [AUTH: 01 §12, §46]",
            )
        )
    return out


DIGEST_PIN = re.compile(r"@sha256:[0-9a-f]{64}")
MUTABLE_FROM = re.compile(r"^FROM\s+(?!\$\{)(\S+)", re.MULTILINE)


def check_image_pins(root: Path) -> list[Violation]:
    """Base images are pinned by immutable digest, never by a mutable tag [AUTH: 01 §12(8)(9)]."""
    f = root / "Dockerfile"
    if not f.is_file():
        return [Violation("I6", "Dockerfile", "missing")]
    out: list[Violation] = []
    for match in MUTABLE_FROM.finditer(f.read_text(encoding="utf-8")):
        ref = match.group(1)
        if not DIGEST_PIN.search(ref):
            out.append(Violation("I6", "Dockerfile", f"base image '{ref}' is not pinned by digest"))
    return out


#: Host-generated state that must never enter a Docker build context [AUTH: 01 §12, §46].
DOCKERIGNORE_REQUIRED: tuple[str, ...] = (
    ".venv",
    "**/.venv",
    "__pycache__",
    "**/__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
)
#: The image's own interpreter, whose validity every full-context COPY must re-prove.
VENV_GUARD_MARKER = "/repo/.venv/bin/python"


def _dockerfile_stages(text: str) -> list[tuple[str, list[str]]]:
    stages: list[tuple[str, list[str]]] = []
    name: str = "<preamble>"
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("FROM "):
            stages.append((name, body))
            parts = line.split()
            name = parts[parts.index("AS") + 1] if "AS" in parts else parts[1]
            body = []
        else:
            body.append(line)
    stages.append((name, body))
    return stages


def check_build_context(root: Path) -> list[Violation]:
    """I16: a host virtualenv cannot enter the image, and every full-context COPY re-proves
    the image's own interpreter.

    Root cause of the S00-B image defect: with no .dockerignore the host macOS .venv entered
    the build context and `COPY . .` overwrote the Linux .venv that `uv sync` had just built,
    leaving /repo/.venv/bin/python dangling at /opt/homebrew [AUTH: 01 §12, §46].
    """
    out: list[Violation] = []
    ignore = root / ".dockerignore"
    if not ignore.is_file():
        return [Violation("I16", ".dockerignore", "missing; the host .venv enters the image")]
    patterns = {
        line.strip()
        for line in ignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for required in DOCKERIGNORE_REQUIRED:
        if required not in patterns:
            out.append(Violation("I16", ".dockerignore", f"does not exclude {required!r}"))
    if ".git" in patterns:
        out.append(
            Violation(
                "I16",
                ".dockerignore",
                "excludes .git; the image ships /repo as a pinned worktree so the bootstrap"
                " can verify the commit and the clean tree",
            )
        )

    dockerfile = root / "Dockerfile"
    if not dockerfile.is_file():
        return out + [Violation("I16", "Dockerfile", "missing")]
    for name, body in _dockerfile_stages(dockerfile.read_text(encoding="utf-8")):
        copy_at = next((i for i, line in enumerate(body) if line.strip() == "COPY . ."), None)
        if copy_at is None:
            continue
        if not any(VENV_GUARD_MARKER in line for line in body[copy_at:]):
            out.append(
                Violation(
                    "I16",
                    f"Dockerfile:{name}",
                    "`COPY . .` is not followed by a check that the image's own .venv still"
                    " executes; a host virtualenv could shadow it silently",
                )
            )
    return out


IMAGE_RECORD_REL = "manifests/environments/S00B_IMAGE_RECORD.json"


def check_image_record_lifecycle(root: Path) -> list[Violation]:
    """I17: a completed image record must never be baked into a build state.

    The record describes one specific built image. If a build commit already contains a
    record for an older image, the image built from that commit conflicts with it, and
    escaping needs another record, commit and rebuild - endlessly. The record is materialised
    on the pod by capture and committed with the closure evidence, so a build state carrying
    one but no captured environment is stale by construction [AUTH: 01 §12(8)(9), §16].
    """
    record = root / IMAGE_RECORD_REL
    if not record.is_file():
        return []
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", IMAGE_RECORD_REL],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        return []
    env_dir = root / "manifests" / "environments"
    captured = (
        [p for p in env_dir.glob("*.json") if re.fullmatch(r"[0-9a-f]{64}", p.stem)]
        if env_dir.is_dir()
        else []
    )
    if captured:
        return []
    return [
        Violation(
            "I17",
            IMAGE_RECORD_REL,
            "a completed image record is committed with no captured environment; it is a"
            " record for an earlier image and would be baked into the next build commit",
        )
    ]


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
    check_image_pins,
    check_build_context,
    check_image_record_lifecycle,
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
