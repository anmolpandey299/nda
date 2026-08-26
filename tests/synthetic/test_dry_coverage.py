"""Dry-run coverage status: what Block B ran, and what is dependency-deferred.

This test passes; it records a dependency rather than hiding a failure. DRY-D is not skipped
or xfailed — it is not implementable in this block, and the reason is asserted against the
absence of the objects it needs [AUTH: 00 §34B.1 DRY-D, §28B; 01 §39 S07, S08].
"""

from __future__ import annotations

from pathlib import Path

from src.analysis.dryrun import (
    COVERED_DRY_SCENARIOS,
    DRY_D_REASON,
    DRY_D_STATUS,
    REQUIRED_DRY_SCENARIOS,
)


def test_dry_d_status_is_a_recorded_dependency() -> None:
    assert DRY_D_STATUS == "NOT_RUN_DEPENDENCY(S07_S08)"
    assert "S07" in DRY_D_REASON and "S08" in DRY_D_REASON
    assert "DRY-D" not in COVERED_DRY_SCENARIOS


def test_the_dry_d_dependency_genuinely_does_not_exist(repo_root: Path) -> None:
    """The deferral is verified, not asserted: the merge and recovery packages that would
    produce Delta_tail_priv carry no implementation yet."""
    for package in ("merge", "recovery"):
        modules = sorted(
            path.name
            for path in (repo_root / "src" / package).glob("*.py")
            if path.name != "__init__.py"
        )
        assert modules == [], f"src/{package} now has {modules}; DRY-D may be implementable"


def test_every_other_required_scenario_is_covered() -> None:
    missing = set(REQUIRED_DRY_SCENARIOS) - set(COVERED_DRY_SCENARIOS) - {"DRY-D"}
    assert missing == set(), f"unrun scenarios without a recorded dependency: {sorted(missing)}"


def test_the_covered_scenarios_have_test_modules(repo_root: Path) -> None:
    """Each covered scenario is exercised by a module in this lane."""
    lane = repo_root / "tests" / "synthetic"
    text = "\n".join(path.read_text(encoding="utf-8") for path in lane.glob("test_*.py"))
    for scenario in COVERED_DRY_SCENARIOS:
        assert scenario in text, f"{scenario} has no test in the synthetic lane"
