"""Dry-run coverage status: what Block B ran, and what is dependency-deferred.

This test passes; it records a dependency rather than hiding a failure. DRY-D is not skipped
or xfailed — it is not implementable in this block, and the reason is asserted against the
absence of the objects it needs [AUTH: 00 §34B.1 DRY-D, §28B; 01 §39 S07, S08].
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
    """The deferral is verified, not asserted.

    Delta_tail_priv(s) = R_priv^trunc(s) - f_priv^rank(e_floor(s)) [AUTH: 00 §28B.4], so it
    needs BOTH an O3 truncation artifact and a recovery R_priv on it. S07 supplied the first
    half; the second does not exist, so the scenario is still unrunnable.

    Checked three ways, all of which re-fire the moment S08 lands:

    1. the recovery package carries no implementation;
    2. no merge module produces a recovery or privacy quantity, so S07 cannot smuggle one in;
    3. 00 §28B runs only under P1_PRIMARY_LOSSY_OPERATOR, which is REQUIRED_NOT_CALIBRATED
       and therefore fails closed.
    """
    recovery = sorted(
        path.name
        for path in (repo_root / "src" / "recovery").glob("*.py")
        if path.name != "__init__.py"
    )
    assert recovery == [], f"src/recovery now has {recovery}; DRY-D may be implementable"

    banned = ("r_priv", "delta_tail", "recover", "reconstruct")
    for path in sorted((repo_root / "src" / "merge").glob("*.py")):
        source = path.read_text(encoding="utf-8").lower()
        for name in banned:
            assert f"def {name}" not in source, f"{path.name} defines {name}; DRY-D input?"

    from src.materials import UncalibratedConstantError, material
    from src.merge.settings import merge_settings

    with pytest.raises(UncalibratedConstantError):
        material(merge_settings(repo_root), "primary_lossy_operator")


def test_every_other_required_scenario_is_covered() -> None:
    missing = set(REQUIRED_DRY_SCENARIOS) - set(COVERED_DRY_SCENARIOS) - {"DRY-D"}
    assert missing == set(), f"unrun scenarios without a recorded dependency: {sorted(missing)}"


def test_the_covered_scenarios_have_test_modules(repo_root: Path) -> None:
    """Each covered scenario is exercised by a module in this lane."""
    lane = repo_root / "tests" / "synthetic"
    text = "\n".join(path.read_text(encoding="utf-8") for path in lane.glob("test_*.py"))
    for scenario in COVERED_DRY_SCENARIOS:
        assert scenario in text, f"{scenario} has no test in the synthetic lane"
