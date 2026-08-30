"""Dry-run coverage status: every required scenario is now run, none is dependency-deferred.

This module used to record DRY-D as `NOT_RUN_DEPENDENCY(S07_S08)` and verified the deferral
against the *absence* of the objects DRY-D needs. S07 (merge operators) and S08 (recovery)
supplied them, so the same tests are inverted: the dependency is verified to *exist*, and the
carve-out that let DRY-D sit outside the covered set is gone. Returning DRY-D to permanent
deferral now fails here rather than passing quietly [AUTH: 00 §34B.1 DRY-D, §28B; 01 §39].
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

#: The vocabulary a deferral would have to use. Any of these reappearing on DRY-D means the
#: scenario has been switched back off [AUTH: 00 §34B.2].
DEFERRAL_MARKERS = ("NOT_RUN", "DEFERRED", "PENDING", "BLOCKED", "SKIP")


def test_dry_d_is_covered_and_carries_no_deferral_status() -> None:
    assert DRY_D_STATUS == "COVERED"
    for marker in DEFERRAL_MARKERS:
        assert marker not in DRY_D_STATUS.upper(), f"DRY-D status re-deferred via {marker!r}"
    assert "DRY-D" in COVERED_DRY_SCENARIOS
    assert "S07" in DRY_D_REASON and "S08" in DRY_D_REASON


def test_the_dry_d_reason_does_not_overstate_what_the_statistic_consumes() -> None:
    """R11: 00 §28B.4 defines Delta_tail over truncation-vs-calibration, not over recovered A.

    S07 supplies the truncation geometry the statistic is evaluated at, and S08 satisfies the
    planned recovery-stage dependency checkpoint. Saying the statistic is *defined over* the
    recovered artifacts would be a stronger claim than the spec makes.
    """
    lowered = DRY_D_REASON.lower()
    assert "does not consume a recovered a" in lowered
    assert "defined over" not in lowered
    assert "truncation geometry" in lowered and "calibration curve" in lowered


def test_no_required_scenario_is_uncovered() -> None:
    """The DRY-D carve-out is deleted, not widened.

    While DRY-D was deferred this comparison subtracted `{"DRY-D"}`. It no longer does, so any
    scenario dropping out of the covered set — DRY-D included — fails here.
    """
    missing = set(REQUIRED_DRY_SCENARIOS) - set(COVERED_DRY_SCENARIOS)
    assert missing == set(), f"unrun scenario(s) with no coverage: {sorted(missing)}"
    assert set(COVERED_DRY_SCENARIOS) == set(REQUIRED_DRY_SCENARIOS)


def test_the_dry_d_dependency_now_genuinely_exists(repo_root: Path) -> None:
    """The activation is verified the same three ways the deferral was.

    Delta_tail_priv(s) = R_priv^trunc(s) - f_priv^rank(e_floor(s)) [AUTH: 00 §28B.4], so it
    needs BOTH an O3 truncation artifact and a recovery on it. Each of the three checks that
    previously proved absence now proves presence, so an S08 rollback re-fires all three.
    """
    recovery = sorted(
        path.name
        for path in (repo_root / "src" / "recovery").glob("*.py")
        if path.name != "__init__.py"
    )
    assert "solvers.py" in recovery, f"src/recovery lost its solver; DRY-D input gone: {recovery}"
    assert "evaluation.py" in recovery, "src/recovery lost its e_floor evaluator"

    from src.merge.family import build_o3_descendant
    from src.recovery.evaluation import o3_error_report
    from src.recovery.solvers import recover_c2_fixed_alpha
    from src.recovery.truth import bind_truth

    for produced in (build_o3_descendant, recover_c2_fixed_alpha, o3_error_report, bind_truth):
        assert callable(produced)

    #: 00 §28B still runs only under P1_PRIMARY_LOSSY_OPERATOR, which stays uncalibrated. The
    #: dry run exercises the frozen mechanism; it does not license the real §28B experiment.
    from src.materials import UncalibratedConstantError, material
    from src.merge.settings import merge_settings

    with pytest.raises(UncalibratedConstantError):
        material(merge_settings(repo_root), "primary_lossy_operator")


def test_the_covered_scenarios_have_test_modules(repo_root: Path) -> None:
    """Each covered scenario is exercised by a module in this lane."""
    lane = repo_root / "tests" / "synthetic"
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in lane.glob("test_*.py")
        if path.name != Path(__file__).name
    )
    for scenario in COVERED_DRY_SCENARIOS:
        assert scenario in text, f"{scenario} has no test in the synthetic lane"
