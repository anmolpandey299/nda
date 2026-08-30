#!/usr/bin/env python3
"""The full-sweep entry point [AUTH: 00 §35, §36; 01 §16, §39].

The S09 launcher runs ONE registered cell. This runs the STUDY: it enumerates every task the
frozen registry implies, reports what is ready, and — once the bindings exist — walks them in
dependency order under the accepted S01 run lifecycle.

Four modes:

    plan-full     enumerate every task; deterministic, side-effect free, writes nothing
    status        readiness audit: READY / GATED_BY_RESULT / MISSING_EXECUTION_BINDING
    execute-full  walk the ready tasks in dependency order, one attempt each
    resume        the same walk, skipping tasks whose artifacts already verify

`plan-full` and `status` are pure reads: no attempt is claimed, no manifest written, no wall
clock consulted, so repeating either is bitwise identical.

`execute-full` refuses to start while any task reports MISSING_EXECUTION_BINDING. A sweep that
began with a hole in it would produce a partial study whose gaps are invisible afterwards; the
refusal names the missing bindings instead [AUTH: 01 §16].

CLI
    python scripts/run_study.py plan-full [--out artifacts/p0_pre/P0_00_FULL_SWEEP_PLAN.json]
    python scripts/run_study.py status
    python scripts/run_study.py execute-full [--gate-state PATH]
    python scripts/run_study.py resume [--gate-state PATH]
Exit
    0 = the mode completed; 1 = a refusal, or a sweep that is not code-ready
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.gates import GateError, GateState, load_gate_state  # noqa: E402
from src.experiments.registry import RegistryError  # noqa: E402
from src.experiments.stages import StageError  # noqa: E402
from src.experiments.sweep import (  # noqa: E402
    READY,
    SweepError,
    build_sweep_plan,
    readiness_audit,
)
from src.provenance.hashing import read_canonical_json  # noqa: E402


def _gate_state(root: Path, path: str | None) -> GateState:
    """A validated gate document, or the honest unresolved state.

    No CLI flag asserts a gate here either. Absent a document every gate is UNRESOLVED, which
    is what "no P0 outcome has been opened" means [AUTH: 00 §35].
    """
    if path is None:
        return GateState(provenance_class="NON_EVIDENTIARY_FIXTURE", records={})
    document = read_canonical_json(Path(path))
    if not isinstance(document, dict):
        raise GateError(f"{path}: the gate-state document is not a JSON object")
    return load_gate_state(document, root=root)


def _plan_full(root: Path, arguments: argparse.Namespace) -> int:
    plan = build_sweep_plan(root, state=_gate_state(root, arguments.gate_state))
    document = plan.as_dict()
    document["plan_sha256"] = plan.identity()
    rendered = json.dumps(document, indent=2, sort_keys=True)
    if arguments.out:
        destination = root / arguments.out
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {arguments.out} ({len(plan.tasks)} tasks, sha256 {plan.identity()[:16]})")
    else:
        print(rendered)
    return 0


def _status(root: Path, arguments: argparse.Namespace) -> int:
    plan = build_sweep_plan(root, state=_gate_state(root, arguments.gate_state))
    print(json.dumps(readiness_audit(plan), indent=2, sort_keys=True))
    return 0 if plan.ready else 1


def _execute_full(root: Path, arguments: argparse.Namespace) -> int:
    """Walk the ready tasks. Refuses outright while any binding is missing."""
    plan = build_sweep_plan(root, state=_gate_state(root, arguments.gate_state))
    if not plan.ready:
        print(
            "execute-full: the sweep is not code-ready. The following execution binding(s) do"
            f" not exist yet: {', '.join(plan.missing_bindings)}. Starting anyway would produce"
            " a partial study whose gaps are invisible afterwards [AUTH: 01 §16].",
            file=sys.stderr,
        )
        return 1
    runnable = plan.by_readiness(READY)
    print(
        f"execute-full: {len(runnable)} ready task(s) of {len(plan.tasks)}."
        " Each claims one attempt through the accepted S01 lifecycle.",
        file=sys.stderr,
    )
    # The walk itself is deliberately not reachable while TRAINING_BINDING is missing: every
    # artifact-producing task depends on it, so there is nothing honest to run yet.
    print(
        "execute-full: no artifact-producing task is currently bound; nothing was run.",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    modes = parser.add_subparsers(dest="mode", required=True)
    for name, helptext in (
        ("plan-full", "enumerate every task; side-effect free"),
        ("status", "readiness audit"),
        ("execute-full", "walk the ready tasks in dependency order"),
        ("resume", "the same walk, skipping verified artifacts"),
    ):
        mode = modes.add_parser(name, help=helptext)
        mode.add_argument("--gate-state", default=None)
        if name == "plan-full":
            mode.add_argument("--out", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    known, unknown = parser.parse_known_args(argv)
    if unknown:
        # As with the S09 launcher: a scientific dimension typed at the command line would
        # define a new experiment rather than select the registered one [AUTH: 00 §36].
        print(
            f"run_study: unrecognised argument(s) {unknown}. The study's scientific content"
            " comes from the registry, not the command line [AUTH: 00 §36].",
            file=sys.stderr,
        )
        return 1
    root = Path(known.root).resolve()
    handlers = {
        "plan-full": _plan_full,
        "status": _status,
        "execute-full": _execute_full,
        "resume": _execute_full,
    }
    try:
        return int(handlers[known.mode](root, known))
    except (RegistryError, StageError, GateError, SweepError) as exc:
        print(f"run_study: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
