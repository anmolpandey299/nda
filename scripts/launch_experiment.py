#!/usr/bin/env python3
"""The S09 experiment launcher [AUTH: 00 §35, §36; 01 §15, §16, §39 S09].

Three modes, and nothing else:

    list     the canonical registry rows and their static statuses
    plan     resolve ONE registered cell against the gate state; side-effect free
    execute  claim ONE run attempt through the accepted S01 lifecycle and run ONE task

The caller chooses a registered **cell id**, never a new experiment. There is deliberately no
`--k`, `--lineage`, `--operator`, `--alpha` or `--recovery-method`: those would define a
combination 00 §36 never registered. There is likewise no `--dare-pass`, `--p0-c1-pass` or
`--allow-na`: a scientific truth a user can type is not evidence, and a STRUCTURAL_NA cell is
permanently non-executable.

Gate state is supplied as a validated document (`--gate-state`), which must name the artifact,
hash and RUN_IDs that decided each gate. Absent one, every gate is UNRESOLVED — which is the
honest current state, because no P0 outcome has been opened.

CLI
    python scripts/launch_experiment.py list [--phase p1] [--regime CALIBRATION]
    python scripts/launch_experiment.py plan --cell C05 --model <alias> --seed 101
    python scripts/launch_experiment.py execute --cell <id> --model <alias> --seed 101 \
        --task <registered-task-id>
Exit
    0 = the mode completed; 1 = a refusal or a non-runnable cell
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.gates import (  # noqa: E402
    GateError,
    GateState,
    load_gate_state,
    resolve_primary_lossy_operator,
)
from src.experiments.launcher import (  # noqa: E402
    LaunchError,
    list_cells,
    plan_cell,
    plan_document,
)
from src.experiments.registry import (  # noqa: E402
    RegistryError,
    load_registry,
    refuse_scientific_override,
)
from src.experiments.resolution import ResolutionError  # noqa: E402
from src.experiments.settings import registry_sha256  # noqa: E402
from src.experiments.stages import (  # noqa: E402
    eligibility_report,
    load_stage_registry,
)
from src.provenance.hashing import read_canonical_json  # noqa: E402

PHASES = ("p0", "p1")


def _gate_state(root: Path, path: str | None) -> GateState:
    """A validated gate document, or the unresolved state.

    No CLI flag can assert a gate. Absent a document every gate is UNRESOLVED, which is what
    "no P0 outcome has been opened" actually means [AUTH: 00 §35].
    """
    if path is None:
        return GateState(provenance_class="NON_EVIDENTIARY_FIXTURE", records={})
    document = read_canonical_json(Path(path))
    if not isinstance(document, dict):
        raise GateError(f"{path}: the gate-state document is not a JSON object")
    return load_gate_state(document, root=root)


def _list(root: Path, arguments: argparse.Namespace) -> int:
    if arguments.phase == "p0":
        state = _gate_state(root, arguments.gate_state)
        report = eligibility_report(load_stage_registry(root), state)
        report["selected_primary_lossy_operator"] = resolve_primary_lossy_operator(state)
        print(json.dumps(report, indent=2))
        return 0
    rows = list_cells(root, regime=arguments.regime)
    print(
        json.dumps(
            {
                "registry_sha256": registry_sha256(root),
                "registry_version": load_registry(root).version,
                "count": len(rows),
                "cells": rows,
            },
            indent=2,
        )
    )
    return 0


def _plan(root: Path, arguments: argparse.Namespace) -> int:
    state = _gate_state(root, arguments.gate_state)
    plan = plan_cell(
        root,
        arguments.cell,
        model_alias=arguments.model,
        seed=arguments.seed,
        state=state,
        dp_seeds=tuple(arguments.dp_seed or ()),
    )
    print(json.dumps(plan_document(plan), indent=2))
    return 0 if plan.resolved.runnable else 1


def _execute(root: Path, arguments: argparse.Namespace) -> int:
    """EXECUTE is deliberately not reachable without a registered task definition.

    S09 registers no scientific task: the real ones arrive with S10 and the production
    backend. Refusing here is the honest state — there is nothing registered to run, and
    inventing a task would be inventing an experiment [AUTH: 00 §36; 01 §39].
    """
    del root, arguments
    print(
        "execute: no scientific task is registered yet. The S09 launcher claims one attempt"
        " through the accepted S01 lifecycle and runs one REGISTERED task; the registered"
        " task set arrives with S10 and the production backend. Nothing is invented here"
        " [AUTH: 00 §36; 01 §39].",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    modes = parser.add_subparsers(dest="mode", required=True)

    listing = modes.add_parser("list", help="print the canonical registry")
    listing.add_argument("--phase", choices=PHASES, default="p1")
    listing.add_argument("--regime", choices=("CALIBRATION", "REPRESENTATIVE_DP"), default=None)
    listing.add_argument("--gate-state", default=None)

    planning = modes.add_parser("plan", help="resolve one registered cell")
    planning.add_argument("--cell", required=True)
    planning.add_argument("--model", required=True)
    planning.add_argument("--seed", type=int, required=True)
    planning.add_argument("--gate-state", default=None)
    planning.add_argument("--dp-seed", type=int, action="append", default=None)

    execution = modes.add_parser("execute", help="claim one attempt and run one registered task")
    execution.add_argument("--cell", required=True)
    execution.add_argument("--model", required=True)
    execution.add_argument("--seed", type=int, required=True)
    execution.add_argument("--task", required=True)
    execution.add_argument("--gate-state", default=None)
    execution.add_argument("--dp-seed", type=int, action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    known, unknown = parser.parse_known_args(argv)
    if unknown:
        # A scientific dimension supplied at the command line would define a new experiment
        # rather than select a registered one [AUTH: 00 §36].
        print(
            f"launch_experiment: unrecognised argument(s) {unknown}. The scientific"
            " combination — operator, k, lineage, schedule, alpha, recovery method — comes"
            " from the registry row, not the command line. Choose a registered cell id"
            " [AUTH: 00 §36].",
            file=sys.stderr,
        )
        return 1
    root = Path(known.root).resolve()

    handlers: dict[str, Any] = {"list": _list, "plan": _plan, "execute": _execute}
    try:
        return int(handlers[known.mode](root, known))
    except (RegistryError, ResolutionError, LaunchError, GateError) as exc:
        print(f"launch_experiment: {exc}", file=sys.stderr)
        return 1


def refuse_overrides(**overrides: Any) -> None:
    """Exposed so the launcher's refusal is testable directly [AUTH: 00 §36]."""
    refuse_scientific_override(**overrides)


if __name__ == "__main__":
    raise SystemExit(main())
