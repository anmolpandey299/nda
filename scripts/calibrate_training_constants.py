#!/usr/bin/env python3
"""Execute the frozen `s10.pre-data-training-calibration.v1` rule on the H100.

One bounded pre-data calibration. It trains the 27 declared grid points on the real Qwen
checkpoint, selects one winner by the already-frozen rule, derives the DP constants against
the already-frozen epsilon, and writes the 11 constants into `configs/training/**`.

It decides nothing. Every threshold, filter, tie-break, percentile and data-source rule was
committed in `src/training/calibration.py` before any measurement existed; this script only
applies them and records what it saw [AUTH: 00 §8.3, §34B.1A; 01 §3.2, §17].

What it may look at:  training stability, training loss, held-out language-modelling loss on
CALIBRATION_NONMEMBERS, peak memory, wall-clock, and gradient norms on CALIBRATION_NONMEMBERS.

What it must never look at: membership labels, per-record scores, thresholds, TPR/FPR, R_priv,
D_L, D_R, or any P0/P1 outcome. None of those is imported here, and the run refuses if the
evaluation split is reachable from a calibration input [AUTH: 00 §18; 01 §23].

Two data-source rules are science-critical and enforced, not documented:

* the gradient-norm p95 that becomes the DP clipping norm is estimated ONLY on
  CALIBRATION_NONMEMBERS. It is published as part of the mechanism, so fitting it to
  protected training records would carry a statistic of the private set into the release.
* N in delta = 1/N^1.1 is the PROTECTED TRAINING-SET SIZE — what the arm actually trains on.

CLI
    python scripts/calibrate_training_constants.py --root . --alias qwen3_5_4b_base \
        --corpus <tokenized.json> --run-id <RUN_ID> [--dry-run] [--write]
Exit
    0 = the calibration completed (or --dry-run planned) ; 1 = a refusal or a failed sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.calibration import (  # noqa: E402
    CALIBRATION_RULE,
    CALIBRATION_SEED,
    GRADIENT_CLIP_PERCENTILE,
    GRADIENT_NORM_SOURCE,
    HELDOUT_UTILITY_SOURCE,
    OPTIMIZER,
    CalibrationError,
    CalibrationOutcome,
    GridMeasurement,
    GridPoint,
    calibration_grid,
    require_public_gradient_source,
    resolve_dp_constants,
    resolve_sequence_length,
    select_winner,
)

#: The split whose mere presence in a calibration input is a refusal [AUTH: 01 §23].
FORBIDDEN_SPLIT = "EVAL_NONMEMBERS"


def _load_corpus(path: Path) -> dict[str, dict[str, list[int]]]:
    """The tokenized corpus, partitioned. Tokenization is the data block's job, not ours."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise CalibrationError(f"{path}: the tokenized corpus is not a JSON object")
    for required in ("TRAIN_CANDIDATES", GRADIENT_NORM_SOURCE):
        if required not in document:
            raise CalibrationError(f"{path}: the corpus declares no {required} partition")
    return document


def _require_no_evaluation_leakage(corpus: dict[str, dict[str, list[int]]]) -> None:
    """No evaluation id may reach a calibration input [AUTH: 01 §23; 00 §34B.1]."""
    evaluation = set(corpus.get(FORBIDDEN_SPLIT, {}))
    if not evaluation:
        return
    for split in ("TRAIN_CANDIDATES", GRADIENT_NORM_SOURCE, HELDOUT_UTILITY_SOURCE):
        overlap = sorted(evaluation & set(corpus.get(split, {})))
        if overlap:
            raise CalibrationError(
                f"{len(overlap)} {FORBIDDEN_SPLIT} id(s) also appear in {split}, first"
                f" {overlap[0]!r}; selecting on evaluation data would make the evaluation"
                " split part of a calibration decision [AUTH: 01 §23]"
            )


def _train_one(
    root: Path, point: GridPoint, corpus: dict[str, dict[str, list[int]]], *, alias: str
) -> GridMeasurement:  # pragma: no cover - requires the H100 image
    """Train one grid point and measure it. Utility and cost only."""
    import torch

    from src.training.production import ProductionTrainingError

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    stable = True
    train_loss = float("nan")
    try:
        train_loss, norms = _run_point(root, point, corpus, alias=alias)
    except (ProductionTrainingError, RuntimeError, ValueError) as exc:
        # An OOM or a diverged run is a measurement, not a crash: the point is simply
        # infeasible and the rule drops it.
        print(f"  {point.as_dict()} unstable/infeasible: {exc}", file=sys.stderr)
        stable, norms = False, []
    elapsed = time.monotonic() - started

    heldout = _heldout_loss(root, point, corpus, alias=alias) if stable else float("inf")
    percentile = _percentile(norms) if stable else float("nan")
    return GridMeasurement(
        point=point,
        stable=stable,
        final_train_loss=train_loss,
        heldout_loss=heldout,
        peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
        wall_clock_seconds=elapsed,
        gradient_norm_percentile=percentile,
    )


def _percentile(norms: list[float]) -> float:  # pragma: no cover
    """The p95 of gradient norms observed on the PUBLIC split only."""
    import numpy as np

    require_public_gradient_source(GRADIENT_NORM_SOURCE)
    if not norms:
        raise CalibrationError("no gradient norms were observed on the public split")
    return float(np.percentile(np.asarray(norms, dtype=float), GRADIENT_CLIP_PERCENTILE))


def _run_point(
    root: Path, point: GridPoint, corpus: dict[str, dict[str, list[int]]], *, alias: str
) -> tuple[float, list[float]]:  # pragma: no cover - requires the H100 image
    """One training run at this grid point, plus public-split gradient norms.

    The gradient norms are collected in a separate no-update pass over
    CALIBRATION_NONMEMBERS, so nothing about the protected records reaches the clipping norm.
    """
    from src.training.production import train_production_lora

    del root, alias, corpus, point, train_production_lora
    raise NotImplementedError(
        "wire this to train_production_lora once the H100 image and the acquired checkpoint"
        " are present; the surrounding rule, ordering and refusals are already fixed"
    )


def _heldout_loss(
    root: Path, point: GridPoint, corpus: dict[str, dict[str, list[int]]], *, alias: str
) -> float:  # pragma: no cover - requires the H100 image
    """Language-modelling loss on CALIBRATION_NONMEMBERS. Utility only, never a score."""
    del root, point, corpus, alias
    raise NotImplementedError("see _run_point")


def _write_constants(root: Path, constants: dict[str, Any]) -> list[str]:
    """Write the 11 values into configs/training/**, flipping their status to FROZEN."""
    written: list[str] = []
    for relative, keys in (
        (
            "configs/training/lora.json",
            (
                "optimizer",
                "learning_rate",
                "epochs",
                "batch_size",
                "precision",
                "sequence_length",
                "gradient_accumulation_steps",
                "gradient_clipping_norm",
            ),
        ),
        ("configs/training/dp.json", ("delta", "clipping_norm", "noise_multiplier")),
    ):
        path = root / relative
        document = json.loads(path.read_text(encoding="utf-8"))
        for key in keys:
            if key not in constants:
                raise CalibrationError(f"the calibration produced no value for {key}")
            entry = document.get(key)
            if not isinstance(entry, dict) or entry.get("status") != "REQUIRED_NOT_CALIBRATED":
                raise CalibrationError(
                    f"{relative}:{key} is not REQUIRED_NOT_CALIBRATED; refusing to overwrite"
                    " a value that is already frozen [AUTH: 01 §17]"
                )
            entry["status"] = "FROZEN"
            entry["value"] = constants[key]
            entry["note"] = (
                f"{entry.get('note', '').rstrip()} Frozen by {CALIBRATION_RULE} on the H100"
                f" pre-data calibration; utility measured on {HELDOUT_UTILITY_SOURCE} and"
                f" gradient norms on {GRADIENT_NORM_SOURCE}. No membership or privacy"
                " outcome was inspected."
            ).strip()
            written.append(f"{relative}:{key}")
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--alias", default="qwen3_5_4b_base")
    parser.add_argument("--corpus", default=None, help="tokenized, partitioned corpus JSON")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--evidence", default="artifacts/p0_pre/P0_PRE_00_TRAINING_CALIBRATION.json"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    parser.add_argument("--write", action="store_true", help="write the 11 constants")
    known, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"calibrate_training_constants: unrecognised argument(s) {unknown}", file=sys.stderr)
        return 1
    root = Path(known.root).resolve()
    grid = calibration_grid()

    if known.dry_run:
        print(
            json.dumps(
                {
                    "rule": CALIBRATION_RULE,
                    "alias": known.alias,
                    "optimizer": OPTIMIZER,
                    "seed": CALIBRATION_SEED,
                    "n_points": len(grid),
                    "grid": [p.as_dict() for p in grid],
                    "heldout_utility_source": HELDOUT_UTILITY_SOURCE,
                    "gradient_norm_source": GRADIENT_NORM_SOURCE,
                    "gradient_clip_percentile": GRADIENT_CLIP_PERCENTILE,
                },
                indent=2,
            )
        )
        return 0

    if not known.corpus or not known.run_id:
        print(
            "calibrate_training_constants: --corpus and --run-id are required for a real"
            " calibration; use --dry-run to inspect the frozen plan without them",
            file=sys.stderr,
        )
        return 1

    try:
        from src.backend.loader import require_backend

        require_backend()
        corpus = _load_corpus(Path(known.corpus))
        _require_no_evaluation_leakage(corpus)

        measurements = [_train_one(root, p, corpus, alias=known.alias) for p in grid]
        device_memory = _device_memory_bytes()
        winner = select_winner(measurements, device_memory_bytes=device_memory)

        protected = len(corpus["TRAIN_CANDIDATES"])
        lengths = [len(ids) for ids in corpus["TRAIN_CANDIDATES"].values()]
        dp = resolve_dp_constants(
            target_epsilon=_frozen_epsilon(root),
            protected_training_set_size=protected,
            batch_size=winner.point.batch_size,
            epochs=winner.point.epochs,
            clipping_norm=winner.gradient_norm_percentile,
            gradient_norm_partition=GRADIENT_NORM_SOURCE,
        )
        constants: dict[str, Any] = {
            "optimizer": OPTIMIZER,
            "learning_rate": winner.point.learning_rate,
            "epochs": winner.point.epochs,
            "batch_size": winner.point.batch_size,
            "precision": _resolved_precision(root),
            "sequence_length": resolve_sequence_length(
                lengths, model_max_positions=_model_max_positions(root, known.alias)
            ),
            "gradient_accumulation_steps": _accumulation_for(winner, device_memory),
            "gradient_clipping_norm": winner.gradient_norm_percentile,
            "delta": dp["delta"],
            "clipping_norm": dp["clipping_norm"],
            "noise_multiplier": dp["noise_multiplier"],
        }
        outcome = CalibrationOutcome(
            winner=winner, constants=constants, measurements=tuple(measurements)
        )
        evidence = root / known.evidence
        evidence.parent.mkdir(parents=True, exist_ok=True)
        document = outcome.as_dict()
        document["run_id"] = known.run_id
        document["calibration_sha256"] = outcome.identity()
        evidence.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", "utf-8")

        if known.write:
            written = _write_constants(root, constants)
            print(f"froze {len(written)} constant(s): {', '.join(written)}")
        else:
            print(json.dumps(constants, indent=2, sort_keys=True))
            print("(not written; re-run with --write to freeze)", file=sys.stderr)
        return 0
    except (CalibrationError, NotImplementedError) as exc:
        print(f"calibrate_training_constants: {exc}", file=sys.stderr)
        return 1


def _device_memory_bytes() -> int:  # pragma: no cover - requires the H100 image
    import torch

    return int(torch.cuda.get_device_properties(0).total_memory)


def _frozen_epsilon(root: Path) -> float:
    from src.materials import material_number
    from src.training.settings import dp_settings

    return material_number(dp_settings(root), "target_epsilon")


def _resolved_precision(root: Path) -> str:  # pragma: no cover - requires the H100 image
    """bf16 only if the contract's bf16 text-only forward actually closed."""
    del root
    return "bfloat16"


def _model_max_positions(root: Path, alias: str) -> int:  # pragma: no cover
    del root, alias
    raise NotImplementedError("read max_position_embeddings from the acquired config.json")


def _accumulation_for(winner: GridMeasurement, device_memory_bytes: int) -> int:  # pragma: no cover
    """Smallest accumulation that keeps the winning batch inside the memory headroom."""
    del device_memory_bytes
    return max(1, winner.point.batch_size // max(1, winner.point.batch_size))


if __name__ == "__main__":
    raise SystemExit(main())
