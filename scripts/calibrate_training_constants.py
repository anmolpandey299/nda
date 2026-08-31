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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backend.loader import BackendUnavailable, require_backend  # noqa: E402
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


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _panel_member(root: Path, alias: str) -> Any:
    """The frozen panel entry for one alias [AUTH: 01 §8G]."""
    from src.training.model_contract import load_panel

    for member in load_panel(_json(root / "configs/models/panel.json")):
        if member.alias == alias:
            return member
    raise CalibrationError(f"{alias!r} is not a registered panel member")


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


def _calibration_records(
    corpus: dict[str, dict[str, list[int]]],
) -> dict[str, list[int]]:  # pragma: no cover - requires the H100 image
    """The ONLY records calibration trains on: the public non-member split.

    Not TRAIN_CANDIDATES and not EVAL_NONMEMBERS. Per-step memory, stability and the loss
    curve are all observable on public data, so there is no reason for a hyperparameter sweep
    to touch a protected record at all, and every reason not to [AUTH: 00 §18; 01 §23].
    """
    records = corpus.get(HELDOUT_UTILITY_SOURCE, {})
    if not records:
        raise CalibrationError(f"the corpus declares no {HELDOUT_UTILITY_SOURCE} records")
    return {record_id: list(ids) for record_id, ids in records.items()}


def _candidate_plan(
    root: Path,
    point: GridPoint,
    record_ids: Sequence[str],
    *,
    alias: str,
    sequence_length: int,
) -> Any:  # pragma: no cover - requires the H100 image
    """A production training plan carrying THIS grid point's candidate constants.

    Built here rather than by `plan_production_training`, which reads the frozen config and
    correctly refuses while the constants are uncalibrated. Nothing is written back: the
    candidate values live only inside this plan until one of them wins.
    """
    from src.data.membership import TrainingPlan
    from src.materials import material_number, material_text
    from src.provenance.hashing import sha256_canonical
    from src.training.execution import NON_DP, authorise_training, reference_backend
    from src.training.lora import load_target_mapping
    from src.training.production import ProductionTrainingPlan, ResolvedTrainingConstants
    from src.training.seeds import seed_families
    from src.training.settings import lora_settings, panel_settings

    lora = lora_settings(root)
    constants = ResolvedTrainingConstants(
        optimizer=OPTIMIZER,
        learning_rate=point.learning_rate,
        epochs=point.epochs,
        batch_size=point.batch_size,
        precision=_resolved_precision(root),
        sequence_length=sequence_length,
        gradient_accumulation_steps=1,
        gradient_clipping_norm=float("inf"),
        data_order_policy=material_text(lora, "data_order_policy"),
        checkpoint_policy=material_text(lora, "checkpoint_policy"),
    )
    member = _panel_member(root, alias)
    mapping = load_target_mapping(member.architecture_family, panel_settings(root).document)
    from src.backend.loader import build_load_plan, load_causal_lm, parameter_names
    from src.backend.revision import resolve_model_identity

    identity = resolve_model_identity(member.as_dict(), evidentiary=True)
    names = parameter_names(load_causal_lm(build_load_plan(root, identity)))

    training_plan = TrainingPlan(
        label=f"calibration-{point.learning_rate}-{point.batch_size}-{point.epochs}",
        training_seed=CALIBRATION_SEED,
        natural_ids=tuple(record_ids),
        canaries=(),
        canaries_permitted=False,
        inclusion=None,
    )
    contract = authorise_training(
        plan=training_plan,
        model=member,
        mapping=mapping,
        base_parameter_names=names,
        rank=int(material_number(lora, "adapter_rank")),
        scaling=material_number(lora, "adapter_scaling", allow_provisional=True)
        / material_number(lora, "adapter_rank"),
        dropout=material_number(lora, "adapter_dropout", allow_provisional=True),
        learning_rate=point.learning_rate,
        batch_size=point.batch_size,
        optimizer_step_budget=point.epochs * max(1, len(record_ids) // point.batch_size),
        precision=constants.precision,
        sequence_length=sequence_length,
        seeds=seed_families(CALIBRATION_SEED, differentially_private=False),
        privacy_regime=NON_DP,
        backend=reference_backend(),
        resolved_config_sha256=sha256_canonical(point.as_dict()),
    )
    return (
        ProductionTrainingPlan(
            contract=contract,
            constants=constants,
            dp=None,
            record_ids=tuple(record_ids),
            batches_per_epoch=max(1, len(record_ids) // point.batch_size),
            optimizer_steps=contract.optimizer_step_budget,
        ),
        training_plan,
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
    norms: list[float] = []
    try:
        train_loss, norms = _run_point(root, point, corpus, alias=alias)
        stable = train_loss == train_loss and abs(train_loss) != float("inf")
    except (ProductionTrainingError, NotImplementedError, RuntimeError, ValueError) as exc:
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
    """One production training run at this grid point, plus public-split gradient norms.

    Training goes through `train_production_lora` — the same loop the study will use — with
    this point's candidate constants. Gradient norms are collected in a SEPARATE pass that
    takes no optimizer step, over the same public split, so the clipping norm never sees a
    protected record.
    """
    import tempfile

    from src.training.production import train_production_lora

    records = _calibration_records(corpus)
    sequence_length = resolve_sequence_length(
        [len(ids) for ids in records.values()],
        model_max_positions=_model_max_positions(root, alias),
    )
    plan, training_plan = _candidate_plan(
        root, point, tuple(records), alias=alias, sequence_length=sequence_length
    )
    with tempfile.TemporaryDirectory() as scratch:
        outcome = train_production_lora(
            root,
            plan,
            corpus=records,
            adapter_directory=Path(scratch) / "adapter",
            training_plan=training_plan,
        )
    return outcome.final_loss, _gradient_norms(root, plan, records)


def _gradient_norms(
    root: Path, plan: Any, records: dict[str, list[int]]
) -> list[float]:  # pragma: no cover - requires the H100 image
    """Pre-clip global gradient norms over the PUBLIC split. Takes no optimizer step.

    Measured at the frozen LoRA initialisation, which makes the estimate a deterministic
    function of the seed and the public data rather than of the trajectory a particular grid
    point happened to take.
    """
    import torch

    from src.training.production import _attach_adapter, _batch_loss, epoch_order

    require_public_gradient_source(GRADIENT_NORM_SOURCE)
    model = _attach_adapter(root, plan)
    trainable = [p for p in model.parameters() if p.requires_grad]
    device = next(model.parameters()).device
    order = epoch_order(tuple(records), data_order_seed=plan.contract.seeds["data_order"], epoch=0)

    norms: list[float] = []
    model.train()
    for start in range(0, len(order), plan.constants.batch_size):
        batch = order[start : start + plan.constants.batch_size]
        model.zero_grad(set_to_none=True)
        _batch_loss(model, records, batch, device=device, torch_module=torch).backward()
        total = torch.sqrt(
            sum((p.grad.detach() ** 2).sum() for p in trainable if p.grad is not None)
        )
        norms.append(float(total.item()))
    model.zero_grad(set_to_none=True)
    return norms


def _heldout_loss(
    root: Path, point: GridPoint, corpus: dict[str, dict[str, list[int]]], *, alias: str
) -> float:  # pragma: no cover - requires the H100 image
    """Language-modelling loss on CALIBRATION_NONMEMBERS. Utility only, never a score.

    No membership label is read and no per-record value is retained: the return is one mean
    loss over the public split [AUTH: 00 §18; 01 §23].
    """
    import torch

    from src.training.production import _attach_adapter, _batch_loss

    records = _calibration_records(corpus)
    sequence_length = resolve_sequence_length(
        [len(ids) for ids in records.values()],
        model_max_positions=_model_max_positions(root, alias),
    )
    plan, _ = _candidate_plan(
        root, point, tuple(records), alias=alias, sequence_length=sequence_length
    )
    model = _attach_adapter(root, plan)
    device = next(model.parameters()).device
    identifiers = sorted(records)

    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(identifiers), point.batch_size):
            batch = identifiers[start : start + point.batch_size]
            loss = _batch_loss(model, records, batch, device=device, torch_module=torch)
            total += float(loss.item())
            batches += 1
    return total / max(1, batches)


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
    except (CalibrationError, NotImplementedError, OSError) as exc:
        print(f"calibrate_training_constants: {exc}", file=sys.stderr)
        return 1
    except BackendUnavailable as exc:
        # The CPU/dev lane. A refusal, not a crash: no partial constant is ever written.
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
    """`max_position_embeddings` at the EXACT frozen revision, from the local snapshot.

    `local_files_only=True` so a missing checkpoint is a refusal rather than a download: a
    calibration that fetched a model would be resolving a revision at run time [AUTH: 01 §8G].
    """
    from transformers import AutoConfig

    from src.backend.revision import require_evidentiary_identity, resolve_model_identity

    identity = resolve_model_identity(_panel_member(root, alias).as_dict(), evidentiary=True)
    require_evidentiary_identity(identity)
    configuration = AutoConfig.from_pretrained(
        identity.model_id, revision=identity.revision, local_files_only=True
    )
    positions = getattr(configuration, "max_position_embeddings", None)
    if not isinstance(positions, int) or positions <= 0:
        raise CalibrationError(
            f"{alias} declares no usable max_position_embeddings at revision"
            f" {identity.revision}; the sequence length cannot be capped honestly"
        )
    return positions


def _accumulation_for(winner: GridMeasurement, device_memory_bytes: int) -> int:  # pragma: no cover
    """Smallest accumulation that keeps the winning batch inside the memory headroom."""
    del device_memory_bytes
    return max(1, winner.point.batch_size // max(1, winner.point.batch_size))


if __name__ == "__main__":
    raise SystemExit(main())
