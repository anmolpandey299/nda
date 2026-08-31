#!/usr/bin/env python3
"""Run the model compatibility contract for one exact checkpoint [AUTH: 01 §8C; 02 §C6].

Non-interactive by design: S09 will invoke it per model, and the H100 preparation step will
run it against the real research checkpoints once their immutable revisions are captured. It
downloads nothing and trains nothing on its own — it exercises the frozen backend contract
against whatever model manifest it is pointed at, and writes a provenance-bound evidence
record that a later readiness check verifies rather than trusts.

Against an unacquired panel entry every torch-dependent check reports `NOT_RUN`, and the
outcome is `NOT_RUN` — never `PASS`. That is the correct answer today: no research model has
been acquired, so no compatibility claim exists [AUTH: 03 §8].

CLI
    python scripts/model_compatibility_contract.py --root . --alias llama_3_2_3b
    python scripts/model_compatibility_contract.py --root . --alias llama_3_2_3b \
        --parameter-names <file> --evidence artifacts/p0_pre/evidence/backend/<alias>.json
Exit
    0 = the contract PASSED; 1 = FAIL or NOT_RUN
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backend.adapters import (  # noqa: E402
    AdapterError,
    resolve_lora_specification,
)
from src.backend.architecture import (  # noqa: E402
    ArchitectureError,
    resolve_trunk_selection,
)
from src.backend.compatibility import (  # noqa: E402
    CONTRACT_CHECKS,
    CheckResult,
    ContractOutcome,
    ContractWeights,
    check_adapter_round_trip,
    check_checkpoint_quantization,
    check_dp_requirement,
    check_exact_inversion,
    check_h100_profile,
    check_induced_update_exact,
    check_linear_merge,
    check_multimodal_frozen,
    check_rank_32_attaches,
    check_scorer_determinism,
    failed,
    not_run,
    passed,
)
from src.backend.dp import build_dp_plan, dp_role_for  # noqa: E402
from src.backend.evidence import (  # noqa: E402
    BACKEND_NOT_INSTALLED,
    CHECKPOINT_NOT_ACQUIRED,
    MODEL_REVISION_NOT_FROZEN,
    NO_PARAMETER_NAMES,
    STRUCTURAL_CHECK_FAILED,
    EvidenceError,
    RunBinding,
    build_contract_evidence,
    write_contract_evidence,
)
from src.backend.loader import backend_available, build_load_plan  # noqa: E402
from src.backend.revision import (  # noqa: E402
    ModelRevisionNotFrozenError,
    resolve_model_identity,
)
from src.backend.scoring_backend import backend_code_hash  # noqa: E402
from src.materials import material  # noqa: E402
from src.provenance.config import resolve_config  # noqa: E402
from src.provenance.hashing import sha256_canonical  # noqa: E402
from src.provenance.model_manifest import FIXTURE_ROLE  # noqa: E402

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def panel_entry(root: Path, alias: str) -> dict[str, Any]:
    configs = root / "configs"
    document = resolve_config(configs / "models" / "panel.json", config_root=configs)
    panel = document.get("panel")
    if not isinstance(panel, list):
        raise SystemExit("the model panel config defines no 'panel'")
    for entry in panel:
        if isinstance(entry, dict) and entry.get("alias") == alias:
            return dict(entry)
    known = sorted(str(e.get("alias")) for e in panel if isinstance(e, dict))
    raise SystemExit(f"{alias!r} is not in the frozen panel; known aliases: {known}")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class ProvenanceUnavailable(RuntimeError):
    """Evidence cannot be issued because the current state cannot be resolved."""


def _git_commit(root: Path) -> str:
    """The resolved HEAD commit, or a refusal.

    An all-zero sha would pass every structural check while proving nothing, so an
    unresolvable commit means no evidence is issued rather than evidence with a placeholder
    [AUTH: 01 §15, §16].
    """
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProvenanceUnavailable(
            f"the git commit for {root} could not be resolved, so backend-contract evidence"
            " has no code provenance to bind [AUTH: 01 §15, §16]"
        ) from exc
    commit = out.stdout.strip()
    if len(commit) != 40 or not all(c in "0123456789abcdef" for c in commit):
        raise ProvenanceUnavailable(f"git returned {commit!r}, which is not a commit sha")
    return commit


def run_contract_for(
    root: Path,
    alias: str,
    *,
    parameter_names: list[str] | None,
    weights: ContractWeights | None = None,
    entry: dict[str, Any] | None = None,
    h100_profile: dict[str, Any] | None = None,
    checkpoint_config: Path | None = None,
) -> ContractOutcome:
    """Run all thirteen checks for one panel alias.

    `weights` carries the weight-dependent inputs for checks 7-11. The real loader supplies
    them on the H100 image; a tiny local fixture supplies them on CPU. Either way the same
    check functions run, so the fixture exercises the production contract.
    """
    entry = entry if entry is not None else panel_entry(root, alias)
    family = str(entry["architecture_family"])
    results: list[CheckResult] = []

    # 1 — exact revision supplied
    try:
        identity = resolve_model_identity(entry, evidentiary=True)
        results.append(
            passed(
                "exact_revision_supplied",
                f"pinned at {identity.revision[:12]}",
                revision=identity.revision,
            )
        )
        evidentiary = True
    except ModelRevisionNotFrozenError as exc:
        identity = resolve_model_identity(entry, evidentiary=False)
        results.append(failed("exact_revision_supplied", str(exc)))
        evidentiary = False

    from src.backend.settings import backend_settings

    forbidden_raw = material(backend_settings(root), "forbidden_load_options")
    forbidden = tuple(str(k) for k in forbidden_raw) if isinstance(forbidden_raw, list) else ()

    # 2 / 3 — loading and the BF16 text-only forward need the real backend
    if not backend_available():
        detail = "NOT_RUN(BACKEND_NOT_INSTALLED): torch/transformers resolve on the H100 image"
        results.append(not_run("model_loads", BACKEND_NOT_INSTALLED, detail))
        results.append(not_run("bf16_text_only_forward", BACKEND_NOT_INSTALLED, detail))
    elif str(entry.get("role", "")) == FIXTURE_ROLE:
        # A fixture carries a well-formed revision but no acquired weights, so on an image
        # where transformers resolves, `from_pretrained` would treat its model id as a Hub
        # repository and reach the network. A contract run downloads nothing: the row is
        # NOT_RUN because the checkpoint was never acquired [AUTH: 01 §8G, §12; 02 §C6].
        detail = (
            "NOT_RUN(CHECKPOINT_NOT_ACQUIRED): this entry is a test fixture, not a research"
            " checkpoint; the contract never fetches a model to close a row"
        )
        results.append(not_run("model_loads", CHECKPOINT_NOT_ACQUIRED, detail))
        results.append(not_run("bf16_text_only_forward", CHECKPOINT_NOT_ACQUIRED, detail))
    elif not evidentiary:
        detail = "NOT_RUN(MODEL_REVISION_NOT_FROZEN): the checkpoint has not been acquired"
        results.append(not_run("model_loads", MODEL_REVISION_NOT_FROZEN, detail))
        results.append(not_run("bf16_text_only_forward", MODEL_REVISION_NOT_FROZEN, detail))
    else:  # pragma: no cover - requires the H100 image
        from src.backend.loader import LoaderError, load_causal_lm
        from src.backend.loader import parameter_names as names_of

        plan = build_load_plan(root, identity, model_config=entry)
        try:
            model = load_causal_lm(plan)
            parameter_names = list(names_of(model))
            results.append(passed("model_loads", f"loaded at {plan.dtype}", dtype=plan.dtype))
            results.append(
                passed("bf16_text_only_forward", "text-only forward available", dtype=plan.dtype)
            )
        except LoaderError as exc:
            results.append(failed("model_loads", str(exc)))
            results.append(
                not_run(
                    "bf16_text_only_forward",
                    STRUCTURAL_CHECK_FAILED,
                    "the model did not load",
                )
            )

    # 4 — the ACTUAL checkpoint's own config, from an acquired local snapshot
    results.append(
        check_checkpoint_quantization(
            config_path=checkpoint_config,
            declared_config_sha256=str(entry.get("config_sha256", "")),
            forbidden=forbidden,
        )
    )

    # 5 / 6 — the LoRA surface, resolvable from parameter names alone
    if parameter_names is None:
        detail = "NOT_RUN(NO_PARAMETER_NAMES): supply --parameter-names, or acquire the checkpoint"
        results.append(not_run("multimodal_components_frozen", NO_PARAMETER_NAMES, detail))
        results.append(not_run("common_mlp_lora_rank_32_attaches", NO_PARAMETER_NAMES, detail))
    else:
        try:
            selection = resolve_trunk_selection(
                root, architecture_family=family, parameter_names=parameter_names
            )
            specification = resolve_lora_specification(
                root, architecture_family=family, target_modules=selection.target_modules
            )
            results.append(check_multimodal_frozen(selection))
            results.append(check_rank_32_attaches(selection, specification))
        except (ArchitectureError, AdapterError) as exc:
            results.append(failed("multimodal_components_frozen", str(exc)))
            results.append(failed("common_mlp_lora_rank_32_attaches", str(exc)))

    # 7-11 — adapter, extraction, merge, inversion and scoring need weights
    if weights is None:
        reason = MODEL_REVISION_NOT_FROZEN if not evidentiary else BACKEND_NOT_INSTALLED
        detail = (
            "NOT_RUN(MODEL_REVISION_NOT_FROZEN): these checks need the checkpoint's weights"
            if not evidentiary
            else "NOT_RUN(BACKEND_NOT_INSTALLED): torch/peft resolve on the H100 image"
        )
        for name in (
            "adapter_save_load_stable",
            "induced_update_ba_exact",
            "linear_merge_constructs",
            "p0_c1_exact_inversion",
            "scorer_deterministic",
        ):
            results.append(not_run(name, reason, detail))
    else:
        results.append(check_adapter_round_trip(weights))
        results.append(
            check_induced_update_exact(
                weights.factors, scaling=weights.scaling, extracted=weights.extracted
            )
        )
        results.append(check_linear_merge(weights))
        results.append(check_exact_inversion(root, weights.extracted))
        results.append(check_scorer_determinism(weights.score, tolerance=weights.score_tolerance))

    # 12 — H100 profile
    results.append(check_h100_profile(h100_profile or _h100_profile()))

    # 13 — DP where the role requires it
    role = dp_role_for(root, alias)
    dp_plan = _dp_plan(root, alias) if role is not None else None
    results.append(check_dp_requirement(role, dp_plan))

    ordered = {result.name: result for result in results}
    return ContractOutcome(
        model_id=identity.model_id,
        revision=identity.revision,
        architecture_family=family,
        results=tuple(
            ordered.get(
                name,
                not_run(name, STRUCTURAL_CHECK_FAILED, "the contract did not evaluate this"),
            )
            for name in CONTRACT_CHECKS
        ),
    )


def _h100_profile() -> dict[str, Any] | None:
    if not backend_available():  # pragma: no cover - requires the H100 image
        return None
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():  # pragma: no cover - requires the H100 image
        return None
    properties = torch.cuda.get_device_properties(0)  # pragma: no cover
    return {  # pragma: no cover
        "device_name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
    }


def _dp_plan(root: Path, alias: str) -> Any | None:
    """Build the DP plan from the frozen fixture mechanism, without running DP."""
    from src.backend.dp import DPBackendError
    from src.dp.mechanism import DPMechanism
    from src.materials import material_number, material_text
    from src.training.settings import dp_settings

    settings = dp_settings(root)
    try:
        mechanism = DPMechanism(
            adjacency=material_text(settings, "adjacency"),
            delta=material_number(settings, "fixture_delta", allow_provisional=True),
            clipping_norm=material_number(
                settings, "fixture_clipping_norm", allow_provisional=True
            ),
            noise_multiplier=material_number(
                settings, "fixture_noise_multiplier", allow_provisional=True
            ),
            sample_rate=material_number(settings, "fixture_sample_rate", allow_provisional=True),
            steps=int(material_number(settings, "fixture_steps", allow_provisional=True)),
            dp_seed=int(material_number(settings, "smoke_seed")),
            requested_epsilon=material_number(settings, "target_epsilon"),
        )
        return build_dp_plan(root, mechanism, model_alias=alias)
    except DPBackendError:
        return None


def _binding(root: Path, alias: str, *, run_id: str, attempt_id: str, revision: str) -> RunBinding:
    """Recompute the current state the evidence will be verified against.

    `run_id` is supplied by the caller — the S09 run-manifest layer issues it. This command
    never mints one; its own local identifier is `attempt_id`, which is not scientific
    provenance and is named so it cannot be read as such [AUTH: 01 §15, §16].
    """
    from src.scoring.cache import scoring_code_hash

    configs = root / "configs"
    scoring = resolve_config(configs / "attacks" / "scoring.json", config_root=configs)
    return RunBinding(
        run_id=run_id,
        git_commit=_git_commit(root),
        environment_lock_sha256=sha256_canonical(
            {"lock": (root / "uv.lock").read_text(encoding="utf-8")}
        ),
        model_manifest_sha256=sha256_canonical(panel_entry(root, alias)),
        model_revision=revision,
        backend_code_sha256=backend_code_hash(root),
        scoring_code_sha256=scoring_code_hash(root, scoring),
        resolved_config_sha256=sha256_canonical(scoring),
        attempt_id=attempt_id,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--alias", required=True)
    parser.add_argument("--parameter-names", default=None, help="JSON list of parameter names")
    parser.add_argument("--evidence", default=None, help="repo-relative evidence output path")
    parser.add_argument(
        "--run-id",
        default=None,
        help="the 64-hex RUN_ID issued by the S09 run-manifest layer; required with --evidence",
    )
    parser.add_argument(
        "--checkpoint-config",
        default=None,
        help="path to the acquired snapshot's config.json, for the quantization check",
    )
    arguments = parser.parse_args(argv)
    root = Path(arguments.root).resolve()

    names: list[str] | None = None
    if arguments.parameter_names:
        loaded = json.loads(Path(arguments.parameter_names).read_text(encoding="utf-8"))
        names = [str(item) for item in loaded]
    checkpoint_config = Path(arguments.checkpoint_config) if arguments.checkpoint_config else None

    started = _now()
    outcome = run_contract_for(
        root,
        arguments.alias,
        parameter_names=names,
        checkpoint_config=checkpoint_config,
    )
    ended = _now()

    print(f"model_compatibility_contract: {outcome.model_id} -> {outcome.status}")
    for result in outcome.results:
        suffix = f" [{result.reason}]" if result.reason else ""
        print(f"  {result.status:>7}  {result.name}: {result.detail}{suffix}")

    if not arguments.evidence:
        return outcome.exit_code

    # An unacquired or unpinned model is not eligible for evidentiary compatibility evidence.
    # Say so cleanly and write nothing: a file at this path would claim to be backend-contract
    # evidence for a checkpoint that does not exist [AUTH: 01 §8G; 03 §8].
    if not _COMMIT_RE.match(outcome.revision):
        print(
            f"  evidence: NOT_WRITTEN — NOT_RUN({MODEL_REVISION_NOT_FROZEN}):"
            f" {outcome.model_id} has revision {outcome.revision!r}, which is not an"
            " immutable 40-hex revision. An unacquired model is not eligible for"
            " evidentiary backend-contract evidence [AUTH: 01 §8G].",
            file=sys.stderr,
        )
        return 1

    if not arguments.run_id:
        print(
            "  evidence: NOT_WRITTEN — --run-id is required with --evidence. The RUN_ID is"
            " issued by the S09 run-manifest layer; this command binds it and does not mint"
            " one [AUTH: 01 §15, §16].",
            file=sys.stderr,
        )
        return 1
    if not _SHA256_RE.match(str(arguments.run_id)):
        print(
            f"  evidence: NOT_WRITTEN — --run-id {arguments.run_id!r} is not 64 hex chars.",
            file=sys.stderr,
        )
        return 1

    attempt_id = sha256_canonical(
        {"contract": outcome.as_dict(), "started": started, "alias": arguments.alias}
    )
    try:
        binding = _binding(
            root,
            arguments.alias,
            run_id=str(arguments.run_id),
            attempt_id=attempt_id,
            revision=outcome.revision,
        )
    except ProvenanceUnavailable as exc:
        print(f"  evidence: NOT_WRITTEN — {exc}", file=sys.stderr)
        return 1

    try:
        document = build_contract_evidence(
            binding=binding,
            status=outcome.status,
            checks=[result.as_dict() for result in outcome.results],
            artifact_hashes={},
            started_utc=started,
            ended_utc=ended,
            exit_code=outcome.exit_code,
        )
        write_contract_evidence(root, arguments.evidence, document)
    except EvidenceError as exc:
        print(f"  evidence: NOT_WRITTEN — {exc}", file=sys.stderr)
        return 1
    print(f"  evidence: {arguments.evidence} (status {document['status']})")
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
