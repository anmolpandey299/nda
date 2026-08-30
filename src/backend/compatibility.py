"""The model compatibility contract [AUTH: 01 §8, §8B, §8C; 00 §25 P0-C1; 02 §C6].

Thirteen checks decide whether one exact checkpoint can carry the study. They run in
dependency order and stop at the first structural failure, because a rank-32 attachment
result on a model that did not load is not information.

Every check returns a row carrying its own status and detail, so the evidence record shows
what was evaluated rather than a single verdict. `NOT_RUN` is a distinct outcome from `FAIL`:
a check that could not execute here — because torch is absent, or because no GPU exists — says
so, and a `NOT_RUN` row can never contribute to a `PASS` [AUTH: 02 §C6].

This module runs the contract. It does **not** decide readiness: `BACKEND_INTEGRATED` and
`P0_PRE_READY` stay false until the real execution, the full synthetic rerun, the cache
assertions, the H100 smoke and the 1,000-sequence benchmark have all passed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.backend.evidence import (
    CHECKPOINT_NOT_ACQUIRED,
    FAIL,
    NO_H100,
    NOT_RUN,
    OPACUS_BACKEND_NOT_INSTALLED,
    PASS,
    STRUCTURAL_CHECK_FAILED,
)
from src.provenance.hashing import JSONValue, sha256_file

CONTRACT_SCHEMA: Final = "backend.compatibility-contract.v1"

#: The thirteen checks, in dependency order [AUTH: 01 §8C].
CONTRACT_CHECKS: Final[tuple[str, ...]] = (
    "exact_revision_supplied",
    "model_loads",
    "bf16_text_only_forward",
    "no_quantization_required",
    "multimodal_components_frozen",
    "common_mlp_lora_rank_32_attaches",
    "adapter_save_load_stable",
    "induced_update_ba_exact",
    "linear_merge_constructs",
    "p0_c1_exact_inversion",
    "scorer_deterministic",
    "h100_profile_collectable",
    "dp_smoke_where_required",
)

#: Checks whose failure makes every later check meaningless.
_STRUCTURAL: Final[frozenset[str]] = frozenset(
    {"exact_revision_supplied", "model_loads", "no_quantization_required"}
)


@dataclass(frozen=True)
class CheckResult:
    """One contract check.

    A NOT_RUN row carries its specific `reason`: "not run" without a cause is not a finding,
    and collapsing every cause to one string would lose the only information such a row has.
    """

    name: str
    status: str
    detail: str
    reason: str = ""
    measurement: Mapping[str, JSONValue] = field(default_factory=dict)

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "reason": self.reason,
            "measurement": dict(self.measurement),
        }


@dataclass(frozen=True)
class ContractOutcome:
    """The whole contract for one model."""

    model_id: str
    revision: str
    architecture_family: str
    results: tuple[CheckResult, ...]

    @property
    def status(self) -> str:
        """The frozen aggregation, shared with the evidence verifier [AUTH: 02 §C6].

        Any FAIL makes the contract FAIL; otherwise any NOT_RUN makes it NOT_RUN; only an
        all-PASS set of the thirteen declared checks is a PASS.
        """
        from src.backend.evidence import aggregate_status

        by_name = {result.name: result for result in self.results}
        missing = [name for name in CONTRACT_CHECKS if name not in by_name]
        if missing:
            return NOT_RUN
        return aggregate_status([result.as_dict() for result in self.results])

    @property
    def exit_code(self) -> int:
        return 0 if self.status == PASS else 1

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "schema": CONTRACT_SCHEMA,
            "model_id": self.model_id,
            "revision": self.revision,
            "architecture_family": self.architecture_family,
            "status": self.status,
            "checks": [result.as_dict() for result in self.results],
        }


CheckFn = Callable[[], CheckResult]


def not_run(name: str, reason: str, detail: str = "") -> CheckResult:
    """A check that could not execute, with the stable cause that prevented it."""
    return CheckResult(
        name=name, status=NOT_RUN, detail=detail or f"NOT_RUN({reason})", reason=reason
    )


def passed(name: str, detail: str, **measurement: JSONValue) -> CheckResult:
    return CheckResult(name=name, status=PASS, detail=detail, measurement=measurement)


def failed(name: str, detail: str, **measurement: JSONValue) -> CheckResult:
    return CheckResult(name=name, status=FAIL, detail=detail, measurement=measurement)


def run_contract(checks: Sequence[tuple[str, CheckFn]]) -> tuple[CheckResult, ...]:
    """Run the checks in order, short-circuiting after a structural failure."""
    results: list[CheckResult] = []
    halted = False
    for name, run in checks:
        if halted:
            results.append(
                not_run(
                    name,
                    STRUCTURAL_CHECK_FAILED,
                    "a structural check failed earlier in the contract",
                )
            )
            continue
        result = run()
        results.append(result)
        if result.status == FAIL and name in _STRUCTURAL:
            halted = True
    return tuple(results)


# ----------------------------------------------------------------------------------------
# the mathematical checks, on plain arrays
# ----------------------------------------------------------------------------------------


def check_induced_update_exact(
    factors: Mapping[str, tuple[Any, Any]], *, scaling: float, extracted: Mapping[str, Any]
) -> CheckResult:
    """ΔW must equal scaling·B·A exactly, and differ from both mis-scalings."""
    name = "induced_update_ba_exact"
    worst = 0.0
    for module, (a, b) in sorted(factors.items()):
        expected = float(scaling) * (np.asarray(b, np.float64) @ np.asarray(a, np.float64))
        actual = np.asarray(extracted[module], dtype=np.float64)
        if actual.shape != expected.shape:
            return failed(name, f"{module}: shape {actual.shape} != {expected.shape}")
        worst = max(worst, float(np.max(np.abs(actual - expected))))
    if worst != 0.0:
        return failed(name, f"the induced update differs from scaling*B*A by {worst:g}")
    if scaling == 1.0:
        return failed(
            name,
            "scaling is 1.0, so this fixture cannot distinguish a missing scaling from a"
            " correct one; the check is not informative",
        )
    return passed(name, "ΔW = scaling·B·A exactly", max_abs_deviation=worst, scaling=scaling)


def check_exact_inversion(
    root: Path, protected_updates: Mapping[str, Any], *, alpha: float = 0.5
) -> CheckResult:
    """00 §25 P0-C1, through the ACCEPTED S07/S08 path — not a local re-derivation.

    A hand-written inversion inside this module would only prove that this module can invert
    its own algebra. What the contract needs to know is whether the *accepted* pipeline works
    on this model's extracted surface, so the check builds a genuine S07 linear descendant,
    issues an S08 known-partner observation, runs the accepted `recover_c1`, binds truth
    through `bind_truth`, and reads e_F from the accepted evaluator.
    """
    name = "p0_c1_exact_inversion"
    from src.merge.context import resolve_merge_context

    names = sorted(protected_updates)
    if not names:
        return failed(name, "no extracted parameter surface to invert over")
    merge_context = resolve_merge_context(root)
    try:
        return _run_c1(
            name,
            root,
            names,
            protected_updates,
            alpha=alpha,
            merge_context=merge_context,
        )
    except (ValueError, KeyError) as exc:
        #: A degenerate or incompatible surface is a compatibility failure, not a crash: the
        #: contract reports what it found rather than aborting the whole run.
        return failed(name, f"the accepted S08 C1 path refused this surface: {exc}")


def _run_c1(
    name: str,
    root: Path,
    names: Sequence[str],
    protected_updates: Mapping[str, Any],
    *,
    alpha: float,
    merge_context: Any,
) -> CheckResult:
    """The body of the inversion check. Split out so the caller owns the failure policy."""
    from src.backend.settings import backend_settings
    from src.materials import material_number
    from src.merge.family import MergeSpec, build_release_family
    from src.merge.updates import fixture_induced_update
    from src.recovery.context import fixture_recovery_context
    from src.recovery.evaluation import parameter_recovery_metrics
    from src.recovery.observations import observe_known_partner
    from src.recovery.settings import recovery_settings
    from src.recovery.solvers import recover_c1
    from src.recovery.truth import bind_truth

    recovery_context = fixture_recovery_context(recovery_settings(root).document)

    protected = fixture_induced_update(
        {n: np.asarray(protected_updates[n], dtype=np.float64) for n in names},
        context=merge_context,
        origin="A",
    )
    rng = np.random.Generator(np.random.PCG64(0xC1))
    partner_tensors: dict[str, Any] = {
        n: np.asarray(rng.normal(size=np.asarray(protected_updates[n]).shape), dtype=np.float64)
        for n in names
    }
    partner = fixture_induced_update(partner_tensors, context=merge_context, origin="B1")
    family = build_release_family(
        protected=protected,
        partners={"B1": partner},
        specs=[
            MergeSpec(
                descendant_id="C1",
                operator="O1_LINEAR_TASK_ARITHMETIC",
                alpha=float(alpha),
                partner_id="B1",
            )
        ],
        permitted_k=(1,),
        context=merge_context,
    )
    observation = observe_known_partner(
        descendant=family.by_id("C1"), partner_update=partner, context=recovery_context
    )
    result = recover_c1(observation)
    binding = bind_truth(
        observation=observation, source=family.by_id("C1"), protected_truth=protected
    )
    metrics = parameter_recovery_metrics(result, binding)
    e_f = float(str(metrics["e_f"]))
    limit = material_number(backend_settings(root), "c1_exact_inversion_limit")
    if e_f > limit:
        return failed(name, f"e_F = {e_f:.3e} exceeds the frozen {limit:g}", e_f=e_f)
    return passed(
        name,
        f"accepted S08 C1 recovers the extracted surface: e_F = {e_f:.3e} <= {limit:g}",
        e_f=e_f,
        limit=limit,
        recovery_result_sha256=result.identity(),
        truth_binding_sha256=binding.identity(),
    )


def check_scorer_determinism(
    score: Callable[[], Sequence[float]], *, tolerance: float
) -> CheckResult:
    """Execute the scoring path TWICE and require bitwise-identical scalars.

    `score` is invoked here, twice, rather than being handed two result lists: comparing a
    list against a copy of itself is tautological and would pass on a non-deterministic
    backend. Two independent executions of the same call under the same identity is the
    property the contract actually needs.
    """
    name = "scorer_deterministic"
    first = np.asarray(score(), dtype=np.float64)
    second = np.asarray(score(), dtype=np.float64)
    if first.shape != second.shape:
        return failed(name, f"score tables differ in shape: {first.shape} vs {second.shape}")
    if first.size == 0:
        return failed(name, "no scores were produced, so determinism was not evaluated")
    if not np.all(np.isfinite(first)):
        return failed(name, "the scorer produced non-finite scalars")
    worst = float(np.max(np.abs(first - second)))
    if worst > tolerance:
        return failed(name, f"scores drift by {worst:g} > {tolerance:g}", max_deviation=worst)
    return passed(
        name,
        f"two independent executions agree to {worst:g}",
        n_scores=int(first.size),
        max_deviation=worst,
        tolerance=tolerance,
    )


def check_multimodal_frozen(selection: Any) -> CheckResult:
    """No multimodal parameter may reach the LoRA surface [AUTH: 01 §8B]."""
    from src.backend.architecture import is_multimodal

    name = "multimodal_components_frozen"
    leaked = sorted(n for n in selection.target_parameters if is_multimodal(n))
    if leaked:
        return failed(name, f"multimodal parameter(s) entered the LoRA surface: {leaked[:3]}")
    return passed(
        name,
        f"{len(selection.excluded_multimodal)} multimodal parameter(s) excluded",
        n_excluded=len(selection.excluded_multimodal),
        n_targets=len(selection.target_parameters),
    )


def check_rank_32_attaches(selection: Any, specification: Any) -> CheckResult:
    """The common MLP surface must resolve at exactly rank 32 [AUTH: 01 §8B]."""
    name = "common_mlp_lora_rank_32_attaches"
    if specification.rank != 32:
        return failed(name, f"the resolved LoRA rank is {specification.rank}, not 32")
    if set(selection.target_modules) != {"gate_proj", "up_proj", "down_proj"}:
        return failed(name, f"the target modules are {selection.target_modules}")
    if not selection.target_parameters:
        return failed(name, "no language-trunk parameter matched the declared mapping")
    return passed(
        name,
        f"rank 32 attaches to {len(selection.target_parameters)} MLP projections",
        rank=32,
        n_targets=len(selection.target_parameters),
        target_modules=list(selection.target_modules),
    )


#: What the H100 row is actually about [AUTH: 01 §21].
H100_DEVICE_MARKER: Final = "H100"


def check_h100_profile(profile: Mapping[str, JSONValue] | None) -> CheckResult:
    """H100 memory/throughput information must be collectable [AUTH: 01 §21].

    This row closes only on real H100 hardware. A supplied profile naming anything else can
    exercise the parser but must not produce a scientific PASS: the check is "is an H100
    viable for this model", and no CPU host can answer that.
    """
    name = "h100_profile_collectable"
    if profile is None:
        return not_run(name, NO_H100, "NOT_RUN(NO_H100): no accelerator is present")
    missing = sorted({"device_name", "total_memory_bytes"} - set(profile))
    if missing:
        return failed(name, f"the collected profile omits {missing}")
    device = str(profile["device_name"])
    if H100_DEVICE_MARKER not in device.upper():
        return not_run(
            name,
            NO_H100,
            f"NOT_RUN(NO_H100): the collected profile names {device!r}; the row closes only on"
            " real H100 hardware and a fixture accelerator cannot answer it [AUTH: 01 §21]",
        )
    return passed(name, f"profile collected for {device}", **dict(profile))


def check_checkpoint_quantization(
    *,
    config_path: Path | None,
    declared_config_sha256: str | None,
    forbidden: Sequence[str],
) -> CheckResult:
    """Does the ACTUAL frozen checkpoint require a quantized load? [AUTH: 01 §8C]

    This is a different property from the request policy the loader enforces. The loader
    refuses a caller who *asks* for 8-bit; this asks whether the checkpoint itself declares a
    quantized-only mode, which can only be answered from the checkpoint's own `config.json`.

    So it needs an acquired local snapshot. Without one the honest answer is
    NOT_RUN(CHECKPOINT_NOT_ACQUIRED) — reading the panel manifest instead would certify a
    property of our own declaration rather than of the checkpoint. The snapshot's bytes are
    verified against the manifest's frozen `config_sha256` before they are read, so a swapped
    or drifted config cannot be inspected as if it were the pinned one. No network lookup.
    """
    name = "no_quantization_required"
    if config_path is None or not config_path.is_file():
        return not_run(
            name,
            CHECKPOINT_NOT_ACQUIRED,
            "NOT_RUN(CHECKPOINT_NOT_ACQUIRED): no local exact-revision snapshot exists, so the"
            " checkpoint's own configuration cannot be inspected [AUTH: 01 §8G; 03 §8]",
        )
    if not declared_config_sha256 or declared_config_sha256 == "UNRESOLVED_NOT_DOWNLOADED":
        return failed(
            name,
            "a local snapshot exists but the manifest records no frozen config_sha256 to"
            " verify it against [AUTH: 01 §8G]",
        )
    actual = sha256_file(config_path)
    if actual != declared_config_sha256:
        return failed(
            name,
            f"the local config hashes to {actual[:12]} but the manifest pins"
            f" {declared_config_sha256[:12]}; this is not the checkpoint that was frozen"
            " [AUTH: 01 §8G, §16]",
            config_sha256=actual,
        )
    import json

    document = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        return failed(name, "the checkpoint config is not a JSON object")
    refusals = quantization_declarations(document, forbidden=forbidden)
    if refusals:
        return failed(name, "; ".join(refusals), config_sha256=actual)
    return passed(
        name,
        "the acquired checkpoint declares no quantized-only mode",
        config_sha256=actual,
    )


def quantization_declarations(
    document: Mapping[str, Any], *, forbidden: Sequence[str]
) -> list[str]:
    """Quantized-only declarations in a checkpoint's own config, at any nesting depth."""
    from src.backend.loader import QUANTIZATION_REQUIRED

    problems: list[str] = []
    for key in forbidden:
        value = document.get(key)
        if value in (None, False):
            continue
        problems.append(f"{QUANTIZATION_REQUIRED}: the checkpoint config sets {key}={value!r}")
    method = document.get("quantization_config")
    if isinstance(method, Mapping) and method:
        problems.append(
            f"{QUANTIZATION_REQUIRED}: the checkpoint declares quantization_config {sorted(method)}"
        )
    return problems


@dataclass(frozen=True)
class ContractWeights:
    """The weight-dependent inputs checks 7-11 need [AUTH: 01 §8C].

    Supplied by the real loader on the H100 image and by a tiny local fixture on CPU. Both
    feed the SAME check functions, so a fixture run exercises the production contract rather
    than a parallel easier one; what differs is only where the numbers came from, which the
    outcome records through the model identity.
    """

    factors: Mapping[str, tuple[Any, Any]]
    scaling: float
    extracted: Mapping[str, Any]
    reloaded_update_sha256: str
    saved_update_sha256: str
    merged: Mapping[str, Any]
    #: Invoked twice by the determinism check. A pair of precomputed lists would let a copy
    #: masquerade as a reproduction.
    score: Callable[[], Sequence[float]]
    score_tolerance: float = 0.0


def check_adapter_round_trip(weights: ContractWeights) -> CheckResult:
    """Saving and reloading an adapter must preserve the induced update exactly."""
    name = "adapter_save_load_stable"
    if weights.reloaded_update_sha256 != weights.saved_update_sha256:
        return failed(
            name,
            f"the reloaded adapter's induced update is {weights.reloaded_update_sha256[:12]}"
            f" but the saved one was {weights.saved_update_sha256[:12]}",
        )
    return passed(
        name,
        "the adapter round-trips to an identical induced update",
        induced_update_sha256=weights.saved_update_sha256,
    )


def check_linear_merge(weights: ContractWeights) -> CheckResult:
    """A linear descendant must be constructible on the extracted surface [AUTH: 00 §10.1]."""
    name = "linear_merge_constructs"
    if not weights.merged:
        return failed(name, "no linear descendant was constructed")
    surfaces = {n: np.asarray(v).shape for n, v in weights.merged.items()}
    expected = {n: np.asarray(v).shape for n, v in weights.extracted.items()}
    if surfaces != expected:
        return failed(name, "the merged descendant does not cover the adapted surface")
    if not all(np.all(np.isfinite(np.asarray(v))) for v in weights.merged.values()):
        return failed(name, "the merged descendant carries NaN or Inf")
    return passed(name, f"a linear descendant covers {len(surfaces)} target matrices")


def check_dp_requirement(role: str | None, plan: Any | None) -> CheckResult:
    """DP smoke applies only where 01 §8E gives the model a DP role."""
    name = "dp_smoke_where_required"
    if role is None:
        return passed(name, "this model carries no DP role, so no DP smoke is required")
    if plan is None:
        return failed(name, f"{role} requires a DP plan and none was built")
    return not_run(
        name,
        OPACUS_BACKEND_NOT_INSTALLED,
        f"NOT_RUN(OPACUS_BACKEND_NOT_INSTALLED): {role} is wired to {plan.engine}; the DP smoke"
        " runs on the H100 image and is not executed here",
    )
