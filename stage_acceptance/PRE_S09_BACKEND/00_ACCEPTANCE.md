# PRE-S09 Acceptance — Production HF/PEFT Backend

## 1. Identity and scope
```text
STAGE_ID = PRE_S09_BACKEND · STAGE_NAME = PRODUCTION_HF_PEFT_BACKEND · STATUS = ACCEPTED
PRE_S09_BACKEND_IMPLEMENTATION_COMMIT = 9a2437eb9f98e490a1109d1d12396492566b0946
                                        "feat(pre-s09): integrate production hf peft backend"
BRANCH = stage/pre-s09-production-backend · DATE = 2026-08-30T16:04:04+01:00
BASE_ACCEPTED_STATE   = stage-s08-v1
S08_ACCEPTANCE_COMMIT = 7762d4cb3e5187f03a2b3df404555cf57f8b4c18

implemented      exact-revision HF model identity · production causal-LM loading contract ·
                 text-only language-trunk selection · explicit four-family MLP target
                 mappings · PEFT LoRA rank-32 contract · adapter persistence and
                 verification · induced ΔW = scaling·B·A extraction · S07 verified
                 task-vector handoff · canonical scorer backend · token alignment and
                 masking · backend/scoring cache-identity integration · frozen S06 DP
                 production hook · model compatibility command · provenance-bound
                 compatibility evidence
not implemented  S09 experiment registry / orchestrator · real model downloads · real model
                 compatibility · real training · H100 execution · the 1,000-sequence
                 benchmark · privacy outcomes · P0
authority        00 §8, §14, §22, §25, §34A · 01 §7-§10, §12, §13-§18, §21, §39-§42 ·
                 02 §C6, §C7, §C8 · 03 · 04
```

This is backend **code** integration on deterministic local fixtures: no PubMed record fetched,
no research model downloaded or trained, no privacy outcome inspected, no GPU used, no package
installed and no dependency lock touched.

## 2. Model and revision contract
```text
Qwen/Qwen3.5-4B-Base · google/gemma-4-E4B ·
mistralai/Ministral-3-3B-Base-2512 · meta-llama/Llama-3.2-3B
```
Scientific execution requires a canonical immutable 40-hex revision. `main`, `master`, `HEAD`,
`latest`, any `refs/…`, `None`, an empty string and any non-40-hex value all reject; the
revision is normalised once — stripped and lower-cased — before validation, so validation and
the stored identity operate on the same value and no accepted identity carries whitespace or
mixed case [AUTH: 01 §8G].

Every panel revision is `UNRESOLVED_NOT_DOWNLOADED`. That is a valid *declaration* and an
invalid *evidentiary* value, so each model can be recorded now and cannot be run on later.
No revision SHA was invented [AUTH: 03 §8].

```text
REAL_MODEL_COMPATIBILITY = NOT_RUN
```
This acceptance certifies no real checkpoint.

## 3. LoRA contract
```text
TARGET_POLICY = LANGUAGE_TRUNK_MLP_ONLY · LORA_RANK = 32
roles         gate projection · up projection · down projection
forbidden     attention · embeddings · LM head · vision tower · audio tower ·
              multimodal projector · PLE embeddings · router / expert gating
```
The 01 §8B.1 fallback `LANGUAGE_TRUNK_ALL_COMMON_LINEAR` is **not** activated; it stays gated
on the later P0-A measurement-power condition.

`adapter_config.json` travels with the adapter and can be edited, so it is treated as an
untrusted declaration to validate rather than as authority. The canonical specification is
resolved from version-controlled config and the declaration is compared against it field by
field; a mismatch is a refusal, never an override. Validated fields: `r`, `lora_alpha`,
`lora_dropout`, `bias`, `task_type`, `lora_target_policy`, `lora_scaling_rule`, `use_rslora`,
`architecture_mapping_version`, and the target set.

`use_rslora` resolves through the same fail-closed material accessor as every other
scaling-affecting value. A missing, `REQUIRED_NOT_CALIBRATED` or non-boolean entry refuses; it
cannot silently become `False`, because it changes the scaling rule outright.

## 4. The induced update
```text
ΔW = scaling · (B @ A)          scaling = lora_alpha / r, applied exactly once
frozen fixture: alpha = 64 · rank = 32 · scaling = 2
```
Scaling 2 is deliberately not 1, so an omitted or doubled factor is detectable rather than
invisible. Verified independently: the orientation is PEFT's (A is `(r, in)`, B is `(out, r)`),
the scaling is applied once, and the unscaled and double-scaled variants both differ from the
issued update.

Raw factors are never the scientific object: the factorisation is not unique, so `(SA, BS⁻¹)`
and `(A, B)` name the same ΔW while comparing as different matrices [AUTH: 00 §22]. That
equality is asserted directly.

## 5. Target-surface boundary
The persisted tensor keys decide what enters ΔW, so they are checked against a canonical
expected selection derived from the **base model's own parameter names** under the frozen
policy — never from the adapter. Matching is suffix-exact on the declared module path, not
substring, so `lm_head.down_proj` and `vision_tower…mlp.up_proj` cannot be captured.

Rejected: attention `q_proj`/`o_proj`, LM head, embeddings, vision and audio towers, the
multimodal projector, per-layer embeddings, router/expert gating, any extra target, any
missing expected target, and duplicates. An adapter whose `target_modules` truthfully claims
the MLP set while its tensors carry a forbidden module is rejected on the tensors.

An exact gate/up/down surface passes. `expected_selection` is a required argument on both the
writer and the reader: a default derived from what is being written would make the check
circular.

## 6. S07 handoff
```text
backend adapter → validated artifact → accepted S06-style manifest →
S07 verified_task_vector_from_adapter → independently re-derived ΔW
```
S07 re-validates the manifest, re-reads and re-hashes the artifact bytes, reloads the adapter
and derives ΔW itself; nothing the backend asserts is taken on trust. The two derivations are
compared and must agree. No parallel trusted-vector path exists — an AST check refuses any
backend import of `_issue`, `VerifiedTaskVector` or `_canonical_bytes` from S07.

A tampered alpha and a forbidden target both reject **on read**, before any S06 artifact is
written, so no accepted downstream object is ever produced from them. S07 production is
byte-unchanged.

## 7. Scorer backend
The backend supplies token/model execution evidence only: correctly aligned per-token
log-probabilities and a validity mask. The accepted scorer remains the sole authority for the
reference-calibrated score, the Min-K% reduction, membership metrics, ROC, TPR@FPR and all
cross-fitting and statistical logic [AUTH: 00 §34A.1, §34A.2]. No `min_k`,
`reference_calibrated_score` or `negative_log_likelihood` definition exists under `src/backend`.

Independently verified: position *t*'s logits predict token *t+1*, so an unshifted read is a
different number and is asserted to differ; padding, `IGNORE_INDEX` labels and prompt positions
never enter a reduction; log-softmax accumulates in float64 regardless of forward precision.

The declared truncation policy `RIGHT_TRUNCATE_TO_MAX_SEQUENCE_LENGTH` is **executed**, not
merely recorded: the left prefix is retained and ids, attention mask and prompt mask are cut at
the same index. An unimplemented policy fails closed rather than being approximated.

## 8. Cache identity
```text
authorised accepted-stage change: src/scoring/cache.py — identity surface only
```
Purpose: bind the HF/PEFT backend's executable semantics and its resolved runtime config into
the scoring/cache identity under 02 §C7. Verified by diff that no line touches cache lookup,
cache storage, payload fields, score rows, statistical formulas, estimators, thresholds or
membership logic.

Invalidation verified for: backend source, the causal shift, masking, adapter extraction,
tokenizer identity, model revision, the score reducer, `attn_implementation` (sdpa and
flash_attention_2), `forward_precision`, `trust_remote_code`, `lora_scaling_rule`,
`use_rslora`, `training_precision` and `local_files_only`. None of the last six edits a single
source byte, and every one changes what the backend computes.

Unchanged source, config and inputs give a stable identity and a hit returning
bitwise-identical rows.

## 9. The absent-config marker
`BACKEND_RUNTIME_CONFIG_ABSENT` exists **only** for generic source/cache hashing over a
source-only copied tree that carries no `configs/backend/runtime.json`. Established by
verification:

* absent and present hash to different identities, so a tree that gains the config invalidates
  every score taken without it;
* scientific backend execution with the runtime config absent **fails closed** — both
  `backend_runtime_config_sha256` and `resolve_lora_specification` raise `ConfigError`.

The marker maps to no executable default. It describes what a tree carries; it never supplies
a value anything runs on.

## 10. Checkpoint quantization
Two distinct properties, deliberately separated:

* **request policy** — the loader refuses a caller asking for 8-bit, 4-bit, GPTQ, AWQ or a
  `quantization_config`, and a refused plan cannot produce `from_pretrained` kwargs;
* **checkpoint compatibility** — whether the frozen checkpoint itself requires a quantized
  mode. That can only be answered from the checkpoint's own `config.json`.

```text
snapshot absent                → NOT_RUN(CHECKPOINT_NOT_ACQUIRED)
unquantized, hash matches      → PASS
quantized config               → FAIL
config hash ≠ manifest pin     → FAIL
```
The snapshot's bytes are verified against the manifest's frozen `config_sha256` before they are
read, so a swapped or drifted config cannot be inspected as if it were the pinned one. No panel
manifest can certify the checkpoint as unquantized — reading our own declaration would certify
a property of the declaration, not of the checkpoint. No network lookup occurs inside the check.

## 11. DP
The DP production hook consumes only the accepted, frozen S06 mechanism: the clipping norm,
noise multiplier, sample rate, step count, seed and adjacency are read from it, so the engine
cannot be pointed at values the accountant did not charge for. No alternate accountant, epsilon
calculator, clipping rule or sampling interpretation was introduced — `RDPAccountant` and
`get_epsilon` appear nowhere under `src/backend`. Epsilon comes from `src.dp.mechanism.account`
and from nothing else [AUTH: 00 §8.2].

`llama_3_2_3b` is DP_PRIMARY and `qwen3_5_4b_base` DP_SECONDARY; a model with no DP role cannot
be given one. Real DP compatibility remains hardware- and model-deferred.

## 12. Compatibility command
```text
 1 exact repo/revision      2 model loads             3 BF16 text-only forward
 4 no required checkpoint quantization                5 multimodal frozen
 6 rank-32 MLP LoRA attaches                          7 adapter round trip
 8 induced BA extraction    9 linear merge           10 C1 exact inversion
11 deterministic scorer    12 H100 profile           13 DP smoke where applicable
```
Deterministic fixture state at acceptance:
```text
PASS = 10 · NOT_RUN = 3 · FAIL = 0 · OVERALL = NOT_RUN

model_loads             → BACKEND_NOT_INSTALLED
bf16_text_only_forward  → BACKEND_NOT_INSTALLED
h100_profile_collectable→ NO_H100
```
Intentional, and not an unresolved software failure: the CUDA-coupled science set resolves on
the H100 image and is absent from the CPU/dev lane by design, and the H100 row closes only on
real hardware. Each NOT_RUN row carries its own stable cause; a reasonless NOT_RUN row is
refused, because the cause is the only information such a row holds.

## 13. C1 and determinism checks
The inversion row runs the **accepted** stack rather than re-deriving the algebra locally:
```text
build_release_family → observe_known_partner → recover_c1 → bind_truth →
parameter_recovery_metrics                          e_F ≈ 1.838e-08  ≤  1e-5
```
A hand-written inversion inside the backend would only prove the backend can invert its own
arithmetic; what the contract needs is whether the accepted pipeline works on this model's
extracted surface.

The determinism row invokes the scoring path **twice, independently**, and requires finite,
bitwise-identical scalars. A precomputed pair — one list compared against a copy of itself —
would pass on a non-deterministic backend and is explicitly not what runs. The tolerance is
supplied by the contract, not by a caller who could relax it to certify changed outputs.

## 14. H100 check
A non-H100 host, an injected fixture accelerator and an A100 all fail to certify the H100
compatibility row: the check asks whether an H100 is viable for this model, and no CPU host can
answer it. Only a validated genuine NVIDIA H100 profile may PASS.
```text
H100_PROFILE = NOT_RUN(NO_H100)
```

## 15. Evidence provenance
Every record binds: the external RUN_ID, the git commit, the environment lock, the model
manifest, the exact model revision, the backend code identity, the scoring code identity, the
resolved config identity, artifact hashes, start and end timestamps, the exit code, and the
full check bank with its reasons. Verification recomputes each binding against the *current*
state rather than comparing the file to itself.

The backend does **not** mint the scientific RUN_ID: 01 §15/§16 run identity belongs to the S09
run-manifest layer, and an externally issued RUN_ID is consumed via `--run-id`. The command's
own local identifier is `attempt_id`, named so it cannot be read as scientific provenance. An
unresolvable git commit refuses to issue evidence rather than writing an all-zero placeholder.

```text
status algebra   ANY FAIL → FAIL · else ANY NOT_RUN → NOT_RUN · else PASS
exit invariant   (exit_code == 0) == (status == PASS)
```
Final six-way verification: `PASS/0` accepted · `PASS/1` rejected · `NOT_RUN/0` rejected ·
`NOT_RUN/1` accepted · `FAIL/0` rejected · `FAIL/1` accepted.

An unresolved revision produces a clean `NOT_RUN(MODEL_REVISION_NOT_FROZEN)`, a non-zero exit,
no traceback and **no evidentiary file** — an unacquired model is not eligible for evidentiary
compatibility evidence.

## 16. Review history
Initial independent adversarial review found five MAJOR backend defects: the evidence CLI
crashing on an unresolved revision; untrusted LoRA read semantics; an unvalidated target
surface; the backend runtime config absent from the cache identity; and the checkpoint
quantization check inspecting the panel manifest rather than the checkpoint. All five were
repaired, together with thirteen adjacent findings (revision normalisation, `use_rslora`
fail-closed resolution, evidence status aggregation, git-commit fail-closed, the RUN_ID
boundary, specific NOT_RUN reasons, the accepted-C1 and two-execution determinism checks, the
H100 row and the truncation policy).

The subsequent changed-surface review found one final evidence consistency defect: a `NOT_RUN`
or `FAIL` record carrying `exit_code 0` passed verification, because only `PASS ⇒ exit 0` was
enforced. It was repaired with the biconditional `(exit_code == 0) == (status == PASS)`.

Final independent verification returned PASS, `NEW_BLOCKER = none`, `NEW_MAJOR = none`,
`RECOMMENDATION = CLOSE_PRE_S09_BACKEND`.

## 17. Test gate and protected state
```text
UNIT = 1392 passed · INTEGRATION = 551 passed · SYNTHETIC = 182 passed
FORMAT = PASS · LINT = PASS · TYPECHECK = PASS · INVARIANTS = 0 violations
BACKEND_CONTRACT = NOT_RUN(BACKEND_NOT_INTEGRATED) · GPU_SMOKE = not run

S00_UNCHANGED = TRUE · BLOCK_A_PRODUCTION_UNCHANGED = TRUE
BLOCK_C_SCIENCE_UNCHANGED = TRUE · S07_PRODUCTION_UNCHANGED = TRUE
S08_PRODUCTION_UNCHANGED = TRUE · BLOCK_B_SCIENCE_MODIFIED = FALSE
pyproject.toml unchanged · uv.lock unchanged
```
No package install, no network, no real model, no training, no H100, no privacy outcome.

## 18. Readiness state
```text
BACKEND_CODE_IMPLEMENTATION = CLOSED
REAL_MODEL_COMPATIBILITY    = NOT_RUN
BACKEND_INTEGRATED          = FALSE
P0_PRE_READY                = FALSE
```
The last three remain false because real backend and H100 execution are still pending. A green
synthetic suite does not override them [AUTH: 02 §C6].

## 19. Next
```text
NEXT_STAGE = S09_REGISTRY_ORCHESTRATOR
```
S09 issues and binds experiment and run identities and encodes the frozen experiment registry.
After accepted S09: `NEXT_EXECUTION_STAGE = S10_H100_BENCHMARK`.

## 20. Verdict
```text
PRE_S09_BACKEND_STATUS=ACCEPTED
UNRESOLVED_BLOCKER=NONE
UNRESOLVED_MAJOR=NONE
HF_BACKEND_CODE=CLOSED
PEFT_BACKEND_CODE=CLOSED
ADAPTER_VALIDATION=CLOSED
INDUCED_UPDATE_EXTRACTION=CLOSED
SCORER_BACKEND=CLOSED
CACHE_BACKEND_IDENTITY=CLOSED
DP_BACKEND_HOOK=CLOSED
COMPATIBILITY_COMMAND=CLOSED
EVIDENCE_BINDING=CLOSED
REAL_MODEL_COMPATIBILITY=NOT_RUN
BACKEND_INTEGRATED=FALSE
P0_PRE_READY=FALSE
REAL_MODEL=FALSE
TRAINING=FALSE
H100=FALSE
PRIVACY_OUTCOMES=FALSE
NEXT_STAGE=S09_REGISTRY_ORCHESTRATOR
```
