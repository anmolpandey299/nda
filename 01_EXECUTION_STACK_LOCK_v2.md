# 01_EXECUTION_STACK_LOCK

**Project:** Privacy Leakage / Recoverability in Ordinary Model-Merge Release Families  
**Date locked:** 20 August 2026  
**Version:** `v2.0`
**Status:** `PRE-EXECUTION AUTHORITY — REPLACES v1`  
**Controlling scientific specification:** `00_MEASUREMENT_SPEC_v1.8_FINAL.md`

---

# 0. Purpose

This file freezes **how the study is engineered and reviewed**.

`00_MEASUREMENT_SPEC_v1.8_FINAL.md` defines what must be measured.

This file defines:

- which AI systems may architect, implement, review, and adjudicate code;
- which research models are used;
- how H100 compute is used;
- repository architecture;
- model/version/data/environment provenance;
- stage boundaries;
- review independence;
- acceptance evidence;
- reproducibility requirements;
- rules for changing the execution stack.

No implementation may silently override this document.

---

# 1. Separation of concerns

The project has three completely separate model roles.

## 1.1 Software-engineering model

Used to:

- architect the repository;
- implement code;
- write tests;
- debug;
- refactor;
- produce stage evidence.

This role is **not an experimental subject**.

## 1.2 Independent engineering/research reviewer models

Used to:

- inspect code/diffs/tests;
- challenge statistical implementation;
- identify leakage, confounds, incorrect inference, reproducibility failure, and spec drift;
- never modify the implementation while reviewing it.

These models are **not experimental subjects**.

## 1.3 Research subject models

These are the language models trained, merged, reconstructed, and audited in the scientific study.

Their weights, revisions, training conditions, seeds, and outputs are part of the scientific evidence.

The three roles must never be conflated.

---

# 2. Software-engineering model lock

## 2.1 Primary architect + implementer

```text
TOOL  = Claude Code
MODEL = strongest Claude Opus-class coding model available in the user's Claude Code account at S00 freeze
ROLE  = ARCHITECT + IMPLEMENTER
```

At S00, record:

```text
Claude Code client version
exact exposed Claude model name
date/time UTC
permission mode
repository commit
```

After S00 begins, do not silently change the coding model mid-stage.

Claude Code remains the sole primary code-authoring agent unless this lock is formally revised.

---

# 3. Claude role separation

Claude occupies three roles, always in **fresh sessions**.

## 3.1 ARCHITECT

Permissions:

```text
READ
SEARCH
PLAN
NO EDIT
NO COMMIT
```

Produces:

```text
STAGE_PLAN.md
```

The plan contains:

- controlling specification sections;
- bounded objective;
- exact files expected to change;
- interfaces;
- invariants;
- tests;
- failure branches;
- generated artifacts;
- acceptance criteria;
- explicit non-goals.

---

## 3.2 IMPLEMENTER

Fresh Claude Code session.

Responsibilities:

```text
implement bounded stage
write tests
run tests
emit evidence bundle
no unrelated redesign
no scientific-threshold changes
no result-driven tuning
```

Output:

```text
code
tests
STAGE_IMPLEMENTATION_REPORT.md
raw logs
artifact hashes
```

---

## 3.3 CLAUDE ADVERSARIAL REVIEWER

Fresh Claude Code session.

Reviewer receives only:

```text
scientific spec
execution lock
stage plan
git diff
relevant source
test source
raw test outputs
artifact manifests
```

Do **not** give the reviewer the implementer's persuasive rationale.

Permissions:

```text
READ ONLY
NO EDIT
NO COMMIT
NO AUTO-FIX
```

Verdict:

```text
PASS
PASS_WITH_NONBLOCKING_NOTES
FAIL
```

Findings:

```text
BLOCKER
MAJOR
MINOR
NOTE
```

Every Claude implementation stage must receive this separate adversarial review before acceptance.

---

# 4. Independent cross-review lock — OpenAI Codex

The cross-vendor engineering reviewer is:

```text
TOOL = OpenAI Codex
ROLE = INDEPENDENT CODE / REPRODUCIBILITY REVIEWER
MODE = review-only / suggest / no-edit
```

OpenRouter is **not** used for code review.

Codex is used because it is an independent coding system from Claude and can inspect a repository, execute tests, and perform code review.

At the first Codex review, record:

```text
Codex client/product
client version where exposed
exact model name where exposed
review date/time UTC
prompt SHA256
repository commit
review output SHA256
```

If the underlying model is not exposed by the product, record that explicitly rather than inventing a model identifier.

## 4.1 Codex permissions

```text
READ
RUN TESTS
INSPECT DIFFS
NO PRODUCTION EDITS
NO COMMIT
```

A Codex reviewer may create scratch diagnostics outside authoritative source paths only if those diagnostics are included in the review bundle.

## 4.2 Mandatory Codex reviews

Codex independent review is mandatory for:

```text
S01 provenance/config
S03 scorer/cache/cross-fit
S04 statistical engine
S05 data/split pipeline
S06 training + DP plumbing
S07 merge operators
S08 recovery
S09 registry/orchestrator
S10 benchmark path
P0-A measurement-power gate
P0 final gate
P1 final gate
results freeze
reproducibility release
```

For documentation-only or cosmetic edits, Claude adversarial review is sufficient.

---

# 5. Final adjudicator

After:

```text
Claude implementation
Claude adversarial review
required fixes
re-tests
Codex independent review where mandatory
```

the Stage Acceptance Bundle is sent to ChatGPT.

Final stage decision:

```text
ACCEPT
ACCEPT_WITH_NONBLOCKING_NOTES
REJECT
```

No `REJECT` stage enters the protected experiment branch.

The adjudication is based on raw evidence, not reviewer confidence.

---

# 6. Experimental compute/API separation

The engineering-review stack and scientific-compute stack are deliberately separate.

## 6.1 H100 SXM

RunPod H100 SXM is reserved for **scientific execution**:

```text
training
DP-LoRA
merge construction
recovery
privacy scoring
fidelity ladders
model-family replication
scale replication
```

It is not used to host coding/reviewer agents.

## 6.2 OpenRouter

OpenRouter is reserved for **scientific experimental use only**.

It is not used for:

```text
code generation
code review
stage adjudication
statistical interpretation
selecting interesting results
```

OpenRouter is not automatically inserted into the methodology simply because access exists.

An OpenRouter model can enter the scientific study only if a future pre-result experiment specification gives it:

```text
an explicit scientific role
a fixed model slug
a fixed endpoint/provider policy
a reproducible prompt/config
a reason it cannot be replaced by local weights
```

The current core privacy/model-merging experiment does **not require OpenRouter**.

---

# 7. Research subject model philosophy

The core scientific comparison should use:

> the newest strong **base/pretrained, open-weight, text-trainable checkpoints** that preserve direct weight access and a clean LoRA/merge observation model.

"Latest" does not mean blindly using the largest current flagship.

A model is core-eligible only if it supports:

```text
base/pretrained checkpoint
full weight access
BF16/FP32 loading without quantization
text-only forward path
language-trunk LoRA
adapter save/load
deterministic parameter extraction
ordinary parameter merging
white-box scoring
exact model revision pinning
```

Mixture-of-experts or quantized-only models are separate regimes unless explicitly authorized.

---

# 8. Core four-family research-model lock

The core panel is intentionally near the 3–5B language-model scale.

## 8.1 Qwen — PRIMARY MODERN MODEL

```text
model_id = Qwen/Qwen3.5-4B-Base
role     = PRIMARY_DEEP_GRID
family   = Qwen3.5
type     = pretrained/base
```

Rationale:

- current Qwen3.5 base family;
- approximately 4B language model;
- model card explicitly supports fine-tuning / LoRA-style PEFT;
- modern hybrid Gated DeltaNet + attention architecture;
- Apache 2.0;
- strong contrast with classical Llama-style transformers.

Use:

```text
FULL P0
FULL MVRS
FULL P1 if compatibility gate passes
```

Freeze the vision path.

The scientific subject is the **language trunk only**.

---

## 8.2 Gemma — MODERN CROSS-FAMILY MODEL

```text
model_id = google/gemma-4-E4B
role     = CORE_CROSS_FAMILY
family   = Gemma 4
type     = pretrained/base
```

Rationale:

- current Gemma 4 base family;
- roughly 4.5B effective language-scale model;
- Apache 2.0;
- materially different architecture, including Per-Layer Embeddings;
- official pretraining-data cutoff: January 2025.

Use text only.

Freeze:

```text
vision encoder
audio encoder
non-language modality adapters
```

LoRA and merging are restricted to the frozen common language-trunk target policy.

Required:

```text
FULL P0 core
FULL MVRS
P1 replication after compatibility gate
```

---

## 8.3 Mistral — MODERN CROSS-FAMILY MODEL

```text
model_id = mistralai/Ministral-3-3B-Base-2512
role     = CORE_CROSS_FAMILY
family   = Ministral 3
type     = pretrained/base
```

Rationale:

- current compact Mistral base model family;
- approximately 3B language model plus a frozen vision encoder;
- explicitly intended for custom post-training;
- Apache 2.0;
- straightforward dense language trunk.

Use:

```text
FULL P0 core
FULL MVRS
P1 replication after compatibility gate
```

Freeze the vision encoder.

---

## 8.4 Llama — CLASSICAL DENSE + DP ANCHOR

```text
model_id = meta-llama/Llama-3.2-3B
role     = CLASSICAL_DENSE_ANCHOR + PRIMARY_DP_ANCHOR
family   = Llama 3.2
type     = pretrained/base
```

This is deliberately **not** Llama 4.

Reason:

- Llama 3.2 3B is the current compact dense pretrained Llama suitable for clean LoRA merging;
- its pretraining-data cutoff is documented as December 2023;
- its architecture is mature and provides the cleanest baseline for DP-LoRA tooling;
- it anchors the study against a conventional dense transformer rather than making every subject a 2026 multimodal/hybrid architecture.

Use:

```text
FULL P0
FULL MVRS
FULL P1
PRIMARY representative DP-LoRA arm
```

---

# 8A. Why Llama 4 is not a core model

The current Llama 4 Scout base checkpoint is:

```text
17B active parameters
109B total parameters
16 experts
MoE
native multimodal
```

The current model card states an August 2024 knowledge cutoff.

Llama 4 is newer, but using it in the core grid changes the scientific object:

```text
dense -> MoE
single shared parameter path -> routed experts
ordinary low-rank residual geometry -> expert-dependent geometry
single-H100 BF16 feasibility -> multi-GPU or quantized loading
```

Quantizing Scout merely to fit one H100 would additionally change the observation channel.

Therefore:

```text
meta-llama/Llama-4-Scout-17B-16E
role = OPTIONAL_MOE_ARCHITECTURE_STRESS
```

It is not used to define the main effect.

If activated:

```text
BF16 weights
multi-H100 if required
text-only
no quantization
non-DP targeted replication only initially
separate MoE interpretation
```

It answers:

> Does the central phenomenon survive a routed-expert architecture?

It does not replace Llama-3.2-3B.

---

# 8B. Common LoRA target policy

Cross-family comparison requires the parameter object to mean the same thing.

Primary LoRA target:

```text
LANGUAGE_TRUNK_MLP_ONLY
rank = 32
```

Map only each architecture's decoder MLP projection matrices corresponding to:

```text
gate projection
up projection
down projection
```

or the exact architecture-equivalent matrices.

Do not LoRA:

```text
embeddings
LM head
vision tower
audio tower
multimodal projector
router/expert gating
PLE embeddings
```

unless a later robustness experiment explicitly authorizes it.

This target policy gives the cleanest common low-rank object across Qwen, Gemma, Mistral, and Llama.

## 8B.1 Target-policy fallback

Before real privacy outcomes are opened, run a measurement-power pilot.

If `LANGUAGE_TRUNK_MLP_ONLY` fails the predeclared P0-A privacy-power gate, a single pre-registered fallback is permitted:

```text
LANGUAGE_TRUNK_ALL_COMMON_LINEAR
```

This expands to architecture-matched attention + MLP projections.

If the fallback is activated:

- activate it for all core families;
- restart affected P0 experiments from clean checkpoints;
- never mix primary/fallback target-policy results;
- record the transition before viewing merge/recovery outcomes.

The policy is not chosen based on which produces a more interesting result.

---

# 8C. Core model compatibility gate

Before any real membership result is opened, each core checkpoint must pass:

```text
MODEL_COMPATIBILITY_GATE
```

Required tests:

1. exact HF/model repository is accessible;
2. immutable revision SHA captured;
3. BF16 text-only forward passes;
4. no quantization required;
5. multimodal towers can be frozen/ignored;
6. common MLP LoRA rank-32 attaches correctly;
7. LoRA save/load is byte/provenance stable;
8. induced \(\Delta W=BA\) can be extracted exactly;
9. linear merge construction works;
10. P0-C1 exact inversion achieves the spec tolerance;
11. scorer produces deterministic outputs within frozen numerical tolerance;
12. H100 memory/throughput is viable;
13. DP smoke passes for models assigned to the DP arm.

A failure is an engineering incompatibility, not a scientific result.

No model is silently replaced after privacy results are viewed.

---

# 8D. Model execution tiers

## Tier A — mandatory four-family MVRS

All four core models run:

```text
P0-A measurement power
P0-C1 exact oracle
primary rank-matched ladder
C01/C02/C03 four-view audit
reference + Min-K%
3 seeds
```

This creates a four-family answer to the core scientific question.

## Tier B — full mechanistic grid

Mandatory on:

```text
Qwen3.5-4B-Base
Llama-3.2-3B
```

Authorized on Gemma 4 and Ministral 3 after their compatibility gates pass.

No model is selected for full P1 because its preliminary result was larger.

## Tier C — targeted scale replication

After the central result and nulls are frozen, run targeted replication on:

```text
Qwen/Qwen3.5-9B-Base
google/gemma-4-12B
mistralai/Ministral-3-8B-Base-2512
```

Run only the predeclared smallest subset needed to reproduce the main effect/null:

```text
oracle privacy power
local ladder region around the observed transition
C01/C02/C03 lineage audit
strongest operator effect where relevant
```

## Tier D — optional latest-Llama MoE stress

```text
meta-llama/Llama-4-Scout-17B-16E
```

Targeted, non-DP, BF16, separate analysis.

---

# 8E. DP model lock

Formal/representative DP evaluation does **not** need to run on every architecture.

Primary:

```text
DP_PRIMARY = meta-llama/Llama-3.2-3B
```

Secondary, only if the frozen DP stack passes the compatibility smoke without architecture-specific scientific changes:

```text
DP_SECONDARY = Qwen/Qwen3.5-4B-Base
```

Gemma 4 and Ministral 3 DP runs are optional replication.

Failure of a new architecture's DP library integration cannot be repaired by changing clipping/accounting rules after results.

---

# 8F. Natural-data recency lock

The old 2025-PubMed assumption is replaced.

For cross-family natural-record experiments, use one common corpus whose **publication date** satisfies:

```text
publication_date >= 2026-04-01
```

and whose IDs are frozen at data-manifest creation.

Why:

- Qwen3.5 small base checkpoints were publicly available by March 2026, while their model card does not expose a precise training-data cutoff;
- Gemma 4 documents a January 2025 training cutoff;
- Ministral 3 was released in December 2025;
- Llama 3.2 documents a December 2023 cutoff;
- Llama 4 documents an August 2024 cutoff.

Thus a publication-date floor of 1 April 2026 gives one conservative post-training/post-release natural corpus for the core panel without relying on an undocumented Qwen/Mistral cutoff.

The exact PubMed query, retrieval timestamp, PMIDs, publication dates, deduplication, and hashes are frozen in the data manifest.

Indexing date is not accepted as a substitute for publication date.

---

# 8G. Research-model provenance

For every model:

```text
model_id
exact revision SHA
base vs instruct
architecture family
total parameters
language parameters/effective parameters where relevant
dtype
license
training-data cutoff if officially disclosed
download timestamp
config SHA256
tokenizer SHA256
weight-file SHA256/Xet identifier
LoRA target mapping
frozen/non-frozen module list
```

No floating `main` revision enters an evidentiary run.

---

# 9. H100 SXM compute lock

Primary scientific compute:

```text
provider = RunPod
GPU      = NVIDIA H100 SXM 80GB
```

Use H100s for:

```text
all research-subject training
DP-LoRA
model merging
recovery
scoring
fidelity ladders
four-family replication
scale replication
```

Because compute budget is not a limiting factor, parallelize **independent seeds and model families**, not unnecessary distributed training.

Preferred scheduling:

```text
worker 1 -> model/family A, seed 101
worker 2 -> model/family A, seed 202
worker 3 -> model/family A, seed 303

additional workers -> independent model families
```

For models that fit one H100 in BF16, do not use tensor-parallel complexity.

For optional Llama 4 Scout BF16, multi-H100 is permitted because it is a separately declared MoE stress regime.

# 10. Precision policy

## Training

Default:

```text
BF16
```

unless a component formally requires FP32.

## Merge / recovery arithmetic

For numerical oracle checks and quantities where rounding matters:

```text
FP32
```

as required by the measurement spec.

## Metrics/statistics

Store derived scalar quantities in:

```text
float64
```

where practical.

Never allow an AMP/autocast default to silently determine scientific arithmetic.

Every run manifest records the actual dtype of:

```text
base weights
LoRA weights
optimizer states
merge arithmetic
recovery arithmetic
forward scoring
stored statistics
```

---

# 11. Repository authority

Recommended repository:

```text
privacy-model-merging/
│
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── Makefile
│
├── specs/
│   ├── 00_MEASUREMENT_SPEC_v1.8_FINAL.md
│   └── 01_EXECUTION_STACK_LOCK.md
│
├── configs/
│   ├── models/
│   ├── data/
│   ├── training/
│   ├── attacks/
│   ├── recovery/
│   ├── p0/
│   └── p1/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── dp/
│   ├── merge/
│   ├── recovery/
│   ├── scoring/
│   ├── attacks/
│   ├── analysis/
│   ├── provenance/
│   └── cli/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   ├── synthetic/
│   └── gpu_smoke/
│
├── manifests/
│   ├── models/
│   ├── data/
│   ├── environments/
│   └── runs/
│
├── results/
│   ├── p0/
│   └── p1/
│
├── artifacts/
├── reviews/
├── logs/
└── scripts/
```

---

# 12. Environment lock

Use:

```text
Python + pyproject.toml + uv.lock
Dockerfile
```

The first environment stage freezes:

```text
Python version
PyTorch version
CUDA runtime
Transformers
PEFT
Accelerate
DP library
NumPy
SciPy
scikit-learn
pandas
safetensors
tokenizers
plotting libraries
test/lint/typecheck tooling
```

Do not hand-write guessed versions into the scientific spec.

Instead:

1. Claude proposes one mutually compatible environment.
2. Install on the actual H100 image.
3. Run compatibility smoke tests.
4. Freeze exact versions.
5. Hash `uv.lock`.
6. Record `nvidia-smi`.
7. Record CUDA driver/runtime.
8. Build/tag Docker image.
9. Record Docker image digest.

No `pip install -U` occurs during an experiment run.

---

# 13. Model provenance manifest

Every research model has:

```text
manifests/models/<model_alias>.json
```

Required fields:

```json
{
  "model_id": "...",
  "revision": "...",
  "config_sha256": "...",
  "tokenizer_sha256": "...",
  "license_snapshot_sha256": "...",
  "download_timestamp_utc": "...",
  "parameter_count": 0,
  "dtype_on_disk": "...",
  "role": "..."
}
```

---

# 14. Data provenance manifest

Every dataset version has:

```text
manifest hash
source query/version
download timestamp
raw-data hash
preprocessing-code commit
preprocessing-config hash
split RNG seed
split IDs
deduplication method/version
final split hashes
```

Canary manifests are immutable.

No record may move between member/calibration/evaluation partitions after privacy results are opened.

---

# 15. Run identity

Every experiment run receives a deterministic identity derived from:

```text
git commit SHA
scientific spec SHA256
execution lock SHA256
config SHA256
model revision
data manifest SHA256
environment lock SHA256
training seed
```

Conceptually:

```text
RUN_ID = SHA256(all provenance inputs)
```

The run writes:

```text
manifests/runs/<RUN_ID>.json
```

before training/scoring begins.

---

# 16. Run manifest minimum fields

Every run records:

```text
RUN_ID
git commit
git dirty status
spec hash
execution-lock hash
model revision
tokenizer hash
data-manifest hash
environment hash
GPU model
GPU UUID where available
CUDA
PyTorch
precision
seed(s)
config
wall-clock start/end
exit code
artifact paths
artifact hashes
metrics paths
stdout/stderr log paths
```

A scientific result without a valid run manifest is:

```text
NON_EVIDENTIARY
```

and cannot enter the dissertation/paper.

---

# 17. Configuration system

All scientific choices must come from version-controlled config.

No material experimental constant may live only in source code.

Examples:

```text
model
rank
dataset
seed
alpha
k
operator
DARE p
SVD rank
attack
target FPR
Min-K fraction
canary count
evaluation set
bootstrap replicates
```

Each CLI invocation must be reconstructable from a saved resolved config.

---

# 18. CLI rule

Every scientific stage must be runnable non-interactively.

Example shape:

```text
python -m src.cli.train --config configs/p0/train_seed101.yaml
python -m src.cli.merge --config configs/p0/c01.yaml
python -m src.cli.score --config configs/p0/score_c01.yaml
python -m src.cli.analyse --config configs/p0/analysis.yaml
```

No notebook is the authoritative execution path.

Notebooks may be used only for exploration/visualization from already-produced immutable result tables.

---

# 19. Result storage

Authoritative machine-readable results:

```text
Parquet
JSON
JSONL
```

Use CSV only where the measurement spec explicitly requests it or interoperability requires it.

Every result table includes provenance keys:

```text
run_id
artifact_id
seed
model_revision
config_hash
data_manifest_hash
code_commit
```

---

# 20. Tiny deterministic test model

Unit/integration testing must not depend on downloading a real research model.

Instantiate a tiny Llama-style model from a fixed local Transformers config with a fixed RNG seed.

Use it for:

- LoRA attach/save/load;
- merge algebra;
- score plumbing;
- cache;
- reconstruction shape checks;
- end-to-end CLI tests.

This model is not a research subject.

---

# 21. GPU smoke lane

Before a full experimental run, use the real primary model with a tiny synthetic/data subset.

Example:

```text
Llama-3.2-1B
32–128 records
1 short training epoch / bounded steps
one LoRA adapter
one merge
one scoring pass
```

The smoke run tests infrastructure only.

Smoke outputs are never mixed into scientific results.

---

# 22. Test hierarchy

Every stage must pass, in order:

```text
UNIT
→ INTEGRATION
→ SYNTHETIC/GOLDEN
→ GPU SMOKE where applicable
→ SCIENTIFIC GATE
```

## Unit

Pure functions and mathematical identities.

## Integration

Multiple modules with tiny fixtures.

## Synthetic/golden

Known planted truths from `00_MEASUREMENT_SPEC_v1.8_FINAL.md`.

## GPU smoke

Actual model/library/runtime compatibility.

## Scientific gate

Real experiment.

---

# 23. Critical invariants that must have tests

At minimum:

```text
exact C1 linear inversion
same-observation R used for compared recovery methods
no evaluation IDs in calibration
no calibration/evaluation overlap
arm-matched thresholding
pooled view never invokes model
cache version invalidation
base-reference output cache consistency
Min-K inline reduction
realised FPR reporting
no extrapolation beyond e support
DP smoke cannot enter inferential tables
zero-delta anchor exactly e=1
rank-matched ladder reaches target e tolerance
O3 analytic floor
varying-alpha rescaling identity
run-manifest completeness
```

---

# 24. OpenRouter experimental-use policy

OpenRouter is **not part of the engineering review chain**.

Review chain:

```text
Claude Code
→ fresh Claude adversarial reviewer
→ Codex independent reviewer
→ ChatGPT adjudication
```

OpenRouter credentials are reserved for scientific experiments only.

The current weight-space privacy experiment should prefer local open-weight models on H100 because:

- parameter access is required;
- reproducibility is stronger;
- token-level scoring is controlled;
- merge/recovery requires weights.

Therefore:

```text
OPENROUTER_DEFAULT_ROLE = UNUSED_RESERVED_EXPERIMENTAL_RESOURCE
```

If an API-only experimental baseline is later scientifically justified, freeze before execution:

```text
exact model slug
provider routing
API parameters
prompt
request schema
date
response hashes
fallback policy = NONE
```

No OpenRouter output may select experimental cells, seeds, thresholds, or headline results.

---

# 25. Reviewer independence rules

A reviewer must not:

- edit the code it is judging;
- see the implementer's chain of thought;
- inherit the implementer's conversational session;
- accept a claim because tests are green;
- infer missing evidence from the implementation report.

A reviewer must independently inspect:

```text
spec
diff
tests
raw outputs
failure paths
```

---

# 26. Stage Acceptance Bundle

Every stage submitted for final adjudication contains exactly:

```text
1. STAGE_ID
2. bounded objective
3. controlling spec sections
4. base git commit
5. head git commit
6. git diff / patch
7. changed-file list
8. resolved config(s)
9. unit-test log
10. integration-test log
11. synthetic/golden-test log
12. GPU-smoke log where relevant
13. produced artifact manifest + SHA256
14. implementer report
15. Claude reviewer report
16. GPT-5.6 Sol Pro report where mandatory
17. GLM 5.2 report if invoked
18. list of unresolved findings
19. explicit statement of whether real scientific data were inspected
20. requested verdict: ACCEPT / REJECT
```

No narrative summary substitutes for raw evidence.

---

# 27. Git workflow

Protected branches:

```text
main
experiment-frozen
```

Development:

```text
stage/<stage-id>
```

Rules:

- one bounded stage per branch;
- no drive-by refactors;
- no experiment code changes during a live run;
- tag every accepted stage;
- preserve failed experimental runs and their manifests;
- never rewrite published experiment history.

Example tags:

```text
stage-p0-pre-v1
stage-training-v1
stage-scoring-v1
p0-freeze
p1-freeze
paper-results-freeze
```

---

# 28. Claude Code repository memory

`CLAUDE.md` must be short and operational.

It should contain:

```text
project purpose
controlling docs
non-negotiable scientific invariants
repo commands
test commands
stage workflow
review workflow
forbidden actions
artifact locations
```

It must **not** duplicate the 6,000-line measurement spec.

Claude is told:

> Read the relevant spec sections for the current task. Never summarize the entire spec into memory and then work from the summary alone.

---

# 29. No-result-peeking rule

Until a component's code/tests/review are frozen:

```text
do not inspect the corresponding real scientific outcome
```

Examples:

- freeze attack implementation before comparing member/nonmember performance;
- freeze operator gate code before privacy outcomes;
- freeze recovery code before privacy recovery outcomes;
- freeze statistical scripts before final result tables.

If an implementation bug is found after peeking:

```text
SPEC_DEVIATION / IMPLEMENTATION_REPAIR
```

must document it.

---

# 30. Training reproducibility

For each training seed record separately:

```text
Python RNG
NumPy RNG
PyTorch CPU RNG
PyTorch CUDA RNG
data-order seed
canary-inclusion seed
LoRA initialization seed
DP/noise RNG where applicable
```

Do not use one undocumented global seed as a substitute.

Where CUDA kernels are nondeterministic, document the exact source and measure run-to-run tolerance.

---

# 31. H100 parallelization

If multiple H100 SXM GPUs are available:

Parallelize across **independent experimental units**:

```text
seed 101 -> GPU/worker A
seed 202 -> GPU/worker B
seed 303 -> GPU/worker C
```

Prefer this over distributed training of a 1B model.

Reason:

- easier reproducibility;
- less distributed-system complexity;
- genuine wall-clock acceleration;
- seed independence remains explicit.

Do not run multiple seeds in one process.

---

# 32. Research model serving

For scoring, prefer direct local PyTorch/Transformers execution first.

Only introduce:

```text
vLLM
TensorRT-LLM
custom serving
```

if the benchmark shows direct batched scoring cannot meet execution needs.

A serving-engine change changes the numerical/software observation pipeline and requires:

```text
equivalence test
new environment hash
review
```

The simplest correct implementation wins.

---

# 33. Code-quality gate

Before scientific use:

```text
format = PASS
lint = PASS
typecheck = PASS for typed core modules
unit = PASS
integration = PASS
synthetic = PASS
GPU smoke = PASS
```

Recommended tools may include:

```text
ruff
mypy or pyright
pytest
coverage
```

Exact versions are frozen in the environment lock.

Coverage percentage is not itself a correctness criterion.

Critical invariants matter more than global line coverage.

---

# 34. Statistical-code rule

Scientific statistics live in importable tested Python modules.

Figures call those modules.

Do not reimplement statistics inside plotting scripts.

For example:

```text
src/analysis/metrics.py
src/analysis/bootstrap.py
src/analysis/isotonic.py
src/analysis/audit_gaps.py
src/analysis/tail.py
```

Tests include planted truths and nulls.

---

# 35. Experiment scheduler

Claude may build a lightweight launcher that:

1. resolves config;
2. creates RUN_ID;
3. verifies clean/frozen code state;
4. records manifest;
5. executes one run;
6. captures logs;
7. hashes outputs;
8. marks status.

Do not build a giant autonomous multi-agent research platform.

The repository exists to run the science, not to become the research project itself.

---

# 36. Failure handling

Every run ends in exactly one state:

```text
SUCCESS
FAILED_IMPLEMENTATION
FAILED_ENVIRONMENT
FAILED_RESOURCE
FAILED_SCIENTIFIC_GATE
CANCELLED
```

Never overwrite a failed run directory.

Reruns receive a new RUN_ID.

---

# 37. AI-output provenance

AI-generated code is treated exactly like human-generated code:

```text
not trusted until tested/reviewed
```

Do not cite AI model output as scientific evidence.

AI-review reports are process artifacts, not paper evidence.

The paper's evidence is:

```text
formal derivation
code
data
model artifacts
experiment outputs
statistical analysis
literature
```

---

# 38. Research-result model independence

OpenRouter frontier models must never:

- label membership examples;
- choose which real experimental cells to report;
- decide which seeds are outliers;
- tune thresholds using evaluation results;
- select “interesting” outcomes for the paper.

Those decisions are fixed by the measurement specification and statistical code.

AI can help interpret results only **after** tables are generated under the frozen analysis.

---

# 39. Stage build order

Use this exact engineering order.

## S00 — repository bootstrap

Build:

```text
repo structure
CLAUDE.md
pyproject
uv lock
Dockerfile
Makefile
spec copies
git hooks
CI skeleton
```

No research model run.

---

## S01 — provenance/config core

Build:

```text
config resolver
SHA256 utilities
model manifest
data manifest
run manifest
run ID
artifact hashing
immutable run directory
```

This is foundational.

---

## S02 — deterministic tiny-model fixtures

Build:

```text
tiny local Llama fixture
tiny corpus
tiny LoRA
known merge fixtures
```

No network dependency for tests.

---

## S03 — canonical scoring/cache/cross-fit engine

Port and harden the v1.8 scaffold:

```text
parameterised scorer
reference loss
Min-K%
content-addressed cache
cross-fitting controller
pooled score transforms
```

Run DRY-H/cache tests.

---

## S04 — statistical analysis engine

Implement:

```text
eligibility
R_priv
R_func
D_L/D_R
G_L/G_R
isotonic
bootstrap
e50
residuals
spectral-tail logic
null handling
censoring
```

Run DRY-A…G.

No real outcomes yet.

---

## S05 — data pipeline

Build:

```text
PubMed acquisition
normalization
dedup
splits
canaries
blind split/dedup test
manifesting
```

Freeze IDs and hashes.

---

## S06 — training pipeline

Build:

```text
base loader
LoRA rank 32
three-seed training
no-canary control
DP smoke
checkpoint/adaptor manifesting
```

GPU smoke before full training.

---

## S07 — merge operators

Implement:

```text
linear / task arithmetic
DARE
conditional O3 SVD truncation
```

Each gets mathematical unit tests.

No TIES yet.

---

## S08 — recovery

Implement:

```text
C1 exact inversion
C2 spectral/common-source reproduction
C2B known-varying-alpha rescaling
conditional O3 recovery
conditioning metrics
```

---

## S09 — experiment registry/orchestrator

Encode P0/P1 registry as data/config.

The launcher does not invent cells.

---

## S10 — real 1,000-sequence H100 benchmark

Only now run the measurement-spec benchmark.

Freeze:

```text
actual throughput
batch size
memory peak
sequence-length distribution
P0 projection
P1 projection
```

---

## S11 — P0-A0 / P0-A

First real scientific measurements.

After this point, result-peeking rules are active.

---

## S12+ — execute measurement spec

Follow:

```text
00_MEASUREMENT_SPEC_v1.8_FINAL.md
```

exactly.

---

# 40. Stage review cadence

For every S00–S10 stage:

```text
Claude architecture
→ Claude implementation
→ deterministic tests
→ fresh Claude adversarial review
→ fixes
→ full re-test
```

Codex independent review is mandatory at:

```text
S01
S03
S04
S05
S06
S07
S08
S09
S10
```

ChatGPT final adjudication occurs after:

```text
S01
S03
S04
S05
S06
S08
S10
P0-A
P0 gate
P1 gate
results freeze
```

No OpenRouter reviewer is part of this workflow.

---

# 41. No monolithic Claude prompt

Claude is never told:

> Build the entire dissertation repository.

Every implementation prompt is bounded to one stage.

A large context window is not a substitute for stage boundaries.

---

# 42. Claude implementation prompt contract

Every implementation prompt contains exactly four blocks:

## A. Authority

```text
controlling specification
execution lock
stage plan
current accepted commit
```

## B. Bounded task

Exact files/features to implement.

## C. Acceptance tests

Exact commands/artifacts expected.

## D. Constraints

```text
do not modify unrelated files
do not weaken tests
do not change scientific thresholds
do not inspect forbidden results
do not commit unless explicitly asked
```

---

# 43. Claude reviewer prompt contract

Every reviewer prompt contains:

```text
ROLE = adversarial independent reviewer
NO EDITS
```

Then asks:

1. Does implementation satisfy the exact spec?
2. Can any test pass while implementation is wrong?
3. Is there any member/evaluation leakage?
4. Is there oracle information in the attacker path?
5. Is any result silently selected/tuned?
6. Is cache/provenance invalidation correct?
7. Are mathematical identities implemented in correct coordinates?
8. Can null results be reported honestly?
9. Could numerical precision manufacture the result?
10. Is the stage reproducible from a clean machine/H100?

The reviewer must search for counterexamples.

---

# 44. Codex independent-review prompt contract

Codex receives:

```text
spec excerpts
execution-lock excerpts
stage plan
git diff
relevant source
tests
raw logs
artifact manifests
Claude adversarial review
```

Codex is placed in review-only mode.

Prompt objective:

> Treat the implementation as potentially wrong. Find the smallest concrete counterexample that could invalidate this stage. Check scientific-spec compliance, oracle leakage, evaluation leakage, cache invalidation, numerical assumptions, test weakness, failure handling, reproducibility, and whether the code could manufacture the claimed result. Separate correctness failures from optional improvements. Do not edit production code.

Codex output:

```text
CODEX_REVIEW.md
verdict
BLOCKER/MAJOR/MINOR/NOTE findings
commands executed
tests independently rerun
unresolved risks
```

---

# 45. Final ChatGPT adjudication package

Before asking for acceptance, produce:

```text
STAGE_ACCEPTANCE_BUNDLE/
  00_INDEX.md
  01_PLAN.md
  02_DIFF.patch
  03_TEST_COMMANDS.txt
  04_TEST_OUTPUTS/
  05_ARTIFACT_MANIFEST.json
  06_IMPLEMENTER_REPORT.md
  07_CLAUDE_REVIEW.md
  08_CODEX_REVIEW.md
  09_UNRESOLVED.md
  10_REPRODUCE.md
```

Send that bundle/repo snapshot for adjudication.

---

# 46. Acceptance rule

A stage may be accepted only when:

```text
all BLOCKER findings = CLOSED
all MAJOR findings = CLOSED or explicitly disproven with evidence
required tests = PASS
required negative controls = PASS
artifact hashes = present
repo is reproducible from frozen environment
scientific spec = unchanged
```

A model saying “looks good” is never enough.

---

# 47. Change-control rule

This execution lock may change only because of:

```text
actual library incompatibility
actual H100 benchmark
actual implementation impossibility
direct mathematical error
direct experimental evidence
```

It may not change because:

```text
a model suggests a more fashionable architecture
a reviewer wants additional complexity
a newer coding model is released mid-study
a result is inconvenient
```

If a software model update occurs during the study, the existing primary agent remains locked until the next stage boundary.

Changing coding/reviewer models requires a provenance entry.

Changing **research subject models** after real results are viewed requires an explicit experiment-scope amendment.

---

# 48. AI engineering-model upgrade policy

Do not chase coding-model releases during an active stage.

Current engineering roles:

```text
Claude Code = architect / implementer / first adversarial reviewer
Codex       = independent cross-reviewer
ChatGPT     = acceptance adjudicator
```

At S00, record the exact exposed Claude and Codex versions/models where available.

A coding/reviewer model changes only if:

1. a stage boundary has been reached;
2. the change is recorded before the next stage;
3. a fixed repository-review benchmark is rerun;
4. previous review evidence remains preserved.

No change to a coding/review model changes the scientific result.

Research-subject model changes are governed separately by §8C–§8G and may not occur after privacy-result inspection except through a formal experiment-scope amendment.

---

# 49. What the H100 should NOT be used for

Do not spend project complexity on:

```text
self-hosting a coding agent
training a code-review model
building a custom LLM gateway
distributed training for a 1B model
serving infrastructure before benchmarking requires it
```

The H100's purpose is to reduce scientific uncertainty.

---

# 50. Final locked workflow

```text
SCIENTIFIC SPEC FROZEN
        ↓
EXECUTION STACK v2 FROZEN
        ↓
S00 repo bootstrap
        ↓
S01 provenance/config
        ↓
S02 tiny deterministic fixtures
        ↓
S02B four-family model compatibility gate
        ↓
S03 scorer/cache/cross-fit
        ↓
S04 statistical engine + DRY-A…H
        ↓
S05 post-2026-04-01 data pipeline
        ↓
S06 four-family LoRA training + DP anchor
        ↓
S07 merge
        ↓
S08 recovery
        ↓
S09 registry/orchestrator
        ↓
S10 H100 benchmark
        ↓
P0-A0 / P0-A
        ↓
four-family MVRS
        ↓
full mechanistic P1
        ↓
targeted 9B/12B/8B replication
        ↓
optional Llama-4 MoE stress
        ↓
final reproducibility audit
        ↓
paper/dissertation evidence freeze
```

For every implementation stage:

```text
Claude Architect
→ Claude Implementer
→ deterministic tests
→ fresh Claude Reviewer
→ fixes
→ re-test
→ Codex independent review when required
→ ChatGPT final adjudication
→ accepted tag
```

OpenRouter and H100 remain on the **scientific experiment side**, not the engineering-review side.

---

# 51. Locked principle

The goal is not to maximize the number of agents, models, frameworks, or experiments.

The goal is:

> **one scientifically frozen question, one reproducible repository, one auditable execution path, independently reviewed code, and results that survive hostile scrutiny.**

That is the standard for this project.


# 52. Model-selection evidence snapshot — 20 August 2026

This snapshot explains why the locked models were chosen.

## Qwen

`Qwen/Qwen3.5-4B-Base` is a pretrained Qwen3.5 checkpoint with a 4B language model and explicit model-card guidance for fine-tuning / LoRA-style PEFT.

## Gemma

`google/gemma-4-E4B` is a current Gemma 4 pretrained checkpoint. The official model card states a January 2025 pretraining-data cutoff.

## Mistral

`mistralai/Ministral-3-3B-Base-2512` is the pretrained compact Ministral 3 checkpoint intended for custom post-training.

## Llama dense anchor

`meta-llama/Llama-3.2-3B` is a pretrained dense 3B checkpoint. Its official model card documents a December 2023 knowledge cutoff.

## Llama latest stress

`meta-llama/Llama-4-Scout-17B-16E` is a 109B-total / 17B-active MoE checkpoint with an August 2024 cutoff. Its MoE structure is why it is kept out of the core dense comparison and reserved for a separate architecture-stress experiment.

## Data consequence

The common natural-record corpus is moved to publication dates on or after:

```text
2026-04-01
```

so the core panel does not rely on the old 2025-PubMed assumption.
