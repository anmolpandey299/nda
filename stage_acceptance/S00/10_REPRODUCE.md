# Reproduce S00 from a clean machine

```bash
git clone <repo> && cd privacy-model-merging
git checkout stage/s00

curl -LsSf https://astral.sh/uv/install.sh | sh     # or: pipx install uv
uv sync --extra cpu-dev                             # resolves ONLY from the committed uv.lock

bash scripts/install_git_hooks.sh                   # convenience guard [AUTH: 01 §39 S00]
make preflight                                      # steps 0-8, plan §10.1
```

Expected terminal state:

```text
preflight PASS
artifacts/p0_pre/P0_PRE_READINESS.json
  BACKEND_INTEGRATED = false
  SUITE_SCOPE        = STATISTICAL_STACK_ONLY
  P0_PRE_READY       = false
  environment_lock_sha256 = TBD_REQUIRES_HARDWARE
```

`make synthetic`, `make backend-contract` and `make gpu-smoke` report `NOT_RUN(<reason>)`.
An empty lane never reports `PASS` [AUTH: 02 §C6; 00 §34B.3].

## S00-B — the frozen science environment

The stock RunPod container is **not** the science environment: its Python is 3.11.10, its
preinstalled torch is 2.4.1+cu124 (no CPython 3.13 build), uv is absent, Docker is
unavailable and `/etc/pmm-image.json` does not exist. The probe is recorded, and explicitly
rejected, in `manifests/environments/S00B_HARDWARE_PROBE.json`.

### Selection, frozen in pyproject.toml + uv.lock

```text
SCIENCE_PYTHON      3.13            (satisfies the frozen requires-python >=3.13,<3.14)
SCIENCE_TORCH       2.13.0          (current production release with a cp313 manylinux
                                     x86_64 wheel; the stock 2.4.1 has none)
SCIENCE_CUDA_BUILD  cu130           (driver 580.126.09 reports CUDA 13.0, so this is an
                                     exact match, not a minor-version fallback)
transformers 5.15.1 · peft 0.20.0 · accelerate 1.14.0 · opacus 1.6.0
CUDA runtime        15 pinned nvidia-* wheels in uv.lock, so no mutable NVIDIA base image
```

### Step 1 — build and seal the image, OFF the pod

Docker is unavailable inside a RunPod pod, so the image is built locally or in CI. Nothing
below requires Docker-in-Docker.

```bash
git checkout <candidate-commit>
docker buildx version          # required; a plain `docker build` on Apple Silicon
                               # produces linux/arm64 and RunPod rejects it with
                               # "no matching manifest for linux/amd64"
bash scripts/build_science_image.sh <registry>/<repo> s00b
```

Both passes build with `docker buildx --platform linux/amd64 --push`, take the digest from
buildx metadata rather than the local image store, and fail closed unless
`docker buildx imagetools inspect` shows a `linux/amd64` manifest in the pushed artifact.
`linux/arm64` is deliberately not built: the scientific execution target is a RunPod H100 on
`linux/amd64`.

Two passes, because an image cannot contain its own digest: pass 1 builds and pushes the
`science` stage and reads its immutable digest; pass 2 builds `science-sealed` from that
digest and bakes `image_ref`, `image_digest`, `base_image_digest`, `source_git_commit` and
`lane: "science"` into `/etc/pmm-image.json`. Both digests are written to
`manifests/environments/S00B_IMAGE_RECORD.json`.

### Step 2 — launch RunPod by digest, never by tag

```text
RunPod -> Pods -> Deploy -> H100 SXM 80GB
Container image: <registry>/<repo>@sha256:<sealed_image_digest>
```

The tag moves; the digest does not [AUTH: 01 §12(9)].

### Step 3 — inside the pod, verify and capture

```bash
git clone <repo> /repo && cd /repo
git checkout <candidate-commit>
bash scripts/bootstrap_runpod_s00b.sh <candidate-commit>
```

That single command fails closed on any mismatch and, in order, verifies the exact commit, a
clean tree, an H100, Python 3.13.x, torch 2.13.0+cu130, `torch.cuda.is_available()`, compute
capability (9, 0), and the sealed image metadata; then runs `make env-capture`,
`make gpu-smoke`, `make preflight` and `make bundle-verify`.

### Commit vocabulary — three different commits

```text
DESCRIBED_COMMIT           the scientific candidate the acceptance bundle describes
BUILD_COMMIT               the bundle commit. It CONTAINS the bundle, so the image is built
                           from it; building from DESCRIBED_COMMIT would ship no bundle
IMAGE_SOURCE_GIT_COMMIT    baked into /etc/pmm-image.json at build time == BUILD_COMMIT
BOOTSTRAP_REQUIRED_COMMIT  == BUILD_COMMIT, the argument to bootstrap_runpod_s00b.sh
```

Never bootstrap the scientific candidate; that commit predates its own bundle.

**Obtain every SHA mechanically.** `git rev-parse HEAD`, `git rev-parse HEAD~1`. Never expand
an abbreviated SHA by hand: a hand-expanded prefix is a fabricated object, and the bootstrap
now refuses any argument that is not a full 40-hex SHA naming a real commit.

### S00-B ordering — what runs when, and why

```text
scientific candidate  C           bundle describes C
bundle commit         B = C+1     BUILD_COMMIT; contains the bundle
image built FROM B                digest-pinned, /etc/pmm-image.json bakes B
        |
runtime environment capture         make env-capture -> manifests/environments/<64hex>.json
        |
hardware lane / evidence            make gpu-smoke -> evidence/lanes/gpu_smoke.json
                                    fail-closed: previous evidence is invalidated first, and
                                    PASS is written only for a fully collected suite with
                                    zero skips, zero failures, a resolved environment and a
                                    successful atomic write
        |
ordered preflight  steps 0-6        step 5 integration uses BUNDLE-VERIFY-SOURCE only
        |
readiness re-derivation  step 8     P0_PRE_READINESS.json rewritten for THIS environment
        |
final runtime-aware verification    make bundle-verify   (source AND runtime)
        |
closure record                      make closure -> 12_S00B_CLOSURE.json
                                    binds: described commit, build commit, baked image
                                    source commit and digest, environment manifest, GPU PASS
                                    for THIS environment, readiness, and the SHA256 of every
                                    artifact consumed
```

The split matters. `bundle-verify-source` checks the described commit, source drift, the diff
and the source artifact hashes — none of which execution touches, so it is safe anywhere in
the ordered gate. `bundle-verify` additionally re-derives readiness, which is only meaningful
once step 8 has written readiness for the environment currently in play. Running the full
verifier at step 5 is circular: readiness still holds the pre-run identity, step 5 fails, and
step 8 is never reached, so readiness can never become consistent.

### Image record states

`S00B_IMAGE_RECORD.json` is written by the build, so it cannot already exist inside the commit
that produced the image; requiring it to would be an endless build/edit/commit/rebuild loop.
The authoritative identity is the one baked into `/etc/pmm-image.json`, which capture copies
into the environment manifest from inside the image.

```text
CONSISTENT       the committed record describes exactly this image
PENDING_COMMIT   no record yet for this image; expected on the pod, closed by the closure
                 commit. Closure may complete in this state
PENDING_REBUILD  the image was built from a different commit than the one being closed.
                 Closure FAILS
TAMPERED         the record claims THIS image but a different source commit, or the baked
                 digest is malformed. Closure FAILS
```

### Step 4 — close S00-B

The bootstrap ends with two verification steps that are deliberately different in kind:

```text
11  make bundle-verify   source artifacts are hash-bound; runtime evidence is verified by
                         re-derivation, so producing the environment manifest and rewriting
                         readiness during this run cannot invalidate the bundle that
                         authorised the run
12  make closure         writes stage_acceptance/S00/12_S00B_CLOSURE.json recording the
                         environment identity, the readiness state and the SHA256 of every
                         artifact this run produced; it writes nothing unless the identity
                         resolved
```

Then bring the runtime evidence back and make it the described state:

```bash
git add manifests/environments/<ENVIRONMENT_LOCK_SHA256>.json \
        artifacts/p0_pre/P0_PRE_READINESS.json \
        stage_acceptance/S00/12_S00B_CLOSURE.json
git commit -m "S00-B: hardware closure evidence"
make bundle && make bundle-verify
```

S00-B is closed when `ENVIRONMENT_LOCK_SHA256` is resolved and the closure record exists.

### Still unmeasured after S00-B

`make env-capture` records only what it observes. These require their own runs and stay
`TBD_REQUIRES_HARDWARE` until then [AUTH: 03 §8; 00 §0.2.4; 01 §30]:

```text
bf16_fp32_tolerance                     measured on the image over fixed sequences
nondeterminism_sources, run-to-run tolerance                              01 §30
batch size, throughput q, peak VRAM, projected GPU-hours   S10 benchmark, 00 §0.2.4
```

## Determinism notes


The CPU/dev lane pins every tool through `uv.lock`, so `make lint` and `make typecheck`
cannot flip against a byte-identical tree. The scientific lane's determinism is measured, not
assumed: run-to-run tolerance and nondeterministic kernel sources are recorded at S00-B
[AUTH: 01 §30].
