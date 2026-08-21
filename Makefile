# Command surface. Only traceable targets [AUTH: 01 §33, §22, §21, §12; 02 §C6; plan §10].

PY  ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
RUFF  = $(PY) -m ruff
MYPY  = $(PY) -m mypy
PYTEST = $(PY) -m pytest
READINESS ?= artifacts/p0_pre/P0_PRE_READINESS.json

.PHONY: format lint typecheck unit integration synthetic backend-contract gpu-smoke \
        env-capture preflight bundle bundle-verify bundle-verify-source \
        closure review-worktree

## make format [CHECK=1] — apply, or verify without mutating [AUTH: 01 §33; plan D5]
format:
ifdef CHECK
	$(RUFF) format --check .
else
	$(RUFF) format .
endif

## [AUTH: 01 §33]
lint:
	$(RUFF) check .

## typed core modules [AUTH: 01 §33; plan D6]
typecheck:
	$(MYPY)

## [AUTH: 01 §22 Unit, §33]
unit:
	$(PYTEST) tests/unit

## [AUTH: 01 §22 Integration, §33]
integration:
	$(PYTEST) tests/integration

## one command for the one §22 synthetic/golden tier [AUTH: 01 §22; 00 §34B]
synthetic:
	@$(PYTEST) tests/synthetic tests/golden; s=$$?; \
	 if [ $$s -eq 5 ]; then \
	   echo "synthetic: NOT_RUN(EMPTY_AT_S00) [AUTH: 02 §C6; 00 §34B.3]"; \
	 elif [ $$s -ne 0 ]; then exit $$s; fi

## Three-way branch. A missing or unreadable readiness file is a hard failure, never a
## NOT_RUN: an unreadable gate must not look like an evaluated one [AUTH: 02 §C6].
backend-contract:
	@$(PY) scripts/preflight.py --backend-integrated "$(READINESS)"; s=$$?; \
	 if [ $$s -eq 0 ]; then \
	   $(PYTEST) tests/backend_contract; \
	 elif [ $$s -eq 1 ]; then \
	   echo "backend-contract: NOT_RUN(BACKEND_NOT_INTEGRATED) [AUTH: 02 §C6]"; \
	 else \
	   echo "backend-contract: readiness gate unreadable [AUTH: 02 §C6]" >&2; exit $$s; fi

## real model/library/runtime compatibility on H100 [AUTH: 01 §21, §22]
gpu-smoke:
	@if ! command -v nvidia-smi >/dev/null 2>&1; then \
	   echo "gpu-smoke: NOT_RUN(NO_GPU) [AUTH: 01 §21; 03 §8]"; \
	 else \
	   out=$$(mktemp); $(PYTEST) tests/gpu_smoke > $$out 2>&1; s=$$?; cat $$out; \
	   if [ $$s -eq 5 ]; then \
	     echo "gpu-smoke: FAIL zero tests collected; the lane cannot report a state" >&2; \
	     rm -f $$out; exit 1; \
	   elif [ $$s -ne 0 ]; then rm -f $$out; exit $$s; \
	   else \
	     $(PY) scripts/preflight.py --record-lane gpu_smoke --outcome PASS \
	       --detail "$$(tail -1 $$out)"; \
	     rm -f $$out; \
	   fi; fi

## 01 §12 steps 2-9 on the real H100 image; every value TBD_REQUIRES_HARDWARE until then
env-capture:
	bash scripts/capture_environment.sh --out manifests/environments

## ordered §33 gate + readiness evaluation [AUTH: 01 §33, §22; 02 §C6; plan §10.1]
preflight:
	$(PY) scripts/preflight.py --root .

## regenerate the acceptance bundle from HEAD [AUTH: 01 §26, §45]
bundle:
	$(PY) scripts/build_bundle.py --root . --stage S00

## record the runtime evidence produced by an actual S00-B hardware run
closure:
	$(PY) scripts/build_bundle.py --root . --stage S00 --closure

## immutable half only: described commit, source drift, source artifact hashes.
## Safe inside the ordered gate; the runtime half needs preflight step 8 to have run.
bundle-verify-source:
	$(PY) scripts/build_bundle.py --root . --stage S00 --verify-source

## FINAL closure gate: immutable source AND runtime evidence. Runs after capture, the GPU
## lane and the complete preflight, never inside the ordered gate.
bundle-verify:
	$(PY) scripts/build_bundle.py --root . --stage S00 --verify

## separate pinned read-only worktree for a blind reviewer [AUTH: 03 §9]
review-worktree:
	bash scripts/make_review_worktree.sh $(COMMIT) $(DEST) $(STAGE)
