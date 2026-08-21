# Command surface. Only traceable targets [AUTH: 01 §33, §22, §21, §12; 02 §C6; plan §10].

PY  ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
RUFF  = $(PY) -m ruff
MYPY  = $(PY) -m mypy
PYTEST = $(PY) -m pytest
READINESS = artifacts/p0_pre/P0_PRE_READINESS.json

.PHONY: format lint typecheck unit integration synthetic backend-contract gpu-smoke \
        env-capture preflight

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
	 if [ $$s -eq 5 ]; then echo "synthetic: NOT_RUN(EMPTY_AT_S00) [AUTH: 02 §C6; 00 §34B.3]"; \
	 elif [ $$s -ne 0 ]; then exit $$s; fi

## never reports PASS while BACKEND_INTEGRATED=FALSE [AUTH: 02 §C6]
backend-contract:
	@if [ "$$($(PY) -c 'import json,sys;print(json.load(open("45608READINESS"))["BACKEND_INTEGRATED"])' 2>/dev/null)" != "True" ]; then \
	   echo "backend-contract: NOT_RUN(BACKEND_NOT_INTEGRATED) [AUTH: 02 §C6]"; \
	 else $(PYTEST) tests/backend_contract; fi

## real model/library/runtime compatibility on H100 [AUTH: 01 §21, §22]
gpu-smoke:
	@if ! command -v nvidia-smi >/dev/null 2>&1; then \
	   echo "gpu-smoke: NOT_RUN(NO_GPU) [AUTH: 01 §21; 03 §8]"; \
	 else $(PYTEST) tests/gpu_smoke; fi

## 01 §12 steps 2-9 on the real H100 image; every value TBD_REQUIRES_HARDWARE until then
env-capture:
	bash scripts/capture_environment.sh --out manifests/environments

## ordered §33 gate + readiness evaluation [AUTH: 01 §33, §22; 02 §C6; plan §10.1]
preflight:
	$(PY) scripts/preflight.py --root .
