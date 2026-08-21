#!/usr/bin/env bash
# Install the convenience hook guard [AUTH: 01 §39 S00 "git hooks"].
#
# Hooks are the WEAKEST enforcement layer: core.hooksPath is per-clone local config and
# `--no-verify` bypasses them. The authority mechanisms are protected branches/tags
# [AUTH: 01 §27] and the evidentiary clean-state gate I15 [AUTH: 01 §35(3), §36].
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
git -C "$ROOT" config core.hooksPath .githooks
chmod +x "$ROOT/.githooks/pre-commit" "$ROOT/.githooks/pre-push"
echo "core.hooksPath = $(git -C "$ROOT" config --get core.hooksPath)"
