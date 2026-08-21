#!/usr/bin/env bash
# Create a SEPARATE PINNED READ-ONLY worktree for a blind reviewer [AUTH: 03 §9; 01 §25].
#
# The primary worktree is never read-only and must not be presented as such. This script
# gives a reviewer their own checkout at a pinned commit, with every production path
# non-writable and writes confined to reviews/<stage>/scratch/.
#
#   bash scripts/make_review_worktree.sh <commit> <dest> [stage]
#   bash scripts/make_review_worktree.sh --remove <dest>
set -euo pipefail

if [ "${1:-}" = "--remove" ]; then
  DEST="${2:?usage: --remove <dest>}"
  ROOT="$(git rev-parse --show-toplevel)"
  chmod -R u+w "$DEST" 2>/dev/null || true
  git -C "$ROOT" worktree remove --force "$DEST"
  echo "removed review worktree $DEST"
  exit 0
fi

COMMIT="${1:?usage: make_review_worktree.sh <commit> <dest> [stage]}"
DEST="${2:?usage: make_review_worktree.sh <commit> <dest> [stage]}"
STAGE="${3:-S00}"
ROOT="$(git rev-parse --show-toplevel)"

PINNED="$(git -C "$ROOT" rev-parse "${COMMIT}^{commit}")"
git -C "$ROOT" worktree add --detach "$DEST" "$PINNED" >/dev/null
DEST="$(cd "$DEST" && pwd)"

SCRATCH="$DEST/reviews/$STAGE/scratch"
mkdir -p "$SCRATCH"

# Read-only everywhere, then re-open exactly one directory. The worktree's index lives in the
# main repository's .git/worktrees/, so git commands still work from here.
chmod -R a-w "$DEST"
[ -f "$DEST/.git" ] && chmod u+w "$DEST/.git"
chmod -R u+w "$SCRATCH"

cat <<SUMMARY
review worktree ready
  path            $DEST
  pinned commit   $PINNED
  writable        reviews/$STAGE/scratch/
  everything else read-only [AUTH: 03 §9]
remove with: bash scripts/make_review_worktree.sh --remove "$DEST"
SUMMARY
