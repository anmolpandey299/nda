CODEX_BLIND_REVIEW_VERDICT = FAIL

  ID: S00-CBR-001
  severity: BLOCKER
  authority citation: 03_REVIEW_GOVERNANCE_LOCK.md §9; 01_EXECUTION_STACK_LOCK_v2.md §25
  location: submitted review root; STAGE_PLAN.md §8.1, lines 583–593; A9–A10
  concrete counterexample: git rev-parse HEAD fails because the supplied directory is not a Git worktree, while test -w STAGE_PLAN.md and the same check on the scientific
  specification both succeed. The reviewed revision cannot be identified, and production inputs can be modified without any hook firing.
  required correction: provide a verifiable pinned Git worktree and commit SHA; make all paths except reviews/S00/scratch/ read-only to the reviewer; add a negative test
  proving writes to production paths fail; rerun the blind review.

  ID: S00-CBR-002
  severity: MAJOR
  authority citation: 01_EXECUTION_STACK_LOCK_v2.md §15, §16, §27, §35
  location: STAGE_PLAN.md §7.1, lines 545–558; I11, lines 808–813; A12, line 834
  concrete counterexample: checkout commit C, modify src/scoring/engine.py without committing, then launch an evidentiary run. The plan derives RUN_ID and code_commit from C;
  git dirty status may be recorded but is neither rejected nor made non-evidentiary. The modified scorer can therefore produce results attributed to clean commit C,
  potentially colliding with the clean run identity.
  required correction: add a pre-manifest clean/frozen-code gate covering staged, unstaged, and untracked production files; refuse evidentiary execution when dirty; test each
  dirty-state case.

  ID: S00-CBR-003
  severity: MAJOR — PROMPT_ONLY_REQUIREMENT
  authority citation: 01_EXECUTION_STACK_LOCK_v2.md §12, §15, §46; 02_PRE_EXECUTION_IMPLEMENTATION_CORRECTIONS.md C6, C7; 03_REVIEW_GOVERNANCE_LOCK.md §14
  location: STAGE_PLAN.md lines 301, 425–426, 560–564, 839–844, and 990
  concrete counterexample: with no H100, record F1 and use line 990’s “S00-B closed, or F1 recorded” branch to accept S00 without the required H100 smoke, frozen lock, or
  Docker digest. Separately, two images sharing uv.lock but having different Docker digests produce the same RUN_ID; a changed PyTorch/Transformers environment can also reuse
  the shared scoring cache because the environment identity is absent from its key.
  required correction: remove F1 as a full-acceptance alternative; require S00-B evidence for acceptance; define the environment-lock hash over the complete frozen
  environment including image digest; include that identity in both RUN_ID and scoring-cache keys, with invalidation tests.

  ID: S00-CBR-004
  severity: MAJOR
  authority citation: 00_MEASUREMENT_SPEC_v1.9_FINAL_CLOSED.md §34A.1–§34A.2; 02_PRE_EXECUTION_IMPLEMENTATION_CORRECTIONS.md C1, C2
  location: STAGE_PLAN.md I1, line 798; §9.3, lines 642–657; A7, line 829
  concrete counterexample: add src/analysis/privacy_metrics.py implementing a second fold-selection path and the prohibited individual-record ROC plus “max TPR at repeated
  FPR,” then route headline analysis through it. Because its basename is not engine, crossfit, cache, or pooled, I1 and A7 pass. Nothing in the plan asserts that headline
  calculations remain M_PRIMARY = common-FPR TPR@1%FPR or use the tie-safe estimator.
  required correction: reserve one canonical cross-fit and fixed-FPR estimator API; enforce dependency boundaries rather than basename uniqueness; test every arm/view/
  headline consumer through that API; assert exact 1% common-FPR semantics and tied-score behavior.

  ID: S00-CBR-005
  severity: MINOR — UNJUSTIFIED_SCOPE
  authority citation: 03_REVIEW_GOVERNANCE_LOCK.md §4, §7
  location: STAGE_PLAN.md lines 158 and 860
  concrete counterexample: .gitignore cites 01 §11, but that authority’s repository tree does not contain .gitignore; searching all four authority inputs finds no mandate for
  it. The plan’s claim that every component is authority-traceable is therefore false.
  required correction: remove .gitignore from S00 or cite a specific binding authority that requires it.