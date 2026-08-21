# SCORER_EXCEPTION register

Duplicated scoring logic is prohibited unless recorded here [AUTH: 00 §34A.5], with:

- why the shared interface cannot represent the case;
- the code path;
- tests establishing numerical equivalence where equivalence is expected;
- the affected claims.

No `SCORER_EXCEPTION` may be introduced silently after viewing the affected result.

## Standing exclusions

These are permanent prohibitions, not exceptions. They can never be granted.

```text
individual-record cumulative ROC + "max TPR at repeated FPR"            PROHIBITED
```

Reason: with equal scalar scores in both classes no threshold separates members of a tie
group, so individual-record ordering manufactures unattainable ROC points. The production
estimator must build ROC points at unique score thresholds, add all observations sharing an
exact score simultaneously, use `drop_intermediate = false`, and interpolate linearly between
valid neighbouring points — exposed as one tested function used by every view and arm
[AUTH: 02 §C2]. The P0 scaffold is a reference fixture and may not be copied merely because a
synthetic fixture passed [AUTH: 02 §C8].

```text
M_PRIMARY(V) = TPR_OOF/eval( V | FPR = 0.01 )      common-FPR TPR@1%FPR      [AUTH: 02 §C1]
```

## Granted exceptions

| ID | Date (UTC) | Stage | Code path | Why the shared interface cannot represent it | Equivalence tests | Affected claims |
|---|---|---|---|---|---|---|
| _(none)_ | | | | | | |
