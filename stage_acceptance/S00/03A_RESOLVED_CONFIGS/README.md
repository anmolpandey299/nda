# Resolved configuration — 01 §26 item 8

S00 resolves **no scientific configuration**: `configs/**` is structure only and S00 does not
touch config content [AUTH: plan §15; 01 §39 S00]. The configuration S00 does resolve is the
environment, whose resolved state is the two files hashed in `resolved_environment.json`.

Material scientific constants (target FPR, Min-K fraction, alpha, k, DARE p, SVD rank,
bootstrap replicates, seeds) will resolve from `configs/**` from S01 onward; invariant I14 and
test A17 already reject any of them appearing as a source literal under `src/**`
[AUTH: 01 §17].
