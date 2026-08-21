# Deferred findings — S00

[AUTH: 03 §10 — every DEFER carries a ticket ID, reason and target stage]

| Ticket | Finding | Severity | Reason | Target stage |
|---|---|---|---|---|
| S00-T01 | `S00-CBR-005` — `.gitignore` cites `01 §11`, whose repository tree does not name it; no authority mandates the file | MINOR / UNJUSTIFIED_SCOPE | Round 2 is scoped to the ten accepted findings; the defect is a citation on one zero-burden file and affects no correctness, leakage, provenance or cache property [AUTH: 03 §6] | S01 — correct the citation or remove the file when run, artifact and cache write paths make the ignore policy substantive |

No BLOCKER is deferred [AUTH: 03 §10].
