# Pre-registered analysis plan — second domain `fab30` (re-entrant wafer fab)

**Status: written on 4 November 2026, BEFORE any `fab30` LLM call.** Commit this file to the repository **before**
starting the run (`git log` time and the SHA-256 in `PREREGISTRATION_fab30.sha256` are the evidence).

## Why
The original submission's fabrication results were produced with a simulator that is lost, with the parsing defect,
and without certification. `fab30.py` implements the 30 original fab scenarios (FD1–FD10, FS1–FS10, FSem1–FSem10) as
mechanisms of a re-entrant fab model (lot routes over 8 machine groups, Poisson releases, priority classes, due dates,
q-time limits, maintenance windows). Its parameters were fixed by rules stated in the module before any LLM run
(arrival rate: litho utilisation closest to 0.70; q-time: 80th percentile of the litho wait under FIFO;
due-date factors and SLA documented in `fab30.py`). Checks before the run: `fab30_sanity.py` (all schedules feasible;
deterministic; per-scenario table in `results_r1/fab30_sanity.json`).

## Design
- 30 scenarios × seeds 0–7 × incumbents FIFO, SPT, CR = **720 trials**; `deepseek-chat`, temperature 0.7; Q = 8 calls,
  patience 8, 5 variants per call; paired superiority gate inside the system (`LAOSA_GATE=paired_sup`).
- Fab static library (8 rules): FIFO, SPT, EDD, CR, LWKR, MWKR, WSJF, PRIORITY. Fallback = best on the 10
  generation instances. Certification: 2,000 instances (seeds 500+s); held-out: 2,000 instances (seeds 900+s).
- Run: `python3 run_r1_llm.py --mode fab30`.

## Endpoints (the main-study claims, tested in a second domain; α = 0.05; clusters = scenario × seed, 240)
| ID | Endpoint | **Replicated** if |
|---|---|---|
| F1 | Held-out violations of the rule deployed online (paired superiority) | ≤ 3 of 720 |
| F2 | Ungoverned violations (generated rule always deployed) | cluster-bootstrap 95% CI lower bound > 0, i.e. unreviewed deployment is harmful in a non-zero share of episodes |
| F3 | Mean held-out gain of the deployed rule over the fallback | cluster-bootstrap 95% CI lower bound > 0 |
| F4 | Semantic vs. distributional+structural ungoverned violation rate | reported with CI; the WGS pattern (semantic higher) counts as replicated only if the CI of the difference excludes 0 |
| F5 | LLM vs. linear (1+1)-ES with equal evaluation budget (`nonllm_baseline.py`, fab features), Beats-Best under superiority | exact McNemar p < 0.05 in favour of the LLM; otherwise "no detectable advantage" |
| F6 | Hypothesis 1 (effort ordering), superiority | supported only if Kruskal–Wallis on scenario-level mean N has p < 0.05 and medians ordered dist ≤ struct ≤ sem |

Descriptive: non-inferiority results, per-scenario table, Beats-Best, budget curve, tokens and cost.
Scenarios whose change leaves every library rule's cost unchanged (expected from `fab30_sanity.json`: FD5, FD7, FS6)
are kept and reported; they test whether governance withholds deployment when nothing can be gained.

## Analysis commands (fixed)
```
python3 r1_offline_eval.py --pkg $P --mode fab30 --trials fab30_trials.jsonl --outdir full
python3 paired_gate.py     --pkg $P --mode fab30 --trials fab30_trials.jsonl --records full/fab30_offline_records.json
python3 h1_paired.py       --pkg $P --mode fab30 --trials fab30_trials.jsonl --records full/fab30_offline_records.json
python3 nonllm_baseline.py --pkg $P --mode fab30 --trials fab30_trials.jsonl --records full/fab30_offline_records.json --outdir full_fab
python3 fab_analysis.py    --dir full --trials fab30_trials.jsonl
```
