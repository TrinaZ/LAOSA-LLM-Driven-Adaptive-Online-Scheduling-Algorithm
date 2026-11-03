# Pre-registered analysis plan — confirmatory replication `confirm30`

**Status: written on 3 November 2026, BEFORE any `confirm30` LLM call.**
Commit this file to the repository before running; the commit time and the SHA-256 of this file (recorded in
`PREREGISTRATION_confirm30.sha256`) are the evidence. Any deviation from this plan will be reported as a deviation.

## Why
In the revision, the paired certification test was adopted after the originally implemented (unpaired) test had
certified 0 of 960 rules, and all analyses of the main study were run on the same data. This replication tests the
main claims on new data, with the analysis fixed in advance and with the paired test running *inside the system*
(`laosa.py` patched by `patch_paired_gate.py`, `LAOSA_GATE=paired_sup`), so that deployment is decided online,
not replayed.

## Design (identical to the main study except the seeds)
- 30 scenarios (`scenarios30.py`), incumbents SPT/LPT/EDF/WSJF, **seeds 8–15**, 960 trials.
- `deepseek-chat`, temperature 0.7, Q = 8 calls, patience 8, 5 variants per call (raw-output patch).
- Generation instances seed s; certification 2,000 instances seed 500+s; held-out 2,000 instances seed 900+s.
- Run: `python3 run_r1_llm.py --mode confirm30` (driver refuses to run without the paired-gate patch).

## Endpoints and decision rules (α = 0.05; 95% intervals; clusters = scenario × seed, 240)
| ID | Endpoint | Claim is **confirmed** if |
|---|---|---|
| P1 | Held-out violations of the rule deployed online (paired superiority) | ≤ 3 of 960 (exact upper bound < 1%) |
| P2 | Ungoverned violations, semantic vs. distributional+structural | cluster-bootstrap 95% CI of the rate difference excludes 0 (semantic higher) |
| P3 | Mean held-out gain of the deployed rule over the fallback | cluster-bootstrap 95% CI lower bound > 0 |
| P4 | Beats-Best of the deployed rule (superiority) | reported with CIs; replication if the 95% CI overlaps the main study's 28.3–36.7% |
| P5 | LLM search vs. non-LLM linear (1+1)-ES with equal evaluation budget (`nonllm_baseline.py`): Beats-Best under superiority, paired | exact McNemar p < 0.05 in favour of the LLM; otherwise reported as "no detectable advantage" |
| P6 | Hypothesis 1 (effort ordering), `h1_paired.py`, superiority | supported only if Kruskal–Wallis on scenario-level mean N has p < 0.05 **and** medians are ordered dist ≤ struct ≤ sem; expected: not supported |
| P7 | Agreement of online deployment decisions with the offline replay (`paired_gate.py`) | ≥ 99% agreement; disagreements listed |

Secondary (descriptive only): non-inferiority results; per-scenario table; tokens and cost.
No other endpoint will be described as confirmatory.

## Analysis commands (fixed)
```
python3 r1_offline_eval.py --pkg $P --mode confirm30 --trials confirm30_trials.jsonl --outdir full
python3 paired_gate.py   --pkg $P --mode confirm30 --trials confirm30_trials.jsonl --records full/confirm30_offline_records.json
python3 h1_paired.py     --pkg $P --mode confirm30 --trials confirm30_trials.jsonl --records full/confirm30_offline_records.json
python3 nonllm_baseline.py --pkg $P --trials confirm30_trials.jsonl --records full/confirm30_offline_records.json --outdir full_confirm
python3 confirm_analysis.py --dir full --trials confirm30_trials.jsonl
```
