"""
run_causal_v2.py
==================
Second version of the causal (energy-operator hint) experiment,
fixing two problems found by inspecting the first run's results:

1. FLOOR EFFECT: llm_calls was exactly 4 for all 80/80 trials, and
   improvement_pct/beats_best were byte-identical between control and
   treatment for all 40 matched (seed, baseline) pairs. Root cause:
   the search NEVER found an improving candidate in either condition
   (every round printed "no improvement"), so the reported
   improvement_pct is entirely explained by the deterministic
   degraded->best_fixed jump computed from simulation alone, with zero
   contribution from anything the LLM generated. A binary
   "beats_best" outcome has no statistical power here because both
   arms are stuck at the floor (0%).

   FIX: instrument every candidate evaluation (not just the final
   winner) by patching LLMAlgorithmAdapter._evaluate_code to log
   (code, score) for every candidate tried, in every round, for every
   trial. This lets us measure "how close did the closest candidate
   get to best_fixed" -- a continuous, non-saturated outcome -- even
   though no candidate ever fully won.

2. PROMPT CONTRADICTION: _build_strong_prompt's "Available task
   fields" section is a hardcoded list that never mentions an energy-
   related field/operator, regardless of what the (separately
   injected) scene description says. The LLM is told about an energy
   formula in one part of the prompt and given a field list that
   doesn't support it in another -- likely neutralizing the hint
   entirely.

   FIX: patch _build_strong_prompt to append the energy field to the
   "Available task fields" section specifically when the treatment
   condition is active, so the hint is actually usable, not just
   mentioned.

Usage:
    python3 run_causal_v2.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

key_found = any(os.environ.get(k) for k in
                ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ZHIPU_API_KEY"])
if not key_found:
    print("WARNING: no recognized API key found in environment.")
    raise SystemExit(1)

sys.path.insert(0, ".")
import numpy as np
import run_objective_spectrum as spec
import run_poms_experiments as poms
from laosa import LLMAlgorithmAdapter
from fix_objective_spectrum_evaluate import make_prob_fixed

# ----------------------------------------------------------------
# Instrumentation: log every (code, score) evaluated, per trial
# ----------------------------------------------------------------
_CANDIDATE_LOG = []  # reset per trial
_ORIGINAL_EVALUATE_CODE = LLMAlgorithmAdapter._evaluate_code

def _instrumented_evaluate_code(self, code, prob):
    score = _ORIGINAL_EVALUATE_CODE(self, code, prob)
    if score is not None:
        _CANDIDATE_LOG.append(score)
    return score

LLMAlgorithmAdapter._evaluate_code = _instrumented_evaluate_code

# ----------------------------------------------------------------
# Instrumentation: fix the "Available task fields" contradiction
# ----------------------------------------------------------------
_INJECT_ENERGY_FIELDS = {"active": False}
_ORIGINAL_BUILD_PROMPT = poms._build_strong_prompt

ENERGY_FIELD_TEXT = (
    "\n- task.task_type-based energy rate (ENERGY_RATE): "
    "'alignment'=3.5, 'variant_calling'=5.0, 'sv_calling'=2.0 "
    "(you may compute energy(task) = task.estimated_runtime * ENERGY_RATE[task.task_type] "
    "directly in your priority function using these constants)"
)

def _patched_build_prompt(*args, **kwargs):
    prompt = _ORIGINAL_BUILD_PROMPT(*args, **kwargs)
    if _INJECT_ENERGY_FIELDS["active"]:
        # insert the energy field right after the existing field list,
        # so it's actually usable, not just mentioned in the free-text
        # scene description (which the original bug left disconnected
        # from what the LLM is told it's allowed to reference).
        marker = "- task.task_type (str): 'alignment' (~10h, 8 CPU) | 'variant_calling' (~5h, 4 CPU) | 'sv_calling' (~20h, 32 CPU)"
        if marker in prompt:
            prompt = prompt.replace(marker, marker + ENERGY_FIELD_TEXT)
    return prompt

poms._build_strong_prompt = _patched_build_prompt


def run_one_trial(mode, sc_id, seed, bl_name, bl_code, inject):
    global _CANDIDATE_LOG
    _CANDIDATE_LOG = []
    _INJECT_ENERGY_FIELDS["active"] = inject

    np.random.seed(seed)
    prob_before = make_prob_fixed("baseline", spec.N_INSTANCES, spec.N_TASKS, spec, force_mode=mode)
    prob_after = make_prob_fixed(sc_id, spec.N_INSTANCES, spec.N_TASKS, spec, force_mode=mode)

    bl_before = 1.0 - prob_before.evaluate(bl_code)
    bl_after = 1.0 - prob_after.evaluate(bl_code)

    all_baseline_scores = {}
    for bn, bc in spec.get_baselines().items():
        v = prob_after.evaluate(bc)
        all_baseline_scores[bn] = (1.0 - v) if v is not None else None
    best_static = min((v for v in all_baseline_scores.values() if v is not None), default=bl_after)

    deg_pct = (bl_after - bl_before) / max(abs(bl_before), 1e-9) * 100

    # CRITICAL FIX (found via dry-run testing): the scene's own base
    # description ALREADY spells out the full energy formula and exact
    # ENERGY_RATE values for BOTH conditions -- meaning the original
    # experiment's "control" was never actually withholding this
    # information. For a genuine causal contrast, redact the formula
    # for control; only treatment gets the full formula (in BOTH the
    # description AND the field-list patch, kept consistent).
    if inject:
        desc = spec._inst_mod.SCENES[sc_id]["description"]  # full formula, as-is
    else:
        desc = ("Tier-1 distributional shift: GPU-intensive alignment fraction 20%->70%. "
                "Objective now includes an energy-consumption penalty that depends on "
                "task type, in addition to makespan. Static baselines ignore energy "
                "entirely.")  # same scenario, no formula/rate table disclosed

    adapted_code, adapted_ms, llm_calls = poms._run_enhanced_evo_search(
        prob_after=prob_after, current_code=bl_code,
        degraded_ms=bl_after, baseline_ms=bl_before,
        degradation_pct=deg_pct,
        scene_change={"id": sc_id, "after": sc_id, "change_type": "distributional",
                      "description": desc},
        algo_name=bl_name,
    )

    beats = bool(adapted_ms is not None and adapted_ms < best_static)

    # FIX: _CANDIDATE_LOG captures EVERY _evaluate_code call, including
    # the baseline-scan loop inside _run_enhanced_evo_search (which
    # evaluates all len(baselines) static codes BEFORE the LLM round
    # loop even starts). Since best_static is ALSO just the min over
    # those same baseline evaluations, min(_CANDIDATE_LOG) trivially
    # equals best_static regardless of what the LLM did -- this
    # produced the meaningless, guaranteed-0.00% gap in the first v2
    # run. Skip the first len(baselines) logged entries (the baseline
    # scan) so the gap metric reflects ONLY LLM-generated candidates.
    n_baselines = len(spec.get_baselines())
    llm_only_scores = _CANDIDATE_LOG[n_baselines:]
    best_candidate_score = min(llm_only_scores) if llm_only_scores else adapted_ms
    gap_to_best_static = max(0.0, best_candidate_score - best_static)
    gap_pct = (gap_to_best_static / max(abs(best_static), 1e-9)) * 100

    return {
        "beats_best": beats,
        "llm_calls": llm_calls,
        "n_candidates_evaluated": len(_CANDIDATE_LOG),
        "n_llm_candidates_only": len(llm_only_scores),
        "best_candidate_score": float(best_candidate_score),
        "best_static": float(best_static),
        "gap_to_best_static_pct": float(gap_pct),
    }


def main():
    baselines = spec.get_baselines()
    conditions = [("control", False), ("treatment", True)]
    results = []

    print(f"Causal v2: {len(conditions)*len(spec.SEEDS)*len(baselines)} trials\n")
    for cond_name, inject in conditions:
        print(f"-- {cond_name} ({'with fixed energy fields' if inject else 'without'}) --")
        for seed in spec.SEEDS:
            for bl_name, bl_code in baselines.items():
                r = run_one_trial("energy", spec.SCENE_ENERGY_ID, seed, bl_name, bl_code, inject)
                r.update({"condition": cond_name, "seed": seed, "baseline": bl_name})
                results.append(r)
                print(f"  {cond_name}/seed{seed}/{bl_name}: BB={r['beats_best']}, "
                      f"gap_to_best_static={r['gap_to_best_static_pct']:.2f}%, "
                      f"n_candidates={r['n_candidates_evaluated']}, calls={r['llm_calls']}")

    import json
    with open("results/causal_v2.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- analysis: compare the CONTINUOUS gap metric, not just BB% ----
    from collections import defaultdict
    import statistics as st
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r["gap_to_best_static_pct"])

    print("\n-- Gap-to-best-static (%) by condition (lower = closer to winning) --")
    for cond_name, _ in conditions:
        gaps = by_cond[cond_name]
        print(f"  {cond_name:10s}: mean={st.mean(gaps):.2f}%  "
              f"median={st.median(gaps):.2f}%  n={len(gaps)}")

    try:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(by_cond["control"], by_cond["treatment"], alternative="two-sided")
        print(f"\n  Mann-Whitney U test (control vs treatment gap): U={u:.1f}, p={p:.4f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
