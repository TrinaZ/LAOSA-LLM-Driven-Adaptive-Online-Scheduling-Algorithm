"""
fix_objective_spectrum_evaluate.py
=====================================
Fixes two compounding bugs found in run_objective_spectrum.py that
fully explain the anomalous 0% Beats-Best results for Weighted
Tardiness, Fairness Throughput, and Energy Makespan in Table
tab:spectrum:

BUG A (spurious early exit): _run_enhanced_evo_search has a hardcoded
  check `if best_fixed_ms <= 1e-6: return immediately, 0 LLM calls`.
  This is correct for makespan-like raw scores (0 = perfect). But
  make_prob()'s custom evaluate() for tardiness/fairness/energy
  returns a FITNESS value centered around 0 by construction
  (fitness = (baseline_score - avg) / baseline_score, ~0 whenever a
  candidate performs similarly to the reference baseline -- NOT
  because it's optimal). Any baseline landing near this artificial
  zero triggers a spurious "already optimal" skip.

BUG B (sign inversion): _run_enhanced_evo_search's acceptance
  criterion is `if cand_ms < best_cand_ms: accept` (lower = better,
  matching the makespan convention used everywhere else). But the
  fitness formula above INCREASES when the real objective improves,
  for all three custom modes (tardiness, fairness, energy) -- so the
  search loop was systematically rewarding candidates that make the
  real objective WORSE, and rejecting genuine improvements.

FIX: stop routing tardiness/fairness/energy through the
"fitness-relative-to-baseline" abstraction entirely for the purposes
of driving the search. Return RAW scores in a consistent
"lower is better" convention instead (negate fairness's raw score,
since fairness is naturally "higher is better"; leave tardiness/
energy as-is, since they're already "lower is better"). This makes
prob.evaluate() behave exactly like the makespan case everywhere else
in the codebase, which is what _run_enhanced_evo_search assumes.

Apply this fix by replacing the `_make_evaluate` function and the
`make_prob` baseline_score/patch logic in run_objective_spectrum.py
with the versions below, then re-run:
    python3 run_objective_spectrum.py --mode run
for the three affected scenes (delete their existing entries from
results/objective_spectrum_raw.json first, or run against a fresh
output file, so old buggy records don't get silently kept).
"""
import sys
import types
import warnings
import numpy as np


def make_evaluate_fixed(objective_mode, scene_config, score_fns):
    """
    Corrected replacement for run_objective_spectrum._make_evaluate.
    Returns RAW scores in a consistent lower-is-better convention --
    no baseline-relative fitness normalization, no sign ambiguity.

    score_fns: dict with keys 'tardiness','fairness','energy' mapping
    to the corresponding _score_* functions imported from
    run_objective_spectrum.py (pass them in to avoid a circular import).
    """
    if objective_mode == "makespan":
        return None  # caller should keep original_evaluate, unchanged

    fairness_weight = scene_config.get("fairness_weight", 0.3)
    energy_rates = scene_config.get("energy_rates", None)
    energy_weight = scene_config.get("energy_weight", 0.2)

    def evaluate_new(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module = types.ModuleType("hm_fixed")
                exec(code_string, module.__dict__)
                sys.modules[module.__name__] = module
                pf = getattr(module, "priority", None)
                if pf is None:
                    return None

                raw_scores = []
                for inst in self.instances:
                    tasks = inst["tasks"]
                    res = inst["total_resources"]
                    if objective_mode == "tardiness":
                        s = score_fns["tardiness"](tasks, res, pf)
                        # already lower-is-better: keep as-is
                    elif objective_mode == "fairness":
                        s = score_fns["fairness"](tasks, res, pf, fairness_weight)
                        # fairness is higher-is-better in its native form
                        # -- negate so lower-is-better holds, matching
                        # every other objective mode's convention.
                        # FIX (found during live testing): raw fairness
                        # scores here are small (~0.10-0.20), so a plain
                        # negation lands every value in [-0.2, -0.1] --
                        # ALL NEGATIVE. _run_enhanced_evo_search's
                        # near-optimal early-exit check
                        # (`if best_fixed_ms <= 1e-6: skip`) then fires
                        # for EVERY trial, since any negative number is
                        # trivially <= a small positive threshold. This
                        # caused calls=0 for 100% of Fairness trials in
                        # the first live run, independent of the
                        # _evaluate_code un-normalization bug fixed
                        # above. Add a constant offset large enough to
                        # push all realistic values solidly positive --
                        # a constant shift preserves relative ordering
                        # exactly, so lower-is-better comparisons are
                        # unaffected, but the near-zero early-exit check
                        # no longer fires spuriously.
                        FAIRNESS_OFFSET = 10.0
                        s = FAIRNESS_OFFSET - s
                    elif objective_mode == "energy":
                        s = score_fns["energy"](tasks, res, pf, energy_rates, energy_weight)
                        # already lower-is-better: keep as-is
                    else:
                        return None
                    raw_scores.append(s)

                # NOTE: no baseline-relative normalization here in the
                # semantic sense -- but see the (1 - raw) encoding trick
                # below, which is REQUIRED for compatibility with
                # laosa.py::LLMAlgorithmAdapter._evaluate_code, which
                # unconditionally applies:
                #     return prob.baseline_makespan * (1 - fitness)
                # to whatever evaluate() returns. _run_enhanced_evo_search
                # (and therefore the ENTIRE search loop, not just the
                # initial baseline scan) calls prob.evaluate() only
                # through _evaluate_code, never directly -- so if we
                # returned desired_score directly here, _evaluate_code
                # would silently mangle it via that hardcoded transform.
                # By setting baseline_makespan=1.0 (below, in
                # make_prob_fixed) and returning `1 - desired_score`
                # here, _evaluate_code's transform becomes
                # `1.0 * (1 - (1 - desired_score)) = desired_score`,
                # exactly recovering the intended value regardless of
                # scale. This was NOT caught by earlier offline testing
                # because that testing called prob.evaluate() directly,
                # bypassing _evaluate_code's transform entirely.
                desired_score = float(np.mean(raw_scores))
                return 1.0 - desired_score

        except Exception:
            return None

    return evaluate_new


def make_prob_fixed(scene_id, n_instances, n_tasks, spec_module, force_mode=None):
    """
    Corrected replacement for run_objective_spectrum.make_prob.
    Uses make_evaluate_fixed instead of the original _make_evaluate,
    and does NOT compute/store a baseline_score at all -- it's no
    longer needed since we return raw scores directly.

    force_mode: explicitly specify which scoring function to use
    (tardiness/fairness/energy/makespan), instead of inferring it from
    scene_id. Needed for prob_before, which is built from the generic
    "baseline" scene (instances BEFORE the distributional shift) but
    must still be scored with the SAME objective function as
    prob_after, so both sides of the degradation-percentage
    calculation are in identical units. Inferring mode from scene_id
    alone (the original bug) silently falls back to "makespan" for any
    scene_id that doesn't literally match one of the three registered
    custom scene IDs -- which is exactly what happened to the original
    "baseline" prob_before.
    """
    from problems.genomic_scheduling.prob import GenomicScheduling

    prob = GenomicScheduling(scene_id, n_instances, n_tasks)

    if force_mode is not None:
        mode = force_mode
    elif scene_id == spec_module.SCENE_TARDINESS_ID:
        mode = "tardiness"
    elif scene_id == spec_module.SCENE_FAIRNESS_ID:
        mode = "fairness"
    elif scene_id == spec_module.SCENE_ENERGY_ID:
        mode = "energy"
    elif scene_id in getattr(spec_module._prob_mod, "WEIGHTED_COMPLETION_SCENES", ()):
        mode = "weighted_ft"
    else:
        mode = "makespan"

    if mode in ("makespan", "weighted_ft"):
        return prob  # unchanged -- these were never buggy

    # scene_config should reflect the AFTER scene's parameters (e.g.
    # fairness_weight, energy_rates) even when scoring the "before"
    # instances, since the objective definition itself doesn't change
    # across the shift -- only the task distribution does.
    ref_scene_id = {
        "tardiness": spec_module.SCENE_TARDINESS_ID,
        "fairness": spec_module.SCENE_FAIRNESS_ID,
        "energy": spec_module.SCENE_ENERGY_ID,
    }[mode]
    scene_config = spec_module._inst_mod.SCENES.get(ref_scene_id, {})
    score_fns = {
        "tardiness": spec_module._score_tardiness,
        "fairness": spec_module._score_fairness,
        "energy": spec_module._score_energy,
    }
    new_eval = make_evaluate_fixed(mode, scene_config, score_fns)
    prob.evaluate = types.MethodType(new_eval, prob)
    prob._objective_mode = mode
    # REQUIRED for _evaluate_code compatibility -- see the comment in
    # make_evaluate_fixed's evaluate_new(). Do not remove: without this,
    # _evaluate_code's `baseline_makespan * (1 - fitness)` transform
    # falls back to GenomicScheduling's default (unrelated, raw-
    # makespan-scale) baseline_makespan value, silently corrupting
    # every score computed via _evaluate_code -- which is how
    # _run_enhanced_evo_search evaluates EVERYTHING internally.
    prob.baseline_makespan = 1.0
    return prob


def run_spectrum_fixed(spec_module, scenes_to_fix=None):
    """
    Corrected replacement for run_objective_spectrum.run_spectrum,
    for the three affected scenes only (tardiness/fairness/energy).
    Both bl_before and bl_after are now computed through the SAME
    make_prob_fixed patched evaluate(), on their respective instance
    sets, using the SAME priority-function code -- eliminating the
    unit mismatch that previously mixed a raw makespan (bl_before)
    with a normalized fitness (bl_after) in the same degradation_pct
    formula.

    scenes_to_fix: subset of spec_module.NEW_SCENES to re-run (default:
    all three). Writes to a SEPARATE output file so old buggy records
    are not silently overwritten -- compare before/after explicitly.
    """
    from run_poms_experiments import _run_enhanced_evo_search

    baselines = spec_module.get_baselines()
    scenes = scenes_to_fix or spec_module.NEW_SCENES
    output_file = "results/objective_spectrum_FIXED.json"
    results = spec_module.load_results(output_file)
    done = {r["run_id"] for r in results}

    total = len(scenes) * len(spec_module.SEEDS) * len(baselines)
    print(f"FIXED re-run: {total} trials ({len(scenes)} objectives x "
          f"{len(spec_module.SEEDS)} seeds x {len(baselines)} baselines)\n")

    for sc in scenes:
        print(f"-- {sc['name']} (alpha_proxy={sc['alpha_proxy']}) --")
        for seed in spec_module.SEEDS:
            np.random.seed(seed)
            for bl_name, bl_code in baselines.items():
                run_id = f"FIXED_{sc['id']}_seed{seed}_{bl_name}"
                if run_id in done:
                    continue

                try:
                    # Determine the objective mode from sc["id"] (the
                    # experiment being run), and force BOTH prob_before
                    # and prob_after to use that SAME scoring function
                    # -- this is the core fix. prob_before uses the
                    # generic "baseline" scene for instance generation
                    # (pre-shift task distribution) but is scored with
                    # the SAME tardiness/fairness/energy function as
                    # prob_after, eliminating the unit mismatch.
                    mode = {
                        spec_module.SCENE_TARDINESS_ID: "tardiness",
                        spec_module.SCENE_FAIRNESS_ID: "fairness",
                        spec_module.SCENE_ENERGY_ID: "energy",
                    }[sc["id"]]

                    prob_before = make_prob_fixed("baseline", spec_module.N_INSTANCES,
                                                    spec_module.N_TASKS, spec_module,
                                                    force_mode=mode)
                    prob_after = make_prob_fixed(sc["id"], spec_module.N_INSTANCES,
                                                   spec_module.N_TASKS, spec_module,
                                                   force_mode=mode)

                    bl_before = 1.0 - prob_before.evaluate(bl_code)
                    bl_after = 1.0 - prob_after.evaluate(bl_code)
                    if bl_before is None or bl_after is None:
                        raise ValueError("baseline eval failed")

                    # precompute once, avoid calling evaluate() twice per baseline
                    all_baseline_scores = {}
                    for bn, bc in baselines.items():
                        v = prob_after.evaluate(bc)
                        all_baseline_scores[bn] = (1.0 - v) if v is not None else None
                    best_static = min(
                        (v for v in all_baseline_scores.values() if v is not None),
                        default=bl_after,
                    )
                    deg_pct = (bl_after - bl_before) / max(abs(bl_before), 1e-9) * 100

                    adapted_code, adapted_ms, llm_calls = _run_enhanced_evo_search(
                        prob_after=prob_after, current_code=bl_code,
                        degraded_ms=bl_after, baseline_ms=bl_before,
                        degradation_pct=deg_pct,
                        scene_change={
                            "id": sc["id"], "after": sc["id"],
                            "change_type": "distributional",
                            "description": spec_module._inst_mod.SCENES[sc["id"]]["description"],
                        },
                        algo_name=bl_name,
                    )
                    # lower-is-better throughout now: "beats" means
                    # adapted is LOWER than the best static reference
                    beats = (adapted_ms is not None and adapted_ms < best_static)

                    # FIX: percentage-change against a near-zero
                    # denominator (bl_after) blows up arbitrarily --
                    # this happened for EDF on tardiness, where EDF's
                    # own tardiness score is legitimately near 0,
                    # producing improvement_pct values in the trillions
                    # even though the ABSOLUTE change is tiny/meaningless.
                    # Use a floor tied to the scale of typical scores in
                    # this domain, not a tiny epsilon, and flag
                    # denominator-degenerate cases explicitly rather
                    # than reporting a nonsensical percentage.
                    DEGENERATE_DENOM_FLOOR = 0.5  # tune per-objective if needed
                    if adapted_ms is None:
                        imp_pct = 0.0
                        denom_degenerate = False
                    elif abs(bl_after) < DEGENERATE_DENOM_FLOOR:
                        # baseline itself already near-perfect on this
                        # objective -- report absolute change instead of
                        # a percentage, and flag it
                        imp_pct = None
                        denom_degenerate = True
                    else:
                        imp_pct = (bl_after - adapted_ms) / abs(bl_after) * 100
                        denom_degenerate = False
                    err = None
                except Exception as e:
                    beats, imp_pct, llm_calls, err = False, 0.0, 0, str(e)

                record = {
                    "run_id": run_id, "scene_id": sc["id"],
                    "objective_name": sc["name"], "alpha_proxy": sc["alpha_proxy"],
                    "seed": seed, "baseline": bl_name,
                    "beats_best": bool(beats),
                    "improvement_pct": imp_pct,  # may be None if denom_degenerate
                    "absolute_change": float(bl_after - adapted_ms) if adapted_ms is not None else 0.0,
                    "denom_degenerate": denom_degenerate,
                    "llm_calls": int(llm_calls), "error": err,
                }
                results.append(record)
                done.add(run_id)
                spec_module.save_results(results, output_file)
                imp_str = f"{imp_pct:.1f}%" if imp_pct is not None else "N/A(denom~0)"
                print(f"  {'OK' if not err else 'ERR'} {run_id}: "
                      f"BB={beats}, imp={imp_str}, calls={llm_calls}")

    print(f"\nDone. Results written to {output_file}")
    print("Compare against the original results/objective_spectrum_raw.json")
    print("for the same three objectives to see the before/after difference.")


def run_causal_fixed(spec_module):
    """
    Corrected replacement for run_objective_spectrum.run_causal.
    Fixes two bugs, neither touched by the earlier run_spectrum_fixed
    patch since this is a separate function:
      1. `beats = (adapted_ms is not None and adapted_ms > best_static)`
         used the OLD "higher is better" convention and a numpy bool_
         that isn't JSON-serializable (`TypeError: Object of type
         bool_ is not JSON serializable`).
      2. Went through the original (unpatched) make_prob / evaluate(),
         which has the same unit-mismatch and sign-inversion bugs
         documented in make_evaluate_fixed's docstring above.
    """
    from run_poms_experiments import _run_enhanced_evo_search

    baselines = spec_module.get_baselines()
    output_file = "results/causal_FIXED.json"
    results = spec_module.load_results(output_file)
    done = {r["run_id"] for r in results}

    conditions = [
        {"name": "control", "inject": False},
        {"name": "treatment", "inject": True},
    ]
    total = len(conditions) * len(spec_module.SEEDS) * len(baselines)
    print(f"FIXED causal experiment: {total} trials\n")

    for cond in conditions:
        print(f"-- {cond['name']} ({'with energy operator' if cond['inject'] else 'without'}) --")
        desc_base = spec_module._inst_mod.SCENES[spec_module.SCENE_ENERGY_ID]["description"]
        desc = desc_base + (spec_module.ENERGY_VOCAB_EXT if cond["inject"] else "")

        for seed in spec_module.SEEDS:
            np.random.seed(seed)
            for bl_name, bl_code in baselines.items():
                run_id = f"FIXED_causal_{cond['name']}_seed{seed}_{bl_name}"
                if run_id in done:
                    continue

                try:
                    prob_before = make_prob_fixed("baseline", spec_module.N_INSTANCES,
                                                    spec_module.N_TASKS, spec_module,
                                                    force_mode="energy")
                    prob_after = make_prob_fixed(spec_module.SCENE_ENERGY_ID,
                                                   spec_module.N_INSTANCES, spec_module.N_TASKS,
                                                   spec_module, force_mode="energy")

                    bl_before = 1.0 - prob_before.evaluate(bl_code)
                    bl_after = 1.0 - prob_after.evaluate(bl_code)
                    if bl_before is None or bl_after is None:
                        raise ValueError("baseline eval failed")

                    all_baseline_scores = {}
                    for bn, bc in baselines.items():
                        v = prob_after.evaluate(bc)
                        all_baseline_scores[bn] = (1.0 - v) if v is not None else None
                    best_static = min(
                        (v for v in all_baseline_scores.values() if v is not None),
                        default=bl_after,
                    )
                    deg_pct = (bl_after - bl_before) / max(abs(bl_before), 1e-9) * 100

                    adapted_code, adapted_ms, llm_calls = _run_enhanced_evo_search(
                        prob_after=prob_after, current_code=bl_code,
                        degraded_ms=bl_after, baseline_ms=bl_before,
                        degradation_pct=deg_pct,
                        scene_change={"id": spec_module.SCENE_ENERGY_ID,
                                      "after": spec_module.SCENE_ENERGY_ID,
                                      "change_type": "distributional", "description": desc},
                        algo_name=bl_name,
                    )
                    # lower-is-better throughout, same as run_spectrum_fixed
                    beats = bool(adapted_ms is not None and adapted_ms < best_static)
                    if adapted_ms is None:
                        imp_pct = 0.0
                    elif abs(bl_after) < 0.5:
                        imp_pct = None
                    else:
                        imp_pct = (bl_after - adapted_ms) / abs(bl_after) * 100
                    err = None
                except Exception as e:
                    beats, imp_pct, llm_calls, err = False, 0.0, 0, str(e)

                record = {
                    "run_id": run_id, "condition": cond["name"],
                    "inject_energy_vocab": bool(cond["inject"]),
                    "seed": seed, "baseline": bl_name,
                    "beats_best": bool(beats),  # explicit bool() -- fixes the JSON bug
                    "improvement_pct": imp_pct,
                    "llm_calls": int(llm_calls), "error": err,
                }
                results.append(record)
                done.add(run_id)
                spec_module.save_results(results, output_file)
                imp_str = f"{imp_pct:.1f}%" if imp_pct is not None else "N/A(denom~0)"
                print(f"  {'OK' if not err else 'ERR'} {run_id}: BB={beats}, imp={imp_str}, calls={llm_calls}")

    print(f"\nDone. Results written to {output_file}")

    # quick control-vs-treatment BB% comparison
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)
    print("\n-- Control vs Treatment BB% --")
    for cond_name in ["control", "treatment"]:
        recs = by_cond.get(cond_name, [])
        if recs:
            n = len(recs)
            beats = sum(1 for r in recs if r["beats_best"])
            print(f"  {cond_name:10s}: BB={beats}/{n} = {100*beats/n:.1f}%")
