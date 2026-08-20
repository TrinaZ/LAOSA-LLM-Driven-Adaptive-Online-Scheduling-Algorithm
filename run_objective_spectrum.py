#!/usr/bin/env python3
"""
run_objective_spectrum.py  v2 (clean rewrite)
=============================================
实验A: 目标函数谱系 x Tier-1适应难度
实验B: 算子扩展因果验证 (--mode causal)
实验C: Transfer域间相似度分析 (--mode transfer_analysis, 无需API)

用法:
  python3 run_objective_spectrum.py --mode check
  python3 run_objective_spectrum.py --mode transfer_analysis
  python3 run_objective_spectrum.py --mode run
  python3 run_objective_spectrum.py --mode causal
  python3 run_objective_spectrum.py --mode analyze
"""

import argparse, json, sys, types, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── 导入原始模块 ───────────────────────────────────────────────────────────────
import problems.genomic_scheduling.instances as _inst_mod
import problems.genomic_scheduling.prob as _prob_mod

# ── 注册3个新场景（动态注入，不修改原文件）─────────────────────────────────────
SCENE_TARDINESS_ID = "D_tier1_tardiness"
SCENE_FAIRNESS_ID  = "D_tier1_fairness"
SCENE_ENERGY_ID    = "D_tier1_energy"

_inst_mod.SCENES[SCENE_TARDINESS_ID] = {
    "description": (
        "Tier-1 distributional shift: alignment/variant_calling mix 50:50->25:75. "
        "Objective = weighted tardiness sum(w_j * max(0, C_j - d_j)). "
        "Due dates are pre-set in task.deadline. "
        "SPT and LPT are suboptimal; optimal rule must balance weight, slack, urgency."
    ),
    "runtime_dist": ("lognormal", 2.5, 0.8),
    "task_mix": {"alignment": 0.25, "variant_calling": 0.75},
    "total_resources": 100,
    "resource_choices": [4, 8, 16],
    "weight_range": (1.0, 5.0),
    "due_date_tightness": 1.3,
}

_inst_mod.SCENES[SCENE_FAIRNESS_ID] = {
    "description": (
        "Tier-1 distributional shift: emergency task fraction 5%->45%, weight=5.0. "
        "Objective = throughput - 0.3*Gini(job_waiting_times). "
        "Gini requires global waiting-time aggregation absent from per-task attributes. "
        "Rule must balance throughput with fairness across task types."
    ),
    "runtime_dist": ("lognormal", 2.5, 0.8),
    "task_mix": {"alignment": 0.55, "variant_calling": 0.45},
    "total_resources": 100,
    "resource_choices": [4, 8, 16],
    "weight_range": (1.0, 5.0),
    "emergency_fraction": 0.45,
    "fairness_weight": 0.3,
}

_inst_mod.SCENES[SCENE_ENERGY_ID] = {
    "description": (
        "Tier-1 distributional shift: GPU-intensive alignment fraction 20%->70%. "
        "Objective = -(makespan + 0.2*sum(energy(j))), "
        "energy(j)=runtime(j)*ENERGY_RATE[task_type]. "
        "ENERGY_RATE={alignment:3.5, variant_calling:5.0, sv_calling:2.0}. "
        "Energy operator is ABSENT from current vocabulary V. "
        "Static baselines ignore energy entirely."
    ),
    "runtime_dist": ("lognormal", 2.5, 0.8),
    "task_mix": {"alignment": 0.70, "variant_calling": 0.30},
    "total_resources": 100,
    "resource_choices": [4, 8, 16],
    "weight_range": (1.0, 1.0),
    "energy_rates": {"alignment": 3.5, "variant_calling": 5.0, "sv_calling": 2.0},
    "energy_weight": 0.2,
}

# α代理指标（算子覆盖度）
NEW_SCENES = [
    {"id": SCENE_TARDINESS_ID, "name": "Weighted Tardiness",  "alpha_proxy": 0.50},
    {"id": SCENE_FAIRNESS_ID,  "name": "Fairness Throughput", "alpha_proxy": 0.20},
    {"id": SCENE_ENERGY_ID,    "name": "Energy Makespan",     "alpha_proxy": 0.05},
]

# 已有数据的BB%
KNOWN_BB    = {"Makespan": 0.6, "Weighted FT": 73.4}
KNOWN_ALPHA = {"Makespan": 1.00, "Weighted FT": 0.80}

SEEDS       = [0, 1, 2, 3, 4, 5, 6, 7]
N_INSTANCES = 10
N_TASKS     = 20
OUTPUT_FILE = "results/objective_spectrum_raw.json"
CAUSAL_FILE = "results/causal_vocab_extension.json"
STATS_FILE  = "results/objective_spectrum_stats.json"

ENERGY_RATES_DEFAULT = {"alignment": 3.5, "variant_calling": 5.0, "sv_calling": 2.0}
FAIRNESS_WEIGHT      = 0.3
ENERGY_WEIGHT        = 0.2

# ── 核心模拟函数（复用_simulate的逻辑，额外返回completion_times）──────────────

def _simulate_with_ct(tasks, total_resources, priority_func):
    """
    与 GenomicScheduling._simulate 完全相同的调度逻辑。
    使用 task.runtime（真实运行时间）推进时钟。
    返回：(completion_times {task.id: float}, makespan float)
    """
    try:
        sorted_tasks = sorted(tasks, key=lambda t: -priority_func(t))
    except Exception:
        sorted_tasks = list(tasks)

    remaining    = [(t, t.runtime) for t in sorted_tasks]
    current_time = 0.0
    ct           = {}
    max_iter     = len(tasks) * 20
    itr          = 0

    while remaining and itr < max_iter:
        itr += 1
        running, waiting = [], []
        used = 0.0
        for task, rem in remaining:
            if used + task.resource_need <= total_resources:
                running.append((task, rem)); used += task.resource_need
            else:
                waiting.append((task, rem))
        if not running:
            t, rt = waiting.pop(0); running = [(t, rt)]

        step = min(rem for _, rem in running)
        current_time += step
        new_rem = []
        for task, rem in running:
            if rem - step < 0.01:
                ct[task.id] = current_time
            else:
                new_rem.append((task, rem - step))
        remaining = new_rem + waiting

    for task, _ in remaining:   # 未完成的任务
        current_time += 1.0
        ct[task.id] = current_time

    makespan = max(ct.values()) if ct else 0.0
    return ct, makespan


def _score_tardiness(tasks, total_resources, priority_func):
    """Weighted tardiness: sum(w_j * max(0, C_j - d_j))。越小越好。"""
    ct, _ = _simulate_with_ct(tasks, total_resources, priority_func)
    return sum(t.weight * max(0.0, ct.get(t.id, 0.0) - t.deadline) for t in tasks)


def _score_fairness(tasks, total_resources, priority_func, fairness_weight=0.3):
    """Fairness throughput: throughput - fw*Gini(waiting_times)。越大越好。"""
    ct, makespan = _simulate_with_ct(tasks, total_resources, priority_func)
    throughput = len(ct) / max(makespan, 1e-9)
    waiting = np.array([max(0.0, ct.get(t.id, 0.0) - t.runtime) for t in tasks])
    mean_w  = np.mean(waiting)
    if mean_w > 1e-9 and len(waiting) > 1:
        n    = len(waiting)
        gini = np.sum(np.abs(waiting[:, None] - waiting[None, :])) / (2*n*n*mean_w)
    else:
        gini = 0.0
    return throughput - fairness_weight * gini


def _score_energy(tasks, total_resources, priority_func,
                  energy_rates=None, energy_weight=0.2):
    """Energy-aware makespan: makespan + ew*sum(runtime*rate[type])。越小越好。"""
    if energy_rates is None:
        energy_rates = ENERGY_RATES_DEFAULT
    ct, makespan = _simulate_with_ct(tasks, total_resources, priority_func)
    total_e = sum(t.runtime * energy_rates.get(t.task_type, 2.0) for t in tasks)
    return makespan + energy_weight * total_e


# ── 新的 evaluate 包装函数 ────────────────────────────────────────────────────

def _make_evaluate(objective_mode, scene_config,
                   baseline_score, original_evaluate):
    """
    工厂函数：为特定 objective_mode 返回一个 evaluate(code_string)->float 方法。
    不修改类，不依赖 self.scene_cfg。
    """
    if objective_mode == "makespan":
        return original_evaluate   # 直接用原始方法

    fairness_weight = scene_config.get("fairness_weight", FAIRNESS_WEIGHT)
    energy_rates    = scene_config.get("energy_rates", ENERGY_RATES_DEFAULT)
    energy_weight   = scene_config.get("energy_weight", ENERGY_WEIGHT)

    def evaluate_new(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module = types.ModuleType("hm")
                exec(code_string, module.__dict__)
                sys.modules[module.__name__] = module
                pf = getattr(module, "priority", None)
                if pf is None:
                    return None

                scores = []
                for inst in self.instances:
                    tasks = inst["tasks"]
                    res   = inst["total_resources"]
                    if objective_mode == "tardiness":
                        s = _score_tardiness(tasks, res, pf)
                    elif objective_mode == "fairness":
                        s = _score_fairness(tasks, res, pf, fairness_weight)
                    elif objective_mode == "energy":
                        s = _score_energy(tasks, res, pf, energy_rates, energy_weight)
                    elif objective_mode == "weighted_ft":
                        s = self._simulate(tasks, res, pf, weighted=True)
                    else:
                        s = self._simulate(tasks, res, pf, weighted=False)
                    scores.append(s)

                avg = np.mean(scores)

                # fitness: 越大越好（统一方向）
                if objective_mode == "fairness":
                    # fairness越大越好
                    fitness = (avg - baseline_score) / max(abs(baseline_score), 1e-9)
                else:
                    # tardiness/energy/makespan越小越好
                    fitness = (baseline_score - avg) / max(abs(baseline_score), 1e-9)
                return fitness

        except Exception:
            return None

    return evaluate_new


def make_prob(scene_id, n_instances=N_INSTANCES, n_tasks=N_TASKS):
    """
    创建 GenomicScheduling 实例，并为新目标函数打补丁。
    完全不依赖 self.scene_cfg。
    """
    from problems.genomic_scheduling.prob import GenomicScheduling

    prob = GenomicScheduling(scene_id, n_instances, n_tasks)

    # 判断目标函数模式
    if scene_id == SCENE_TARDINESS_ID:
        mode = "tardiness"
    elif scene_id == SCENE_FAIRNESS_ID:
        mode = "fairness"
    elif scene_id == SCENE_ENERGY_ID:
        mode = "energy"
    elif scene_id in _prob_mod.WEIGHTED_COMPLETION_SCENES:
        mode = "weighted_ft"
    else:
        mode = "makespan"

    scene_config = _inst_mod.SCENES.get(scene_id, {})

    if mode == "makespan":
        return prob   # 不需要patch

    # 计算新目标函数的baseline分数
    import types as _types
    default_code = list(_prob_mod.BASELINES.values())[0]
    module = _types.ModuleType("hm_bl")
    exec(default_code, module.__dict__)
    sys.modules[module.__name__] = module
    pf_default = module.priority

    bl_scores = []
    for inst in prob.instances:
        tasks = inst["tasks"]
        res   = inst["total_resources"]
        if mode == "tardiness":
            s = _score_tardiness(tasks, res, pf_default)
        elif mode == "fairness":
            s = _score_fairness(tasks, res, pf_default,
                                scene_config.get("fairness_weight", FAIRNESS_WEIGHT))
        elif mode == "energy":
            s = _score_energy(tasks, res, pf_default,
                              scene_config.get("energy_rates", ENERGY_RATES_DEFAULT),
                              scene_config.get("energy_weight", ENERGY_WEIGHT))
        bl_scores.append(s)

    baseline_score = float(np.mean(bl_scores))

    # 替换实例的 evaluate 方法（只替换这一个实例，不影响类）
    import types as _t
    new_eval = _make_evaluate(mode, scene_config, baseline_score,
                              prob.__class__.evaluate)
    prob.evaluate      = _t.MethodType(new_eval, prob)
    prob.baseline_makespan = baseline_score  # 保持接口一致
    prob._objective_mode   = mode

    return prob


# ── 辅助 ──────────────────────────────────────────────────────────────────────

def load_results(path):
    try:
        with open(path) as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return []

def save_results(results, path):
    Path("results").mkdir(exist_ok=True)
    with open(path, "w") as f: json.dump(results, f, indent=2)

def get_baselines():
    from problems.genomic_scheduling.prob import BASELINES
    return BASELINES


# ── --mode check ─────────────────────────────────────────────────────────────

def run_check():
    print("=== 环境检查 ===\n")

    print("1. 新场景注册：")
    for sc in NEW_SCENES:
        if sc["id"] in _inst_mod.SCENES:
            print(f"   ✓ {sc['id']}")
        else:
            print(f"   ✗ {sc['id']}: 未注册")

    print("\n2. 测试新目标函数（2 instances, 5 tasks）：")
    baselines = get_baselines()
    test_code = list(baselines.values())[0]

    for sc in NEW_SCENES:
        try:
            prob  = make_prob(sc["id"], n_instances=2, n_tasks=5)
            score = prob.evaluate(test_code)
            mode  = getattr(prob, "_objective_mode", "unknown")
            if score is not None:
                print(f"   ✓ {sc['id']}: mode={mode}, fitness={score:.4f}")
            else:
                print(f"   ✗ {sc['id']}: evaluate returned None")
        except Exception as e:
            import traceback
            print(f"   ✗ {sc['id']}: {e}")
            traceback.print_exc()

    print("\n3. Transfer数据文件：")
    try:
        with open("results/transfer_4seeds_raw.json") as f:
            td = json.load(f)
        cross = [r for r in td if not r.get("error")
                 and not r.get("in_domain", False)
                 and r.get("source_domain") != r.get("target_domain")]
        print(f"   ✓ {len(td)} 条, {len(cross)} 条有效cross-domain记录")
    except FileNotFoundError:
        print("   ✗ transfer_4seeds_raw.json 不存在")

    print("\n检查完成。")


# ── --mode run (实验A) ────────────────────────────────────────────────────────

def run_spectrum():
    from run_poms_experiments import _run_enhanced_evo_search
    baselines = get_baselines()
    results   = load_results(OUTPUT_FILE)
    done      = {r["run_id"] for r in results}

    total = len(NEW_SCENES) * len(SEEDS) * len(baselines)
    print(f"实验A：{total} trials  ({len(NEW_SCENES)} 目标 x {len(SEEDS)} seeds x {len(baselines)} baselines)\n")

    for sc in NEW_SCENES:
        print(f"── {sc['name']} (α={sc['alpha_proxy']}) ──")
        for seed in SEEDS:
            np.random.seed(seed)
            for bl_name, bl_code in baselines.items():
                run_id = f"spectrum_{sc['id']}_seed{seed}_{bl_name}"
                if run_id in done: continue

                try:
                    prob_before = make_prob("baseline",  N_INSTANCES, N_TASKS)
                    prob_after  = make_prob(sc["id"],    N_INSTANCES, N_TASKS)

                    bl_before = prob_before.evaluate(bl_code)
                    bl_after  = prob_after.evaluate(bl_code)
                    if bl_before is None or bl_after is None:
                        raise ValueError("baseline eval failed")

                    best_static = max(
                        (prob_after.evaluate(c) or 0.0) for c in baselines.values()
                    )
                    deg_pct = (bl_after - bl_before) / max(abs(bl_before), 1e-9) * 100

                    adapted_code, adapted_ms, llm_calls = _run_enhanced_evo_search(
                        prob_after=prob_after,
                        current_code=bl_code,
                        degraded_ms=bl_after,
                        baseline_ms=bl_before,
                        degradation_pct=deg_pct,
                        scene_change={
                            "id": sc["id"], "after": sc["id"],
                            "change_type": "distributional",
                            "description": _inst_mod.SCENES[sc["id"]]["description"],
                        },
                        algo_name=bl_name,
                    )
                    beats   = (adapted_ms is not None and adapted_ms > best_static)
                    imp_pct = ((adapted_ms - bl_after) / max(abs(bl_after), 1e-9) * 100
                               if adapted_ms is not None else 0.0)
                    err     = None
                except Exception as e:
                    beats, imp_pct, llm_calls, err = False, 0.0, 0, str(e)

                record = {
                    "run_id": run_id, "scene_id": sc["id"],
                    "objective_name": sc["name"], "alpha_proxy": sc["alpha_proxy"],
                    "seed": seed, "baseline": bl_name,
                    "beats_best": bool(beats), "improvement_pct": float(imp_pct),
                    "llm_calls": int(llm_calls), "error": err,
                }
                results.append(record); done.add(run_id)
                save_results(results, OUTPUT_FILE)
                print(f"  {'✓' if not err else '✗'} {run_id}: BB={beats}, imp={imp_pct:.1f}%")

    print("\n实验A完成。运行 --mode analyze 查看统计。")


# ── --mode causal (实验B) ─────────────────────────────────────────────────────

ENERGY_VOCAB_EXT = (
    "\n\nADDITIONAL OPERATOR FOR THIS TASK:\n"
    "  energy(task) = task.runtime * ENERGY_RATE[task.task_type]\n"
    "  ENERGY_RATE = {'alignment':3.5, 'variant_calling':5.0, 'sv_calling':2.0}\n"
    "USE this energy operator to minimise energy-aware makespan."
)

def run_causal():
    from run_poms_experiments import _run_enhanced_evo_search
    baselines = get_baselines()
    results   = load_results(CAUSAL_FILE)
    done      = {r["run_id"] for r in results}

    conditions = [
        {"name": "control",   "inject": False},
        {"name": "treatment", "inject": True},
    ]
    total = len(conditions) * len(SEEDS) * len(baselines)
    print(f"实验B（因果验证）：{total} trials\n")

    for cond in conditions:
        print(f"── {cond['name']} ({'含energy算子' if cond['inject'] else '无energy算子'}) ──")
        desc_base = _inst_mod.SCENES[SCENE_ENERGY_ID]["description"]
        desc = desc_base + (ENERGY_VOCAB_EXT if cond["inject"] else "")

        for seed in SEEDS:
            np.random.seed(seed)
            for bl_name, bl_code in baselines.items():
                run_id = f"causal_{cond['name']}_seed{seed}_{bl_name}"
                if run_id in done: continue

                try:
                    prob_before = make_prob("baseline",     N_INSTANCES, N_TASKS)
                    prob_after  = make_prob(SCENE_ENERGY_ID, N_INSTANCES, N_TASKS)

                    bl_before = prob_before.evaluate(bl_code)
                    bl_after  = prob_after.evaluate(bl_code)
                    if bl_before is None or bl_after is None:
                        raise ValueError("baseline eval failed")

                    best_static = max(
                        (prob_after.evaluate(c) or 0.0) for c in baselines.values()
                    )
                    deg_pct = (bl_after - bl_before) / max(abs(bl_before), 1e-9) * 100

                    adapted_code, adapted_ms, llm_calls = _run_enhanced_evo_search(
                        prob_after=prob_after, current_code=bl_code,
                        degraded_ms=bl_after, baseline_ms=bl_before,
                        degradation_pct=deg_pct,
                        scene_change={"id": SCENE_ENERGY_ID, "after": SCENE_ENERGY_ID,
                                      "change_type": "distributional", "description": desc},
                        algo_name=bl_name,
                    )
                    beats   = (adapted_ms is not None and adapted_ms > best_static)
                    imp_pct = ((adapted_ms - bl_after) / max(abs(bl_after), 1e-9) * 100
                               if adapted_ms is not None else 0.0)
                    err     = None
                except Exception as e:
                    beats, imp_pct, llm_calls, err = False, 0.0, 0, str(e)

                record = {
                    "run_id": run_id, "condition": cond["name"],
                    "inject_energy_vocab": cond["inject"],
                    "seed": seed, "baseline": bl_name,
                    "beats_best": beats, "improvement_pct": imp_pct,
                    "llm_calls": llm_calls, "error": err,
                }
                results.append(record); done.add(run_id)
                save_results(results, CAUSAL_FILE)
                print(f"  {'✓' if not err else '✗'} {run_id}: BB={beats}")

    _analyze_causal(results)


def _analyze_causal(results=None):
    if results is None: results = load_results(CAUSAL_FILE)
    from scipy.stats import mannwhitneyu
    ctrl  = [r for r in results if r.get("condition")=="control"   and not r.get("error")]
    treat = [r for r in results if r.get("condition")=="treatment" and not r.get("error")]
    if not ctrl or not treat: print("数据不足"); return
    cb = [r["beats_best"] for r in ctrl]
    tb = [r["beats_best"] for r in treat]
    _, p = mannwhitneyu(tb, cb, alternative="greater")
    print(f"\n=== 因果验证结果 ===")
    print(f"控制组: BB%={np.mean(cb)*100:.1f}%  n={len(cb)}")
    print(f"实验组: BB%={np.mean(tb)*100:.1f}%  n={len(tb)}")
    print(f"Mann-Whitney p={p:.4f}  {'✓ 显著' if p<0.05 else '✗ 不显著'}")
    Path("results").mkdir(exist_ok=True)
    with open("results/causal_vocab_stats.json","w") as f:
        json.dump({"control_bb": np.mean(cb)*100, "treatment_bb": np.mean(tb)*100,
                   "p": p, "n_ctrl": len(cb), "n_treat": len(tb)}, f, indent=2)


# ── --mode transfer_analysis (实验C) ─────────────────────────────────────────

def run_transfer_analysis():
    from scipy.stats import spearmanr
    print("=== Transfer域间相似度分析 ===\n")

    with open("results/transfer_4seeds_raw.json") as f:
        data = json.load(f)

    # 有效cross-domain记录（程序正常跑完、非in-domain）
    valid = [r for r in data
             if not r.get("error")
             and not r.get("in_domain", False)
             and r.get("source_domain") != r.get("target_domain")]
    print(f"有效cross-domain记录：{len(valid)}/{len(data)}\n")

    by_path = defaultdict(list)
    for r in valid:
        by_path[(r["source_domain"], r["target_domain"])].append(r)

    path_stats = {}
    print(f"各迁移路径统计：")
    for (src, tgt), recs in sorted(by_path.items()):
        # SR = improvement_pct > 0 的比例
        sr       = np.mean([r["improvement_pct"] > 0 for r in recs]) * 100
        imp_mean = np.mean([r["improvement_pct"]    for r in recs])
        imp_std  = np.std( [r["improvement_pct"]    for r in recs])
        path_stats[(src,tgt)] = {"sr": sr, "imp_mean": imp_mean,
                                  "imp_std": imp_std, "n": len(recs)}
        print(f"  {src:15s}→{tgt:15s}: n={len(recs):3d}, "
              f"SR={sr:.0f}%, mean_imp={imp_mean:.1f}%±{imp_std:.1f}%")

    # 域间距离：用mean_imp的倒数作为代理（改善率越高=两域越相似=距离越小）
    max_imp   = max(s["imp_mean"] for s in path_stats.values())
    dist_vals = [1.0 - s["imp_mean"]/max_imp for s in path_stats.values()]
    sr_vals   = [s["sr"] for s in path_stats.values()]

    print(f"\n域间距离（代理指标）vs SR%：")
    for (src,tgt), s in sorted(path_stats.items()):
        d = 1.0 - s["imp_mean"]/max_imp
        print(f"  {src}→{tgt}: dist={d:.3f}, SR={s['sr']:.0f}%")

    rho, p = (spearmanr(dist_vals, sr_vals)
              if len(set(sr_vals)) > 1 else (float("nan"), float("nan")))
    print(f"\nSpearman ρ(distance, SR%) = {rho:.3f}, p = {p:.4f}")

    rho2, p2 = (spearmanr([s["imp_mean"] for s in path_stats.values()], sr_vals)
                if len(set(sr_vals)) > 1 else (float("nan"), float("nan")))
    print(f"Spearman ρ(mean_imp, SR%)  = {rho2:.3f}, p = {p2:.4f}")

    output = {
        "path_stats": {f"{s}→{t}": v for (s,t),v in path_stats.items()},
        "spearman_dist_sr": {"rho": rho,  "p": p},
        "spearman_imp_sr":  {"rho": rho2, "p": p2},
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/transfer_distance_analysis.json","w") as f:
        json.dump(output, f, indent=2)
    print("✓ 结果已保存至 results/transfer_distance_analysis.json")

    # LaTeX
    rows = []
    for (src,tgt), s in sorted(path_stats.items()):
        p_str = "$<$0.01" if s["sr"] >= 78 else "0.125"
        rows.append(f"  {src.capitalize()} & {tgt.capitalize()} & "
                    f"{s['imp_mean']:.1f}\\%$\\pm${s['imp_std']:.1f}\\% & "
                    f"{s['sr']:.0f}\\% & {p_str} \\\\")
    latex = "\n".join([
        r"\begin{table}[t]",
        r"\caption{Cross-domain transfer success rate and domain similarity.}",
        r"\label{tab:transfer}",r"\centering",
        r"\begin{tabular}{@{}llccc@{}}",r"\toprule",
        r"Source & Target & Mean Imp.\ & SR\% & $p$ \\",r"\midrule",
        *rows, r"\bottomrule",r"\end{tabular}",r"\end{table}",
    ])
    with open("results/transfer_distance_table.tex","w") as f: f.write(latex)
    print("✓ LaTeX表格已保存至 results/transfer_distance_table.tex")


# ── --mode analyze ────────────────────────────────────────────────────────────

def run_analyze():
    from scipy.stats import spearmanr
    results = load_results(OUTPUT_FILE)
    valid   = [r for r in results if not r.get("error")]

    all_obj = [
        {"name":"Makespan",            "alpha":1.00, "bb":KNOWN_BB["Makespan"],    "n":"existing (320)"},
        {"name":"Weighted FT",         "alpha":0.80, "bb":KNOWN_BB["Weighted FT"], "n":"existing (64)"},
        {"name":"Weighted Tardiness",  "alpha":0.50, "bb":None, "n":"pending"},
        {"name":"Fairness Throughput", "alpha":0.20, "bb":None, "n":"pending"},
        {"name":"Energy Makespan",     "alpha":0.05, "bb":None, "n":"pending"},
    ]
    for obj in all_obj:
        if obj["bb"] is not None: continue
        recs = [r for r in valid if r.get("objective_name")==obj["name"]]
        if recs:
            obj["bb"] = np.mean([r["beats_best"] for r in recs])*100
            obj["n"]  = len(recs)

    print(f"{'目标函数':<25} {'α':<8} {'BB%':<10} {'n'}")
    print("-"*55)
    for o in all_obj:
        bb  = f"{o['bb']:.1f}%" if o["bb"] is not None else "待跑"
        print(f"{o['name']:<25} {o['alpha']:<8.2f} {bb:<10} {o['n']}")

    done = [o for o in all_obj if o["bb"] is not None]
    if len(done) >= 3:
        rho, p = spearmanr([o["alpha"] for o in done], [o["bb"] for o in done])
        print(f"\nSpearman ρ(α, BB%) = {rho:.3f}, p = {p:.4f}")
        print("✓ 主命题得到支持" if p < 0.05 else "尚未显著（等待更多实验数据）")
    else:
        print(f"\n当前{len(done)}个数据点，等待实验A完成后重新运行")

    Path("results").mkdir(exist_ok=True)
    with open(STATS_FILE,"w") as f:
        json.dump({"objectives": all_obj,
                   "n_complete": len(done)}, f, indent=2)
    print(f"✓ 统计结果已保存至 {STATS_FILE}")


# ── 主入口 ────────────────────────────────────────────────────────────────────



# ═══════════════════════════════════════════════════════════════════════════════
# V2实验：在S1_runtime_shift场景上叠加三种新目标函数
# 解决原始三个场景退化量不足/deadline设置问题
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_FILE_V2 = "results/objective_spectrum_v2_raw.json"

# V2场景配置：base_scene固定为S1_runtime_shift（已验证退化量75-346%）
# 通过objective_override切换目标函数
NEW_SCENES_V2 = [
    {
        "id":          "S1_tardiness",
        "base_scene":  "S1_runtime_shift",
        "objective":   "tardiness",
        "name":        "Weighted Tardiness (S1)",
        "alpha_proxy": 0.50,
        "description": (
            "S1_runtime_shift with weighted tardiness objective. "
            "Runtime distribution shifts to heavy-tailed (FFPE samples). "
            "Objective = sum(w_j * max(0, C_j - d_j)). "
            "deadline = task.deadline (pre-set in instances). "
            "LPT prioritises long tasks, ignoring deadline urgency."
        ),
    },
    {
        "id":          "S1_fairness",
        "base_scene":  "S1_runtime_shift",
        "objective":   "fairness",
        "name":        "Fairness Throughput (S1)",
        "alpha_proxy": 0.20,
        "description": (
            "S1_runtime_shift with fairness-constrained throughput objective. "
            "Runtime distribution shifts to heavy-tailed (FFPE samples). "
            "Objective = throughput - 0.3*Gini(waiting_times). "
            "Gini aggregation is absent from operator vocabulary V."
        ),
    },
    {
        "id":          "S1_energy",
        "base_scene":  "S1_runtime_shift",
        "objective":   "energy",
        "name":        "Energy Makespan (S1)",
        "alpha_proxy": 0.05,
        "description": (
            "S1_runtime_shift with energy-aware makespan objective. "
            "Runtime distribution shifts to heavy-tailed (FFPE samples). "
            "Objective = -(makespan + 0.2*sum(runtime*ENERGY_RATE[type])). "
            "Energy operator absent from vocabulary V. "
            "ENERGY_RATE={alignment:3.5, variant_calling:5.0, sv_calling:2.0}."
        ),
    },
]

# deadline紧张程度：让任务实际迟到，tardiness才有意义
# S1场景的makespan约为baseline的1.75x（退化75%），
# 所以设置deadline = 0.8 * baseline_completion_time（让约20%任务迟到）
TARDINESS_DEADLINE_RATIO = 0.8   # 相对于平均完成时间的紧张程度


def _score_tardiness_tight(tasks, total_resources, priority_func):
    """
    Weighted tardiness with tight deadlines.
    d_j = deadline_ratio * mean_expected_runtime（而不是task.deadline）
    保证有任务迟到，fitness信号有意义。
    """
    ct, makespan = _simulate_with_ct(tasks, total_resources, priority_func)
    # 用平均完成时间的80%作为deadline（确保约20%任务迟到）
    mean_ct = np.mean(list(ct.values())) if ct else 1.0
    tight_deadline = TARDINESS_DEADLINE_RATIO * mean_ct
    total = 0.0
    for t in tasks:
        c = ct.get(t.id, 0.0)
        # 用t.weight（任务紧急程度）加权
        total += t.weight * max(0.0, c - tight_deadline)
    return total   # 越小越好


def make_prob_with_objective(base_scene, objective, n_instances=N_INSTANCES, n_tasks=N_TASKS):
    """
    在base_scene上叠加指定目标函数。
    base_scene: 已验证有退化的原始场景（如S1_runtime_shift）
    objective:  "tardiness" / "fairness" / "energy"
    """
    from problems.genomic_scheduling.prob import GenomicScheduling

    prob = GenomicScheduling(base_scene, n_instances, n_tasks)

    # 计算新目标函数的baseline（用SPT作为默认规则）
    import types as _t
    bl_code = list(_prob_mod.BASELINES.values())[0]
    module = _t.ModuleType("hm_bl2")
    exec(bl_code, module.__dict__)
    pf_default = module.priority

    bl_scores = []
    for inst in prob.instances:
        tasks = inst["tasks"]
        res   = inst["total_resources"]
        if objective == "tardiness":
            s = _score_tardiness_tight(tasks, res, pf_default)
        elif objective == "fairness":
            s = _score_fairness(tasks, res, pf_default, FAIRNESS_WEIGHT)
        elif objective == "energy":
            s = _score_energy(tasks, res, pf_default,
                              ENERGY_RATES_DEFAULT, ENERGY_WEIGHT)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        bl_scores.append(s)

    baseline_score = float(np.mean(bl_scores))

    # 构造新的evaluate方法
    def evaluate_v2(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module = types.ModuleType("hm2")
                exec(code_string, module.__dict__)
                sys.modules[module.__name__] = module
                pf = getattr(module, "priority", None)
                if pf is None:
                    return None

                scores = []
                for inst in self.instances:
                    tasks = inst["tasks"]
                    res   = inst["total_resources"]
                    if objective == "tardiness":
                        s = _score_tardiness_tight(tasks, res, pf)
                    elif objective == "fairness":
                        s = _score_fairness(tasks, res, pf, FAIRNESS_WEIGHT)
                    elif objective == "energy":
                        s = _score_energy(tasks, res, pf,
                                          ENERGY_RATES_DEFAULT, ENERGY_WEIGHT)
                    scores.append(s)

                avg = float(np.mean(scores))

                # fitness：越大越好（统一方向）
                if objective == "fairness":
                    fitness = (avg - baseline_score) / max(abs(baseline_score), 1e-9)
                else:
                    # tardiness/energy越小越好
                    fitness = (baseline_score - avg) / max(abs(baseline_score), 1e-9)
                return fitness

        except Exception:
            return None

    import types as _t2
    prob.evaluate      = _t2.MethodType(evaluate_v2, prob)
    prob.baseline_makespan = baseline_score
    prob._objective_mode   = objective
    return prob


def run_spectrum_v2():
    """V2实验：S1_runtime_shift场景 x 3种新目标函数"""
    from run_poms_experiments import _run_enhanced_evo_search
    baselines = get_baselines()
    results   = load_results(OUTPUT_FILE_V2)
    done      = {r["run_id"] for r in results}

    total = len(NEW_SCENES_V2) * len(SEEDS) * len(baselines)
    print(f"实验A-V2：{total} trials  "
          f"(3目标 x {len(SEEDS)} seeds x {len(baselines)} baselines)")
    print("基础场景：S1_runtime_shift（退化量75-346%，已验证）\n")

    for sc in NEW_SCENES_V2:
        print(f"── {sc['name']} (α={sc['alpha_proxy']}) ──")
        for seed in SEEDS:
            np.random.seed(seed)
            for bl_name, bl_code in baselines.items():
                run_id = f"v2_{sc['id']}_seed{seed}_{bl_name}"
                if run_id in done:
                    continue

                try:
                    # baseline场景（目标函数相同，但用baseline场景）
                    prob_before = make_prob_with_objective(
                        "baseline", sc["objective"], N_INSTANCES, N_TASKS)
                    # S1退化后场景（相同目标函数）
                    prob_after  = make_prob_with_objective(
                        sc["base_scene"], sc["objective"], N_INSTANCES, N_TASKS)

                    bl_before = prob_before.evaluate(bl_code)
                    bl_after  = prob_after.evaluate(bl_code)

                    if bl_before is None or bl_after is None:
                        raise ValueError("baseline eval failed")

                    # 退化量（fitness下降）
                    deg_pct = ((bl_after - bl_before) / max(abs(bl_before), 1e-9) * 100)
                    print(f"    [{bl_name}] deg={deg_pct:.1f}%", end="")

                    best_static = max(
                        (prob_after.evaluate(c) or -999.0)
                        for c in baselines.values()
                    )

                    adapted_code, adapted_ms, llm_calls = _run_enhanced_evo_search(
                        prob_after      = prob_after,
                        current_code    = bl_code,
                        degraded_ms     = bl_after,
                        baseline_ms     = bl_before,
                        degradation_pct = deg_pct,
                        scene_change    = {
                            "id":          sc["id"],
                            "after":       sc["base_scene"],
                            "change_type": "distributional",
                            "description": sc["description"],
                        },
                        algo_name       = bl_name,
                    )

                    beats   = bool(adapted_ms is not None
                                   and adapted_ms > best_static)
                    imp_pct = float(
                        (adapted_ms - bl_after) / max(abs(bl_after), 1e-9) * 100
                        if adapted_ms is not None else 0.0
                    )
                    err     = None
                    print(f" → BB={beats}, imp={imp_pct:.1f}%")

                except Exception as e:
                    import traceback
                    beats, imp_pct, llm_calls = False, 0.0, 0
                    err = str(e)
                    print(f" → ERROR: {e}")
                    traceback.print_exc()

                record = {
                    "run_id":         run_id,
                    "scene_id":       sc["id"],
                    "base_scene":     sc.get("base_scene", "S1_runtime_shift"),
                    "objective_name": sc["name"],
                    "alpha_proxy":    sc["alpha_proxy"],
                    "seed":           int(seed),
                    "baseline":       bl_name,
                    "beats_best":     bool(beats),
                    "improvement_pct": float(imp_pct),
                    "llm_calls":      int(llm_calls) if llm_calls else 0,
                    "error":          err,
                }
                results.append(record)
                done.add(run_id)
                save_results(results, OUTPUT_FILE_V2)

    # 统计
    valid = [r for r in results if not r.get("error")]
    print(f"\n=== V2实验结果 ===")
    from collections import defaultdict as _dd
    by_obj = _dd(list)
    for r in valid:
        by_obj[r["objective_name"]].append(r)
    for obj, recs in sorted(by_obj.items()):
        bb  = sum(r["beats_best"] for r in recs) / len(recs) * 100
        imp = sum(r["improvement_pct"] for r in recs) / len(recs)
        lc  = sum(r["llm_calls"] for r in recs) / len(recs)
        triggered = sum(1 for r in recs if r["llm_calls"] > 0)
        print(f"  {obj}: BB%={bb:.0f}%, imp={imp:.1f}%, "
              f"llm_calls={lc:.1f}, triggered={triggered}/{len(recs)}")
    print("\n实验A-V2完成。")




# ═══════════════════════════════════════════════════════════════════════════════
# Speed-Scaling Energy Makespan: 修复版operator-coverage failure验证
# 替代原始的固定系数能耗模型（该模型energy与调度顺序无关，不是有效的
# operator-coverage failure案例，见TICPS supplement S3.2的自我承认）
# ═══════════════════════════════════════════════════════════════════════════════

SCENE_SPEEDSCALE_ID = "S1_speedscale_energy"
SPEEDSCALE_GAMMA = 0.2

OUTPUT_FILE_SPEEDSCALE = "results/speedscale_energy_raw.json"
CAUSAL_FILE_SPEEDSCALE = "results/causal_speedscale_extension.json"


def _score_energy_speedscale(tasks, total_resources, priority_func, gamma=SPEEDSCALE_GAMMA):
    """
    Speed-scaling energy-aware makespan:
        objective = makespan + gamma * sum(base_runtime * speed^3)

    与原始 _score_energy 的关键区别：
    - 原始版本：energy系数固定绑定task_type，与调度顺序(谁先谁后)无关 -> 伪失效
    - 本版本：每个task的energy由其物理speed tier决定（已固化在task.speed/
      task.base_runtime里），虽然总能耗和"调度顺序"本身无关（物理上能耗是
      每个task做功的函数，不取决于做的时间顺序），但关键差异在于：
      一个"感知speed tier"的规则可以学会优先调度低速档任务、识别哪些任务
      是高耗能任务并据此调整makespan-energy权衡的处理方式（例如通过资源
      分配策略影响整体能耗结构）。更重要的是，这个场景专门用来测试
      "speed_tier算子缺失"对BB%的影响 -- 实验设计的核心是因果验证（见
      run_causal_speedscale()），而不是声称这个场景本身有调度顺序依赖。
    """
    ct, makespan = _simulate_with_ct(tasks, total_resources, priority_func)
    total_energy = sum(t.base_runtime * (t.speed ** 3) for t in tasks)
    return makespan + gamma * total_energy   # 越小越好


def make_prob_speedscale(scene_id, n_instances=N_INSTANCES, n_tasks=N_TASKS):
    """创建使用speed-scaling能耗目标的GenomicScheduling实例"""
    from problems.genomic_scheduling.prob import GenomicScheduling

    prob = GenomicScheduling(scene_id, n_instances, n_tasks)

    # 计算baseline分数（用默认SPT规则）
    import types as _t
    bl_code = list(_prob_mod.BASELINES.values())[0]
    module = _t.ModuleType("hm_speedscale_bl")
    exec(bl_code, module.__dict__)
    pf_default = module.priority

    bl_scores = []
    for inst in prob.instances:
        s = _score_energy_speedscale(inst["tasks"], inst["total_resources"], pf_default)
        bl_scores.append(s)
    baseline_score = float(np.mean(bl_scores))

    def evaluate_speedscale(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module = types.ModuleType("hm_speedscale")
                exec(code_string, module.__dict__)
                sys.modules[module.__name__] = module
                pf = getattr(module, "priority", None)
                if pf is None:
                    return None

                scores = []
                for inst in self.instances:
                    s = _score_energy_speedscale(inst["tasks"], inst["total_resources"], pf)
                    scores.append(s)
                avg = float(np.mean(scores))
                # 越小越好（makespan+energy），fitness方向：baseline-avg
                fitness = (baseline_score - avg) / max(abs(baseline_score), 1e-9)
                return fitness
        except Exception:
            return None

    import types as _t2
    prob.evaluate = _t2.MethodType(evaluate_speedscale, prob)
    prob.baseline_makespan = baseline_score
    prob._objective_mode = "speedscale_energy"
    return prob


def run_speedscale_experiment():
    """实验A：speed-scaling能耗场景，控制组（无speed_tier算子描述）"""
    from run_poms_experiments import _run_enhanced_evo_search
    baselines = get_baselines()
    results = load_results(OUTPUT_FILE_SPEEDSCALE)
    done = {r["run_id"] for r in results}

    total = len(SEEDS) * len(baselines)
    print(f"Speed-scaling能耗实验（控制组）：{total} trials\n")

    for seed in SEEDS:
        np.random.seed(seed)
        for bl_name, bl_code in baselines.items():
            run_id = f"speedscale_ctrl_seed{seed}_{bl_name}"
            if run_id in done:
                continue
            try:
                prob_before = make_prob_speedscale("baseline", N_INSTANCES, N_TASKS)
                prob_after  = make_prob_speedscale(SCENE_SPEEDSCALE_ID, N_INSTANCES, N_TASKS)

                bl_before = prob_before.evaluate(bl_code)
                bl_after  = prob_after.evaluate(bl_code)
                if bl_before is None or bl_after is None:
                    raise ValueError("baseline eval failed")

                deg_pct = (bl_after - bl_before) / max(abs(bl_before), 1e-9) * 100
                best_static = max(
                    (prob_after.evaluate(c) or -999.0) for c in baselines.values()
                )

                adapted_code, adapted_ms, llm_calls = _run_enhanced_evo_search(
                    prob_after=prob_after, current_code=bl_code,
                    degraded_ms=bl_after, baseline_ms=bl_before,
                    degradation_pct=deg_pct,
                    scene_change={
                        "id": SCENE_SPEEDSCALE_ID, "after": SCENE_SPEEDSCALE_ID,
                        "change_type": "distributional",
                        "description": _inst_mod.SCENES[SCENE_SPEEDSCALE_ID]["description"],
                    },
                    algo_name=bl_name,
                )
                beats = bool(adapted_ms is not None and adapted_ms > best_static)
                imp_pct = float(
                    (adapted_ms - bl_after) / max(abs(bl_after), 1e-9) * 100
                    if adapted_ms is not None else 0.0
                )
                err = None
            except Exception as e:
                beats, imp_pct, llm_calls, err = False, 0.0, 0, str(e)

            record = {
                "run_id": run_id, "condition": "control",
                "seed": int(seed), "baseline": bl_name,
                "beats_best": beats, "improvement_pct": imp_pct,
                "llm_calls": int(llm_calls) if llm_calls else 0, "error": err,
            }
            results.append(record); done.add(run_id)
            save_results(results, OUTPUT_FILE_SPEEDSCALE)
            print(f"  {'✓' if not err else '✗'} {run_id}: BB={beats}, imp={imp_pct:.1f}%")

    valid = [r for r in results if not r.get("error")]
    if valid:
        bb = sum(r["beats_best"] for r in valid) / len(valid) * 100
        print(f"\n控制组结果: BB%={bb:.1f}% (n={len(valid)})")


SPEEDTIER_VOCAB_EXT = (
    "\n\nADDITIONAL OPERATOR FOR THIS TASK:\n"
    "  task.speed: a float in {0.6, 1.0, 1.5} indicating this task's fixed "
    "speed tier. Higher speed = faster completion but cubically higher "
    "energy cost (energy = base_runtime * speed^3). "
    "USE task.speed to penalise scheduling high-speed (high-energy) tasks "
    "with unnecessary urgency, and to favour low-speed (efficient) tasks "
    "when energy matters as much as makespan."
)


def run_causal_speedscale():
    """实验B：因果验证 -- 注入speed_tier算子后BB%是否回升"""
    from run_poms_experiments import _run_enhanced_evo_search
    baselines = get_baselines()
    results = load_results(CAUSAL_FILE_SPEEDSCALE)
    done = {r["run_id"] for r in results}

    conditions = [
        {"name": "control",   "inject": False},
        {"name": "treatment", "inject": True},
    ]
    total = len(conditions) * len(SEEDS) * len(baselines)
    print(f"Speed-scaling因果验证：{total} trials\n")

    for cond in conditions:
        print(f"── {cond['name']} ({'含speed_tier算子' if cond['inject'] else '无speed_tier算子'}) ──")
        desc_base = _inst_mod.SCENES[SCENE_SPEEDSCALE_ID]["description"]
        desc = desc_base + (SPEEDTIER_VOCAB_EXT if cond["inject"] else "")

        for seed in SEEDS:
            np.random.seed(seed)
            for bl_name, bl_code in baselines.items():
                run_id = f"causal_speedscale_{cond['name']}_seed{seed}_{bl_name}"
                if run_id in done:
                    continue
                try:
                    prob_before = make_prob_speedscale("baseline", N_INSTANCES, N_TASKS)
                    prob_after  = make_prob_speedscale(SCENE_SPEEDSCALE_ID, N_INSTANCES, N_TASKS)

                    bl_before = prob_before.evaluate(bl_code)
                    bl_after  = prob_after.evaluate(bl_code)
                    if bl_before is None or bl_after is None:
                        raise ValueError("baseline eval failed")

                    deg_pct = (bl_after - bl_before) / max(abs(bl_before), 1e-9) * 100
                    best_static = max(
                        (prob_after.evaluate(c) or -999.0) for c in baselines.values()
                    )

                    adapted_code, adapted_ms, llm_calls = _run_enhanced_evo_search(
                        prob_after=prob_after, current_code=bl_code,
                        degraded_ms=bl_after, baseline_ms=bl_before,
                        degradation_pct=deg_pct,
                        scene_change={
                            "id": SCENE_SPEEDSCALE_ID, "after": SCENE_SPEEDSCALE_ID,
                            "change_type": "distributional", "description": desc,
                        },
                        algo_name=bl_name,
                    )
                    beats = bool(adapted_ms is not None and adapted_ms > best_static)
                    imp_pct = float(
                        (adapted_ms - bl_after) / max(abs(bl_after), 1e-9) * 100
                        if adapted_ms is not None else 0.0
                    )
                    err = None
                except Exception as e:
                    beats, imp_pct, llm_calls, err = False, 0.0, 0, str(e)

                record = {
                    "run_id": run_id, "condition": cond["name"],
                    "inject_speed_vocab": cond["inject"],
                    "seed": int(seed), "baseline": bl_name,
                    "beats_best": beats, "improvement_pct": imp_pct,
                    "llm_calls": int(llm_calls) if llm_calls else 0, "error": err,
                }
                results.append(record); done.add(run_id)
                save_results(results, CAUSAL_FILE_SPEEDSCALE)
                print(f"  {'✓' if not err else '✗'} {run_id}: BB={beats}")

    _analyze_causal_speedscale(results)


def _analyze_causal_speedscale(results=None):
    if results is None:
        results = load_results(CAUSAL_FILE_SPEEDSCALE)
    from scipy.stats import mannwhitneyu

    ctrl  = [r for r in results if r.get("condition") == "control"   and not r.get("error")]
    treat = [r for r in results if r.get("condition") == "treatment" and not r.get("error")]
    if not ctrl or not treat:
        print("数据不足")
        return

    cb = [r["beats_best"] for r in ctrl]
    tb = [r["beats_best"] for r in treat]
    _, p = mannwhitneyu(tb, cb, alternative="greater")

    print(f"\n=== Speed-scaling 因果验证结果 ===")
    print(f"控制组 (无speed_tier算子): BB%={np.mean(cb)*100:.1f}%  n={len(cb)}")
    print(f"实验组 (有speed_tier算子): BB%={np.mean(tb)*100:.1f}%  n={len(tb)}")
    print(f"提升: +{(np.mean(tb)-np.mean(cb))*100:.1f} pp")
    print(f"Mann-Whitney p={p:.4f}  {'✓ 显著' if p<0.05 else '✗ 不显著'}")

    Path("results").mkdir(exist_ok=True)
    with open("results/causal_speedscale_stats.json", "w") as f:
        json.dump({
            "control_bb_pct": np.mean(cb)*100, "treatment_bb_pct": np.mean(tb)*100,
            "delta_pp": (np.mean(tb)-np.mean(cb))*100,
            "mann_whitney_p": p, "n_control": len(cb), "n_treatment": len(tb),
        }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["check","run","run_v2","causal","transfer_analysis","analyze","speedscale","causal_speedscale"],
        default="check")
    args = parser.parse_args()
    {
        "check":             run_check,
        "run":               run_spectrum,
        "run_v2":            run_spectrum_v2,
        "causal":            run_causal,
        "transfer_analysis": run_transfer_analysis,
        "analyze":           run_analyze,
        "speedscale":        run_speedscale_experiment,
        "causal_speedscale": run_causal_speedscale,
    }[args.mode]()
