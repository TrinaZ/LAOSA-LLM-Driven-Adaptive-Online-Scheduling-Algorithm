"""
laosa.py — LLM-driven Adaptive Online Scheduling Algorithm (LAOSA)
Core orchestrator: detect scene change → diagnose → generate adapted algorithm → evaluate

This module provides:
  1. SceneChangeDetector: CUSUM-based performance degradation detection
  2. LLMAlgorithmAdapter: evolutionary search with up to MAX_LLM_CALLS LLM calls
  3. LAOSAExperiment: full experiment pipeline (baseline → change → adapt → evaluate)
"""
import numpy as np
import json
import time
import os
import types
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple

from problems.genomic_scheduling.prob import GenomicScheduling, BASELINES
from problems.genomic_scheduling.instances import SCENES


# ================================================================
# 0. ARCHIVE LOGGER
# ================================================================

class ArchiveLogger:
    """记录所有LLM生成的候选算法，供实验A（LAOSA-no-generation）使用。"""
    def __init__(self):
        self.entries = []

    def log(self, code: str, bottleneck_vec: list, score: float,
            scenario_id: str, change_type: str):
        self.entries.append({
            "code":           code,
            "bottleneck_vec": list(bottleneck_vec),
            "score":          float(score),
            "scenario_id":    scenario_id,
            "change_type":    change_type,
        })

    def save(self, path: str = "generated_archive.pkl"):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.entries, f)
        print(f"[ArchiveLogger] 已保存 {len(self.entries)} 条记录到 {path}")

    @staticmethod
    def load(path: str):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


_global_archive_logger = None

def set_archive_logger(logger):
    global _global_archive_logger
    _global_archive_logger = logger

def save_archive(path: str = "generated_archive.pkl"):
    if _global_archive_logger is not None:
        _global_archive_logger.save(path)
    else:
        print("[ArchiveLogger] 未启用。")


# ================================================================
# 1. SCENE CHANGE DETECTOR
# ================================================================

class SceneChangeDetector:
    """Detects performance degradation using CUSUM algorithm."""

    def __init__(self, warmup: int = 15, threshold: float = 3.0):
        self.warmup = warmup
        self.threshold = threshold
        self.history = []
        self.baseline_mean = None
        self.baseline_std = None
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

    def update(self, makespan: float) -> dict:
        self.history.append(makespan)
        n = len(self.history)

        if n < self.warmup:
            return {"detected": False, "n": n, "status": "warmup"}

        if self.baseline_mean is None:
            self.baseline_mean = np.mean(self.history[:self.warmup])
            self.baseline_std = max(np.std(self.history[:self.warmup]), 0.01)

        z = (makespan - self.baseline_mean) / self.baseline_std
        self.cusum_pos = max(0, self.cusum_pos + z - 0.5)
        self.cusum_neg = max(0, self.cusum_neg - z - 0.5)

        recent = self.history[-min(10, n):]
        degradation = (np.mean(recent) / self.baseline_mean - 1) * 100

        return {
            "detected":        self.cusum_pos > self.threshold or self.cusum_neg > self.threshold,
            "cusum_pos":       round(self.cusum_pos, 2),
            "cusum_neg":       round(self.cusum_neg, 2),
            "degradation_pct": round(degradation, 1),
            "baseline_mean":   round(self.baseline_mean, 2),
            "recent_mean":     round(np.mean(recent), 2),
            "n":               n,
        }


# ================================================================
# 2. LLM ALGORITHM ADAPTER  —  含 Evolutionary Search
# ================================================================

MAX_LLM_CALLS   = 20
N_CANDIDATES    = 5
PATIENCE        = 3
IMPROVEMENT_TAU = 0.02


class LLMAlgorithmAdapter:
    """
    Generates adapted scheduling algorithms via LLM-guided evolutionary search.

    搜索流程（论文 Section 4.4）：
      1. 以 incumbent（当前最优）为起点
      2. 每轮向 LLM 请求 N_CANDIDATES 个变体
      3. 在 prob_after 上验证，选出最优者
      4. 若改善超过 tau，更新 incumbent
      5. 连续 PATIENCE 轮无改善或达到 MAX_LLM_CALLS 时停止
    """

    def __init__(self, use_real_api: bool = False):
        self.use_real_api = use_real_api
        self._archive_logger = None

    def build_prompt(self, current_code: str, perf_data: dict,
                     scene_diagnosis: str, past_attempts=None, iteration: int = 1) -> str:
        past_section = ""
        if past_attempts:
            lines = ["## Previous Attempts (avoid repeating these approaches)"]
            for i, (code, ms) in enumerate(past_attempts[-3:], 1):
                preview = code.strip().split("\n")[0]
                lines.append(f"  Attempt {i}: {preview}  →  makespan={ms:.2f}")
            past_section = "\n".join(lines) + "\n\n"

        recent_mean = perf_data.get("recent_mean", "N/A")
        recent_str = f"{recent_mean:.2f}" if isinstance(recent_mean, float) else str(recent_mean)

        return f"""You are an expert in online scheduling algorithm design.
A genomic pipeline scheduling system detected a SCENE CHANGE.
The current scheduling algorithm's performance has degraded.
This is iteration {iteration} of an evolutionary search (max {MAX_LLM_CALLS} LLM calls total).

## Current Best Algorithm (incumbent, makespan={recent_str})
```python
{current_code}
```

## Performance Data
- Baseline average makespan: {perf_data.get('baseline_mean', 'N/A')}
- Current (degraded) makespan: {recent_str}
- Performance degradation: {perf_data.get('degradation_pct', 0):.1f}%

## Diagnosed Scene Change
{scene_diagnosis}

{past_section}## Your Task
Generate EXACTLY {N_CANDIDATES} DISTINCT improved priority functions.
Each must be meaningfully different from the current best and from each other.

Format your response as:
### Variant 1
```python
def priority(task):
    ...
```
### Variant 2
```python
def priority(task):
    ...
```
(continue for all {N_CANDIDATES} variants)

The function signature MUST be:
```python
def priority(task):
    # task.estimated_runtime (float): estimated remaining time
    # task.resource_need (float): CPU cores needed
    # task.weight (float): urgency weight
    # task.deadline (float): deadline
    # task.task_type (str): 'alignment', 'variant_calling', or 'sv_calling'
    # Returns: float (higher = higher priority = scheduled first)
```

Requirements:
- Return ONLY the Python functions, no explanation outside code blocks
- Use only basic Python math (no imports needed)
- Handle edge cases (zero/negative values)
- Be creative: each variant should try a fundamentally different approach
- Consider the specific scene change in your design
"""

    def _call_llm_once(self, prompt: str):
        try:
            from llm_adapter import call_llm, detect_provider
            provider, _ = detect_provider()
            if provider != "simulated":
                return call_llm(prompt, max_tokens=2048)
        except ImportError:
            pass
        return None

    def _parse_variants(self, raw_text: str):
        import re
        variants = []
        blocks = re.findall(r"```python\s*(def priority[\s\S]*?)```", raw_text)
        for block in blocks:
            code = block.strip()
            if code.startswith("def priority"):
                variants.append(code)

        if not variants:
            parts = re.split(r"(?=^def priority)", raw_text, flags=re.MULTILINE)
            for part in parts:
                part = part.strip()
                if part.startswith("def priority"):
                    code_lines = []
                    for line in part.split("\n"):
                        if line and not line[0].isspace() and not line.startswith("def") and code_lines:
                            break
                        code_lines.append(line)
                    variants.append("\n".join(code_lines).strip())

        seen = set()
        unique = []
        for v in variants:
            key = v.replace(" ", "").replace("\n", "")
            if key not in seen:
                seen.add(key)
                unique.append(v)

        return unique[:N_CANDIDATES]

    def compile(self, code: str) -> bool:
        try:
            full = f"import math\n{code}"
            module = types.ModuleType("test_module")
            exec(full, module.__dict__)
            func = getattr(module, "priority", None)
            if func is None:
                return False

            @dataclass
            class _DummyTask:
                estimated_runtime: float = 10.0
                resource_need:     float = 8.0
                weight:            float = 1.0
                deadline:          float = 20.0
                task_type:         str   = "alignment"

            result = func(_DummyTask())
            return isinstance(result, (int, float)) and not np.isnan(result)
        except Exception:
            return False

    def _evaluate_code(self, code: str, prob):
        eval_code = f"import math\n{code}"
        fitness = prob.evaluate(eval_code)
        if fitness is None:
            fitness = prob.evaluate(code)
        if fitness is None:
            return None
        return prob.baseline_makespan * (1 - fitness)

    def generate(self, current_code: str, perf_data: dict,
                 scene_diagnosis: str, prob_after=None):
        if prob_after is None:
            return self._generate_single(current_code, perf_data, scene_diagnosis)

        incumbent_code     = current_code
        incumbent_makespan = perf_data.get("recent_mean", float("inf"))
        degraded_makespan  = incumbent_makespan
        total_llm_calls    = 0
        no_improve_count   = 0
        past_attempts      = []

        print(f"    [Evo] incumbent makespan={incumbent_makespan:.2f}, "
              f"max_calls={MAX_LLM_CALLS}, patience={PATIENCE}")

        for iteration in range(1, MAX_LLM_CALLS // N_CANDIDATES + 2):
            if total_llm_calls >= MAX_LLM_CALLS:
                print(f"    [Evo] 达到最大 LLM 调用次数 ({MAX_LLM_CALLS})，停止。")
                break
            if no_improve_count >= PATIENCE:
                print(f"    [Evo] 连续 {PATIENCE} 轮无改善，停止。")
                break

            prompt = self.build_prompt(
                incumbent_code, perf_data, scene_diagnosis,
                past_attempts=past_attempts, iteration=iteration,
            )

            print(f"    [Evo] 第 {iteration} 轮，已用 {total_llm_calls} 次调用...")
            raw = self._call_llm_once(prompt)
            total_llm_calls += 1

            if raw is None:
                print(f"    [Evo] LLM 调用失败，使用 simulated fallback")
                candidates = [self._simulate_response(scene_diagnosis)]
            else:
                candidates = self._parse_variants(raw)
                if not candidates:
                    try:
                        from llm_adapter import _extract_code
                        single = _extract_code(raw)
                        candidates = [single] if single else []
                    except Exception:
                        candidates = []

            if not candidates:
                print(f"    [Evo] 无法解析候选算法，跳过本轮")
                no_improve_count += 1
                continue

            best_candidate_code = None
            best_candidate_ms   = incumbent_makespan
            improved_this_round = False

            for cand_code in candidates:
                if not self.compile(cand_code):
                    continue

                cand_ms = self._evaluate_code(cand_code, prob_after)
                if cand_ms is None:
                    continue

                past_attempts.append((cand_code, cand_ms))

                logger = self._archive_logger or _global_archive_logger
                if logger is not None:
                    bv = [
                        perf_data.get("cpu_util",        0.5),
                        perf_data.get("mem_util",        0.5),
                        perf_data.get("io_util",         0.3),
                        perf_data.get("queue_var",       1.0),
                        abs(perf_data.get("degradation_pct", 0)) / 100.0,
                    ]
                    logger.log(
                        code=cand_code, bottleneck_vec=bv, score=cand_ms,
                        scenario_id=perf_data.get("scene_id", "unknown"),
                        change_type=perf_data.get("change_type", "unknown"),
                    )

                if cand_ms < best_candidate_ms:
                    best_candidate_ms   = cand_ms
                    best_candidate_code = cand_code
                    improved_this_round = True

            if improved_this_round and best_candidate_code is not None:
                rel_improvement = (incumbent_makespan - best_candidate_ms) / incumbent_makespan
                print(f"    [Evo] 改善 {rel_improvement*100:.1f}%: "
                      f"{incumbent_makespan:.2f} → {best_candidate_ms:.2f}")
                incumbent_code     = best_candidate_code
                incumbent_makespan = best_candidate_ms
                no_improve_count   = 0
            else:
                no_improve_count += 1
                print(f"    [Evo] 本轮无改善 (patience={no_improve_count}/{PATIENCE})")

        print(f"    [Evo] 搜索结束：共 {total_llm_calls} 次 LLM 调用，"
              f"最终 makespan={incumbent_makespan:.2f} "
              f"(改善 {(degraded_makespan - incumbent_makespan)/max(degraded_makespan,0.001)*100:.1f}%)")

        final_improvement = (degraded_makespan - incumbent_makespan) / max(degraded_makespan, 0.001)
        if final_improvement >= IMPROVEMENT_TAU and incumbent_code != current_code:
            return incumbent_code, total_llm_calls
        elif incumbent_code == current_code:
            print(f"    [Evo] 进化未能改善，尝试 simulated fallback")
            return self._simulate_response(scene_diagnosis), total_llm_calls
        else:
            return incumbent_code, total_llm_calls

    def _generate_single(self, current_code: str, perf_data: dict, scene_diagnosis: str):
        prompt = self.build_prompt(current_code, perf_data, scene_diagnosis, iteration=1)
        result = None
        try:
            from llm_adapter import call_llm, detect_provider
            provider, _ = detect_provider()
            if provider != "simulated":
                print(f"    Using LLM provider: {provider} (single-call mode)")
                result = call_llm(prompt, max_tokens=1500)
        except ImportError:
            pass

        if result is None:
            result = self._simulate_response(scene_diagnosis)

        if result:
            logger = self._archive_logger or _global_archive_logger
            if logger is not None:
                bv = [
                    perf_data.get("cpu_util",        0.5),
                    perf_data.get("mem_util",        0.5),
                    perf_data.get("io_util",         0.3),
                    perf_data.get("queue_var",       1.0),
                    abs(perf_data.get("degradation_pct", 0)) / 100.0,
                ]
                logger.log(result, bv, perf_data.get("recent_mean", 0.0),
                           perf_data.get("scene_id", "unknown"),
                           perf_data.get("change_type", "unknown"))
        return result, 1

    def _simulate_response(self, diagnosis: str) -> str:
        d = diagnosis.lower()
        if "heavy" in d or "lognormal" in d or "ffpe" in d or "runtime" in d:
            return """def priority(task):
    speed = 1.0 / max(task.estimated_runtime, 0.1)
    res_eff = 1.0 / max(task.resource_need, 1.0)
    urgency = task.weight / max(task.deadline - task.estimated_runtime, 0.1)
    return speed * 2.0 + res_eff * 1.5 + urgency * 0.5"""
        elif "resource" in d and ("drop" in d or "60" in d):
            return """def priority(task):
    small_bonus = 1.0 / max(task.resource_need, 1.0)
    speed = 1.0 / max(task.estimated_runtime, 0.1)
    return small_bonus * 3.0 + speed * 1.0"""
        elif "mix" in d or "proportion" in d or "panel" in d:
            return """def priority(task):
    type_factor = 1.8 if task.task_type == "variant_calling" else 1.0
    speed = 1.0 / max(task.estimated_runtime, 0.1)
    return type_factor * speed * task.weight"""
        elif "sv_calling" in d or "novel" in d:
            return """def priority(task):
    if task.task_type == "sv_calling":
        return -task.estimated_runtime * 0.1
    speed = 1.0 / max(task.estimated_runtime, 0.1)
    res_eff = 1.0 / max(task.resource_need, 1.0)
    return speed * 2.0 + res_eff * 1.0"""
        elif "objective" in d or "weight" in d or "urgent" in d:
            return """def priority(task):
    urgency = task.weight / max(task.estimated_runtime, 0.1)
    deadline_pressure = 1.0 / max(task.deadline - task.estimated_runtime, 0.01)
    return urgency * 2.0 + deadline_pressure * 1.0"""
        else:
            return """def priority(task):
    speed = 1.0 / max(task.estimated_runtime, 0.1)
    res_eff = 1.0 / max(task.resource_need, 1.0)
    urgency = task.weight / max(task.deadline, 0.1)
    return speed * 0.4 + res_eff * 0.3 + urgency * 0.3"""


@dataclass
class DummyTask:
    estimated_runtime: float = 10.0
    resource_need:     float = 8.0
    weight:            float = 1.0
    deadline:          float = 20.0
    task_type:         str   = "alignment"


# ================================================================
# 3. LAOSA EXPERIMENT
# ================================================================

class LAOSAExperiment:
    """Full experiment: baseline → scene change → detect → adapt → evaluate."""

    def __init__(self, baseline_algo: str = "SPT", use_real_api: bool = False):
        self.baseline_algo = baseline_algo
        self.baseline_code = BASELINES[baseline_algo]
        self.adapter = LLMAlgorithmAdapter(use_real_api)

    def run(self, scene_before: str, scene_after: str,
            change_description: str,
            n_eval_baseline: int = 30, n_eval_change: int = 30,
            use_evolutionary_search: bool = True) -> dict:
        """
        Run complete LAOSA experiment.
        use_evolutionary_search=True  → 迭代进化搜索（推荐，与论文一致）
        use_evolutionary_search=False → 单次 LLM 调用（向后兼容）
        """

        print(f"\n{'='*60}")
        print(f"  LAOSA Experiment {'[Evolutionary]' if use_evolutionary_search else '[Single-call]'}")
        print(f"  Algo: {self.baseline_algo}")
        print(f"  Scene: {scene_before} → {scene_after}")
        print(f"{'='*60}")

        # Phase 1: Baseline
        print("\n  [Phase 1] Measuring baseline performance...")
        prob_before = GenomicScheduling(scene_before, n_instances=10, n_tasks=20)
        baseline_fitness  = prob_before.evaluate(self.baseline_code)
        baseline_makespan = (prob_before.baseline_makespan * (1 - baseline_fitness)
                             if baseline_fitness else None)

        if baseline_makespan is None:
            print("    FAILED: baseline algorithm doesn't work")
            return {"error": "baseline_failed"}

        print(f"    Baseline makespan: {baseline_makespan:.2f}")

        # Phase 2: Apply scene change
        print(f"\n  [Phase 2] Applying scene change: {scene_after}...")
        prob_after = GenomicScheduling(scene_after, n_instances=10, n_tasks=20)
        degraded_fitness  = prob_after.evaluate(self.baseline_code)
        degraded_makespan = (prob_after.baseline_makespan * (1 - degraded_fitness)
                             if degraded_fitness else None)

        if degraded_makespan is None:
            print("    FAILED: evaluation after change failed")
            return {"error": "eval_failed"}

        degradation = (degraded_makespan / baseline_makespan - 1) * 100
        print(f"    Degraded makespan: {degraded_makespan:.2f} (degradation: {degradation:.1f}%)")

        # Phase 3: CUSUM Detection
        print(f"\n  [Phase 3] Running change detection...")
        detector = SceneChangeDetector(warmup=10, threshold=2.5)
        detection_step = None

        for i in range(15):
            ms = baseline_makespan + np.random.normal(0, baseline_makespan * 0.05)
            detector.update(ms)

        for i in range(20):
            ms = degraded_makespan + np.random.normal(0, degraded_makespan * 0.1)
            result = detector.update(ms)
            if result["detected"] and detection_step is None:
                detection_step = i + 1
                print(f"    Change DETECTED at step {detection_step}! CUSUM={result['cusum_pos']:.1f}")

        if detection_step is None:
            detection_step = 20
            print("    Change not detected within 20 steps (using full window)")

        perf_data = {
            "baseline_mean":   baseline_makespan,
            "recent_mean":     degraded_makespan,
            "degradation_pct": degradation,
            "scene_id":        scene_after,
            "change_type":     self._infer_change_type(scene_after),
        }

        # Phase 4: LLM Evolutionary Search
        print(f"\n  [Phase 4] LLM {'evolutionary search' if use_evolutionary_search else 'generating adapted algorithm'}...")
        t0 = time.time()

        adapted_code, llm_calls = self.adapter.generate(
            self.baseline_code, perf_data, change_description,
            prob_after=prob_after if use_evolutionary_search else None,
        )
        gen_time = time.time() - t0

        if adapted_code is None:
            print("    FAILED: LLM returned nothing")
            return {"error": "llm_failed"}

        eval_code = f"import math\n{adapted_code}"
        print(f"    Total generation time: {gen_time:.1f}s")
        print(f"    Final code preview: {adapted_code.strip()[:80]}...")

        # Phase 5: Final Evaluation
        print(f"\n  [Phase 5] Evaluating adapted algorithm...")
        adapted_fitness = prob_after.evaluate(eval_code)
        if adapted_fitness is None:
            adapted_fitness = prob_after.evaluate(adapted_code)

        if adapted_fitness is None:
            print("    FAILED: adapted algorithm doesn't run")
            return {"error": "adapted_eval_failed", "adapted_code": adapted_code}

        adapted_makespan = prob_after.baseline_makespan * (1 - adapted_fitness)
        improvement = (1 - adapted_makespan / degraded_makespan) * 100
        recovery    = ((degraded_makespan - adapted_makespan) /
                        max(degraded_makespan - baseline_makespan, 0.01)) * 100

        print(f"    Adapted makespan: {adapted_makespan:.2f}")
        print(f"    Improvement over degraded: {improvement:.1f}%")
        print(f"    Recovery toward baseline: {recovery:.1f}%")

        # Comparison with all baselines
        print(f"\n  [Comparison] All baselines on new scene:")
        best_baseline_name = None
        best_baseline_ms   = float("inf")
        for name, code in BASELINES.items():
            fit = prob_after.evaluate(code)
            if fit is not None:
                ms = prob_after.baseline_makespan * (1 - fit)
                if ms < best_baseline_ms:
                    best_baseline_ms   = ms
                    best_baseline_name = name
                marker = " ← current" if name == self.baseline_algo else ""
                print(f"    {name:20s}: makespan={ms:.2f}{marker}")

        print(f"    {'LAOSA (adapted)':20s}: makespan={adapted_makespan:.2f} ← generated")
        beats_best = adapted_makespan < best_baseline_ms

        return {
            "scene_before":        scene_before,
            "scene_after":         scene_after,
            "baseline_algo":       self.baseline_algo,
            "baseline_makespan":   round(baseline_makespan, 2),
            "degraded_makespan":   round(degraded_makespan, 2),
            "adapted_makespan":    round(adapted_makespan, 2),
            "degradation_pct":     round(degradation, 1),
            "improvement_pct":     round(improvement, 1),
            "recovery_pct":        round(recovery, 1),
            "detection_delay":     detection_step,
            "generation_time_sec": round(gen_time, 2),
            "llm_calls":           llm_calls,
            "best_fixed_baseline": best_baseline_name,
            "best_fixed_makespan": round(best_baseline_ms, 2),
            "beats_best_baseline": beats_best,
            "adapted_code":        adapted_code.strip(),
        }

    def run_with_probs(self, scene_before: str, scene_after: str,
                       change_description: str,
                       prob_before, prob_after,
                       baseline_ms: float, degraded_ms: float,
                       degradation_pct: float,
                       use_evolutionary_search: bool = True) -> dict:
        """
        与 run() 相同，但复用外部已建好的 prob_before/prob_after 实例，
        跳过 Phase 1/2 的重复计算，直接从 Phase 3 开始。
        供 run_experiments.py 在 NAV check 后调用，避免重复建实例。
        """
        print(f"\n{'='*60}")
        print(f"  LAOSA Experiment {'[Evolutionary]' if use_evolutionary_search else '[Single-call]'}")
        print(f"  Algo: {self.baseline_algo}")
        print(f"  Scene: {scene_before} → {scene_after}")
        print(f"{'='*60}")
        print(f"\n  [Phase 1+2 skipped] Using pre-computed values:")
        print(f"    Baseline makespan: {baseline_ms:.2f}")
        print(f"    Degraded makespan: {degraded_ms:.2f} (degradation: {degradation_pct:.1f}%)")

        # Phase 3: CUSUM
        print(f"\n  [Phase 3] Running change detection...")
        detector = SceneChangeDetector(warmup=10, threshold=2.5)
        detection_step = None
        for i in range(15):
            ms = baseline_ms + np.random.normal(0, baseline_ms * 0.05)
            detector.update(ms)
        for i in range(20):
            ms = degraded_ms + np.random.normal(0, degraded_ms * 0.1)
            result = detector.update(ms)
            if result["detected"] and detection_step is None:
                detection_step = i + 1
                print(f"    Change DETECTED at step {detection_step}! CUSUM={result['cusum_pos']:.1f}")
        if detection_step is None:
            detection_step = 20
            print("    Change not detected within 20 steps")

        perf_data = {
            "baseline_mean": baseline_ms, "recent_mean": degraded_ms,
            "degradation_pct": degradation_pct,
            "scene_id": scene_after, "change_type": self._infer_change_type(scene_after),
        }

        # Phase 4: LLM search
        print(f"\n  [Phase 4] LLM {'evolutionary search' if use_evolutionary_search else 'single-call'}...")
        t0 = time.time()
        adapted_code, llm_calls = self.adapter.generate(
            self.baseline_code, perf_data, change_description,
            prob_after=prob_after if use_evolutionary_search else None,
        )
        gen_time = time.time() - t0
        if adapted_code is None:
            return {"error": "llm_failed"}
        print(f"    Total generation time: {gen_time:.1f}s")
        print(f"    Final code preview: {adapted_code.strip()[:80]}...")

        # Phase 5: Evaluate
        print(f"\n  [Phase 5] Evaluating adapted algorithm...")
        eval_code = f"import math\n{adapted_code}"
        adapted_fitness = prob_after.evaluate(eval_code) or prob_after.evaluate(adapted_code)
        if adapted_fitness is None:
            return {"error": "adapted_eval_failed", "adapted_code": adapted_code}

        adapted_ms = prob_after.baseline_makespan * (1 - adapted_fitness)
        improvement = (1 - adapted_ms / degraded_ms) * 100
        recovery    = ((degraded_ms - adapted_ms) / max(degraded_ms - baseline_ms, 0.01)) * 100
        print(f"    Adapted makespan: {adapted_ms:.2f}")
        print(f"    Improvement over degraded: {improvement:.1f}%")
        print(f"    Recovery toward baseline: {recovery:.1f}%")

        # Comparison
        print(f"\n  [Comparison] All baselines on new scene:")
        best_fixed_name = None; best_fixed_ms = float("inf")
        for name, code in BASELINES.items():
            fit = prob_after.evaluate(code)
            if fit is not None:
                ms = prob_after.baseline_makespan * (1 - fit)
                if ms < best_fixed_ms: best_fixed_ms = ms; best_fixed_name = name
                marker = " ← current" if name == self.baseline_algo else ""
                print(f"    {name:20s}: makespan={ms:.2f}{marker}")
        print(f"    {'LAOSA (adapted)':20s}: makespan={adapted_ms:.2f} ← generated")

        return {
            "scene_before": scene_before, "scene_after": scene_after,
            "baseline_algo": self.baseline_algo,
            "baseline_makespan": round(baseline_ms, 2),
            "degraded_makespan": round(degraded_ms, 2),
            "adapted_makespan":  round(adapted_ms, 2),
            "degradation_pct":   round(degradation_pct, 1),
            "improvement_pct":   round(improvement, 1),
            "recovery_pct":      round(recovery, 1),
            "detection_delay":   detection_step,
            "generation_time_sec": round(gen_time, 2),
            "llm_calls":         llm_calls,
            "best_fixed_baseline": best_fixed_name,
            "best_fixed_makespan": round(best_fixed_ms, 2),
            "beats_best_baseline": adapted_ms < best_fixed_ms,
            "adapted_code":      adapted_code.strip(),
        }

    @staticmethod
    def _infer_change_type(scene_after: str) -> str:
        s = scene_after.lower()
        if "runtime" in s or "dist" in s or "var" in s or "corr" in s or "mix" in s:
            return "distributional"
        elif ("topology" in s or "dag" in s or "struct" in s or "chain" in s
              or "resource" in s or "novel" in s):
            return "structural"
        elif "objective" in s or "weight" in s or "fairness" in s or "pnorm" in s:
            return "semantic"
        else:
            return "distributional"
