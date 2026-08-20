"""
prob.py — EoH-compatible problem class for genomic pipeline online scheduling.

修改记录：
  - _simulate() 新增 weighted_makespan 参数，S5 场景用 weighted completion time
  - evaluate() 根据 self.scene 自动选择评估指标
  - S5_objective_change 场景使用 sum(weight * completion_time) 作为目标
  - 其他场景保持原有 makespan 指标
"""
import numpy as np
import types
import sys
import warnings
from .prompts import GetPrompts
from .instances import generate_instance_set, Task, SCENES

# S5 场景使用加权完成时间作为评估指标
WEIGHTED_COMPLETION_SCENES = {"S5_objective_change"}


class GenomicScheduling:
    """
    Online scheduling of genomic pipeline tasks.

    EoH interface:
      - self.prompts: GetPrompts instance
      - self.evaluate(code_string) -> fitness (float, higher is better)

    评估指标：
      - 默认场景（S1-S4）：makespan（所有任务完成时刻）
      - S5_objective_change：weighted completion time（sum of weight × completion_time）
        这样才能检测到权重变化带来的性能退化
    """

    def __init__(self, scene="baseline", n_instances=10, n_tasks=20,
                 scene_context=""):
        self.scene = scene
        self.n_tasks = n_tasks
        self.use_weighted = scene in WEIGHTED_COMPLETION_SCENES
        self.instances = generate_instance_set(scene, n_instances, n_tasks)
        self.prompts = GetPrompts(scene_context)

        # Baseline：random scheduling
        self.baseline_makespan = self._eval_baseline()

    def _eval_baseline(self) -> float:
        """Random scheduling baseline for normalization."""
        scores = []
        for inst in self.instances:
            score = self._simulate(
                inst["tasks"], inst["total_resources"],
                lambda t: np.random.random(),
                weighted=self.use_weighted,
            )
            scores.append(score)
        return np.mean(scores)

    def _simulate(self, tasks, total_resources, priority_func,
                  weighted: bool = False) -> float:
        """
        Simulate online scheduling with given priority function.

        weighted=False → 返回 makespan（最大完成时刻，越小越好）
        weighted=True  → 返回 sum(task.weight * completion_time)（越小越好）
                         用于 S5 场景，反映紧急任务的优先级变化
        """
        try:
            sorted_tasks = sorted(tasks, key=lambda t: -priority_func(t))
        except Exception:
            sorted_tasks = list(tasks)

        remaining = [(t, t.runtime) for t in sorted_tasks]
        current_time = 0.0
        completion_times = {}
        max_iter = len(tasks) * 20

        itr = 0
        while remaining and itr < max_iter:
            itr += 1
            running = []
            used = 0.0
            still_waiting = []

            for task, rem_time in remaining:
                if used + task.resource_need <= total_resources:
                    running.append((task, rem_time))
                    used += task.resource_need
                else:
                    still_waiting.append((task, rem_time))

            if not running:
                t, rt = still_waiting.pop(0)
                running = [(t, rt)]

            min_time = min(rt for _, rt in running)
            current_time += min_time

            new_remaining = []
            for task, rem_time in running:
                if rem_time - min_time < 0.01:
                    completion_times[task.id] = current_time
                else:
                    new_remaining.append((task, rem_time - min_time))

            remaining = new_remaining + still_waiting

        for task, _ in remaining:
            current_time += 1.0
            completion_times[task.id] = current_time

        if not completion_times:
            return 0.0

        if weighted:
            # S5 场景：加权完成时间 = sum(weight × completion_time)
            # 需要从 tasks 中查找每个任务的权重
            task_weights = {t.id: t.weight for t in tasks}
            wct = sum(
                task_weights.get(tid, 1.0) * ct
                for tid, ct in completion_times.items()
            )
            return wct
        else:
            return max(completion_times.values())

    def evaluate(self, code_string: str) -> float:
        """
        EoH interface: evaluate a code string containing a 'priority' function.
        Returns fitness (higher = better, normalized by baseline).

        S5 场景：fitness = (baseline_wct - adapted_wct) / baseline_wct
        其他场景：fitness = (baseline_makespan - adapted_makespan) / baseline_makespan
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                module = types.ModuleType("heuristic_module")
                exec(code_string, module.__dict__)
                sys.modules[module.__name__] = module

                priority_func = getattr(module, "priority", None)
                if priority_func is None:
                    return None

                scores = []
                for inst in self.instances:
                    score = self._simulate(
                        inst["tasks"], inst["total_resources"],
                        priority_func,
                        weighted=self.use_weighted,
                    )
                    scores.append(score)

                avg_score = np.mean(scores)

                # Fitness：改善量（正值=改善，负值=退化）
                fitness = (self.baseline_makespan - avg_score) / self.baseline_makespan
                return fitness

        except Exception:
            return None


# ================================================================
# BASELINE ALGORITHMS
# ================================================================

BASELINES = {
    "SPT": """
def priority(task):
    return -task.estimated_runtime
""",
    "LPT": """
def priority(task):
    return task.estimated_runtime
""",
    "EDF": """
def priority(task):
    return -task.deadline
""",
    "WSJF": """
def priority(task):
    return task.weight / max(task.estimated_runtime, 0.01)
""",
    "ResourceEfficient": """
def priority(task):
    return -task.resource_need
""",
}


def evaluate_baselines(scene="baseline", n_instances=10, n_tasks=20):
    """Evaluate all baseline algorithms on a given scene."""
    prob = GenomicScheduling(scene, n_instances, n_tasks)
    results = {}
    for name, code in BASELINES.items():
        fitness = prob.evaluate(code)
        results[name] = {
            "fitness": fitness,
            "makespan": prob.baseline_makespan * (1 - fitness) if fitness else None,
        }
    return results, prob.baseline_makespan
