"""
cloud_scheduling/prob.py
Cloud cluster scheduling problem (Google Borg-style).
Implements same interface as GenomicScheduling for drop-in compatibility.
"""

import sys, types, warnings
import numpy as np
from .instances import generate_instance_set, SCENES, CloudTask

# ─── Fixed baseline heuristics ───────────────────────────────────────────────

BASELINES = {
    "SPT": """def priority(task):
    return -task.estimated_runtime
""",
    "LPT": """def priority(task):
    return task.estimated_runtime
""",
    "EDF": """def priority(task):
    return -task.deadline
""",
    "WSJF": """def priority(task):
    rt = max(task.estimated_runtime, 0.01)
    return task.weight / rt
""",
    "ResourceEfficient": """def priority(task):
    cpu = max(task.resource_need, 0.1)
    return task.estimated_runtime / cpu
""",
}

WEIGHTED_COMPLETION_SCENES = {"C_Sem1"}


class CloudScheduling:
    """
    Online cloud cluster scheduling.
    evaluate(code_string) -> float fitness (higher = better, normalized by random baseline)
    """

    def __init__(self, scene: str = "baseline", n_instances: int = 10,
                 n_tasks: int = 20, scene_context: str = ""):
        self.scene       = scene
        self.n_tasks     = n_tasks
        self.use_weighted = scene in WEIGHTED_COMPLETION_SCENES
        self.instances   = generate_instance_set(scene, n_instances, n_tasks)
        self.baseline_makespan = self._eval_baseline()

    def _eval_baseline(self) -> float:
        scores = []
        rng = np.random.default_rng(999)
        for inst in self.instances:
            score = self._simulate(
                inst["tasks"], inst["total_resources"],
                lambda t: float(rng.random()),
                weighted=self.use_weighted,
            )
            scores.append(score)
        return float(np.mean(scores))

    def evaluate(self, code_string: str):
        """EoH interface. Returns fitness (higher = better)."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = types.ModuleType("_cloud_heuristic")
                exec(code_string, mod.__dict__)
                sys.modules[mod.__name__] = mod
                fn = getattr(mod, "priority", None)
                if fn is None:
                    return None
                scores = [
                    self._simulate(inst["tasks"], inst["total_resources"],
                                   fn, weighted=self.use_weighted)
                    for inst in self.instances
                ]
            avg = float(np.mean(scores))
            return (self.baseline_makespan - avg) / self.baseline_makespan
        except Exception:
            return None

    def _simulate(self, tasks, total_cpu, priority_func, weighted=False):
        """
        Greedy list-scheduling on a single machine pool (CPU-constrained).
        Returns makespan (weighted=False) or weighted completion time (weighted=True).
        """
        try:
            sorted_tasks = sorted(tasks, key=lambda t: -priority_func(t))
        except Exception:
            sorted_tasks = list(tasks)

        remaining  = [(t, t.runtime) for t in sorted_tasks]
        curr_time  = 0.0
        completion = {}
        max_iter   = len(tasks) * 30

        it = 0
        while remaining and it < max_iter:
            it += 1
            running      = []
            used_cpu     = 0.0
            still_waiting = []

            for task, rem in remaining:
                if used_cpu + task.cpu_need <= total_cpu:
                    running.append((task, rem))
                    used_cpu += task.cpu_need
                else:
                    still_waiting.append((task, rem))

            if not running:
                t, rt = still_waiting.pop(0)
                running = [(t, rt)]

            step = min(rem for _, rem in running)
            curr_time += step

            new_rem = []
            for task, rem in running:
                if rem - step < 1e-6:
                    completion[task.id] = curr_time
                else:
                    new_rem.append((task, rem - step))
            remaining = new_rem + still_waiting

        # any stragglers
        for task, _ in remaining:
            curr_time += 1.0
            completion[task.id] = curr_time

        if not completion:
            return 0.0

        if weighted:
            wmap = {t.id: t.weight for t in tasks}
            return sum(wmap.get(tid, 1.0) * ct for tid, ct in completion.items())
        return max(completion.values())
