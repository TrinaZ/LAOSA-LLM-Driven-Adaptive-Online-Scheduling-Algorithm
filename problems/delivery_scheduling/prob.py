"""delivery_scheduling/prob.py"""
import sys, types, warnings
import numpy as np
from .instances import generate_instance_set, SCENES, WEIGHTED_COMPLETION_SCENES

BASELINES = {
    "SPT":  "def priority(task): return -task.estimated_runtime\n",
    "LPT":  "def priority(task): return task.estimated_runtime\n",
    "EDF":  "def priority(task): return -task.deadline\n",
    "WSJF": "def priority(task): return task.weight / max(task.estimated_runtime, 0.01)\n",
    "ResourceEfficient": "def priority(task): return task.estimated_runtime / max(task.resource_need, 0.1)\n",
}


class DeliveryScheduling:
    def __init__(self, scene="baseline", n_instances=10, n_tasks=20, scene_context=""):
        self.scene        = scene
        self.n_tasks      = n_tasks
        self.use_weighted = scene in WEIGHTED_COMPLETION_SCENES
        self.instances    = generate_instance_set(scene, n_instances, n_tasks)
        self.baseline_makespan = self._eval_baseline()

    def _eval_baseline(self):
        rng = np.random.default_rng(555)
        return float(np.mean([
            self._simulate(inst["tasks"], inst["total_resources"],
                           lambda t: float(rng.random()),
                           weighted=self.use_weighted)
            for inst in self.instances
        ]))

    def evaluate(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = types.ModuleType("_delivery_heuristic")
                exec(code_string, mod.__dict__)
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

    def _simulate(self, tasks, total_resources, priority_func, weighted=False):
        try:
            sorted_tasks = sorted(tasks, key=lambda t: -priority_func(t))
        except Exception:
            sorted_tasks = list(tasks)

        remaining = [(t, t.runtime) for t in sorted_tasks]
        curr_time = 0.0
        completion = {}

        for _ in range(len(tasks) * 30):
            if not remaining:
                break
            running, used, waiting = [], 0.0, []
            for task, rem in remaining:
                if used + task.resource_need <= total_resources:
                    running.append((task, rem)); used += task.resource_need
                else:
                    waiting.append((task, rem))
            if not running:
                t, rt = waiting.pop(0); running = [(t, rt)]

            step = min(r for _, r in running)
            curr_time += step
            new_rem = []
            for task, rem in running:
                if rem - step < 1e-6:
                    completion[task.id] = curr_time
                else:
                    new_rem.append((task, rem - step))
            remaining = new_rem + waiting

        for task, _ in remaining:
            curr_time += 1.0; completion[task.id] = curr_time

        if not completion:
            return 0.0

        if weighted:
            dmap = {t.id: t.deadline for t in tasks}
            wmap = {t.id: t.weight   for t in tasks}
            # weighted lateness: w * max(0, C_j - d_j)
            return sum(wmap[tid] * max(0.0, ct - dmap[tid])
                       for tid, ct in completion.items())
        return max(completion.values())
