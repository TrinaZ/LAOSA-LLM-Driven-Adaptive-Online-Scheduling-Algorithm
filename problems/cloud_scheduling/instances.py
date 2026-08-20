"""
cloud_scheduling/instances.py
Google Borg-style cloud cluster scheduling instances.

场景设计对应论文Table 2：
  baseline  : 正常云集群负载（混合batch/interactive）
  C_D1      : Processing time distribution shift (LN→heavy tail)
  C_D2      : Arrival rate surge (Poisson λ: 5→12)
  C_S1      : Capacity reduction: CPU 100→60 (machine failure)
  C_S2      : Resource dimension addition: 2D→3D (add memory pressure)
  C_Sem1    : Objective shift: makespan → weighted flow time (SLA tiers)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

# ─── Task ───────────────────────────────────────────────────────────────────

@dataclass
class CloudTask:
    id: int
    task_type: str          # 'batch' | 'interactive' | 'dag_job'
    estimated_runtime: float  # seconds (normalized)
    runtime: float            # actual (for simulation)
    cpu_need: float           # fraction of total CPU
    mem_need: float           # fraction of total MEM
    weight: float             # SLA priority (1=standard, 5=premium, 10=critical)
    deadline: float           # relative deadline
    priority_class: int       # 0=best-effort, 1=standard, 2=premium

    # alias for compatibility with generic framework
    @property
    def resource_need(self):
        return self.cpu_need   # primary resource dimension

# ─── Scene definitions ───────────────────────────────────────────────────────

SCENES: Dict[str, dict] = {
    "baseline": {
        "description": "Normal mixed cloud workload (batch 60%, interactive 40%)",
        "cpu_total": 100.0,
        "mem_total": 100.0,
        "task_types": ["batch", "interactive"],
        "type_weights": [0.60, 0.40],
        "runtime_dist": {"batch": ("lognormal", 2.0, 0.6),
                         "interactive": ("lognormal", 0.5, 0.4)},
        "cpu_dist":    {"batch": ("uniform", 0.08, 0.20),
                        "interactive": ("uniform", 0.02, 0.10)},
        "mem_dist":    {"batch": ("uniform", 0.05, 0.15),
                        "interactive": ("uniform", 0.01, 0.08)},
        "weight_dist": {"batch": [1, 5],       "probs_batch": [0.8, 0.2],
                        "interactive": [5, 10], "probs_interactive": [0.6, 0.4]},
        "objective": "makespan",
        "change_type": None,
    },
    "C_D1": {
        "description": "Processing time distribution shift: batch jobs become heavy-tail (FFPE batch surge)",
        "cpu_total": 100.0,
        "mem_total": 100.0,
        "task_types": ["batch", "interactive"],
        "type_weights": [0.60, 0.40],
        "runtime_dist": {"batch": ("lognormal", 2.8, 1.2),    # heavier tail
                         "interactive": ("lognormal", 0.5, 0.4)},
        "cpu_dist":    {"batch": ("uniform", 0.08, 0.20),
                        "interactive": ("uniform", 0.02, 0.10)},
        "mem_dist":    {"batch": ("uniform", 0.05, 0.15),
                        "interactive": ("uniform", 0.01, 0.08)},
        "weight_dist": {"batch": [1, 5],       "probs_batch": [0.8, 0.2],
                        "interactive": [5, 10], "probs_interactive": [0.6, 0.4]},
        "objective": "makespan",
        "change_type": "distributional",
    },
    "C_D2": {
        "description": "Task mix shift: interactive load surges (60% interactive, 40% batch)",
        "cpu_total": 100.0,
        "mem_total": 100.0,
        "task_types": ["batch", "interactive"],
        "type_weights": [0.40, 0.60],   # flipped
        "runtime_dist": {"batch": ("lognormal", 2.0, 0.6),
                         "interactive": ("lognormal", 0.5, 0.4)},
        "cpu_dist":    {"batch": ("uniform", 0.08, 0.20),
                        "interactive": ("uniform", 0.02, 0.10)},
        "mem_dist":    {"batch": ("uniform", 0.05, 0.15),
                        "interactive": ("uniform", 0.01, 0.08)},
        "weight_dist": {"batch": [1, 5],       "probs_batch": [0.8, 0.2],
                        "interactive": [5, 10], "probs_interactive": [0.6, 0.4]},
        "objective": "makespan",
        "change_type": "distributional",
    },
    "C_S1": {
        "description": "Capacity reduction: CPU 100→60 (machine failure, 40% resource loss)",
        "cpu_total": 60.0,   # reduced
        "mem_total": 100.0,
        "task_types": ["batch", "interactive"],
        "type_weights": [0.60, 0.40],
        "runtime_dist": {"batch": ("lognormal", 2.0, 0.6),
                         "interactive": ("lognormal", 0.5, 0.4)},
        "cpu_dist":    {"batch": ("uniform", 0.08, 0.20),
                        "interactive": ("uniform", 0.02, 0.10)},
        "mem_dist":    {"batch": ("uniform", 0.05, 0.15),
                        "interactive": ("uniform", 0.01, 0.08)},
        "weight_dist": {"batch": [1, 5],       "probs_batch": [0.8, 0.2],
                        "interactive": [5, 10], "probs_interactive": [0.6, 0.4]},
        "objective": "makespan",
        "change_type": "structural",
    },
    "C_S2": {
        "description": "New dag_job task type added (GPU-intensive, 25% of tasks)",
        "cpu_total": 100.0,
        "mem_total": 100.0,
        "task_types": ["batch", "interactive", "dag_job"],
        "type_weights": [0.45, 0.30, 0.25],
        "runtime_dist": {"batch": ("lognormal", 2.0, 0.6),
                         "interactive": ("lognormal", 0.5, 0.4),
                         "dag_job": ("lognormal", 3.5, 0.8)},  # long, resource-heavy
        "cpu_dist":    {"batch": ("uniform", 0.08, 0.20),
                        "interactive": ("uniform", 0.02, 0.10),
                        "dag_job": ("uniform", 0.25, 0.40)},   # high CPU
        "mem_dist":    {"batch": ("uniform", 0.05, 0.15),
                        "interactive": ("uniform", 0.01, 0.08),
                        "dag_job": ("uniform", 0.20, 0.35)},
        "weight_dist": {"batch": [1, 5],       "probs_batch": [0.8, 0.2],
                        "interactive": [5, 10], "probs_interactive": [0.6, 0.4],
                        "dag_job": [1, 5],      "probs_dag_job": [0.5, 0.5]},
        "objective": "makespan",
        "change_type": "structural",
    },
    "C_Sem1": {
        "description": "Objective shift: makespan → weighted completion time (SLA tiers activated)",
        "cpu_total": 100.0,
        "mem_total": 100.0,
        "task_types": ["batch", "interactive"],
        "type_weights": [0.60, 0.40],
        "runtime_dist": {"batch": ("lognormal", 2.0, 0.6),
                         "interactive": ("lognormal", 0.5, 0.4)},
        "cpu_dist":    {"batch": ("uniform", 0.08, 0.20),
                        "interactive": ("uniform", 0.02, 0.10)},
        "mem_dist":    {"batch": ("uniform", 0.05, 0.15),
                        "interactive": ("uniform", 0.01, 0.08)},
        "weight_dist": {"batch": [1, 5, 10],   "probs_batch": [0.5, 0.3, 0.2],  # more premium
                        "interactive": [5, 10], "probs_interactive": [0.4, 0.6]},
        "objective": "weighted_completion",   # <-- changed
        "change_type": "semantic",
    },
}

# ─── Instance generation ─────────────────────────────────────────────────────

def _sample_runtime(dist_spec, rng):
    name, *params = dist_spec
    if name == "lognormal":
        mu, sigma = params
        return float(np.clip(rng.lognormal(mu, sigma), 0.1, 200.0))
    elif name == "uniform":
        lo, hi = params
        return float(rng.uniform(lo, hi))
    return 1.0

def generate_instance(scene_name: str, n_tasks: int, seed: int) -> dict:
    rng   = np.random.default_rng(seed)
    scene = SCENES[scene_name]

    types    = scene["task_types"]
    tw       = scene["type_weights"]
    rt_dists = scene["runtime_dist"]
    cpu_d    = scene["cpu_dist"]
    mem_d    = scene["mem_dist"]
    wt_d     = scene["weight_dist"]

    tasks = []
    for i in range(n_tasks):
        ttype = rng.choice(types, p=tw)

        rt_raw = _sample_runtime(rt_dists[ttype], rng)
        cpu    = float(np.clip(_sample_runtime(cpu_d[ttype], rng),  0.01, 0.45))
        mem    = float(np.clip(_sample_runtime(mem_d[ttype], rng),  0.01, 0.40))

        w_vals = wt_d[ttype]
        w_probs= wt_d.get(f"probs_{ttype}", None)
        if w_probs is None:
            w_probs = [1.0 / len(w_vals)] * len(w_vals)
        weight = float(rng.choice(w_vals, p=w_probs))

        pc   = {1: 0, 5: 1, 10: 2}.get(int(weight), 0)
        dl   = rt_raw * rng.uniform(1.5, 4.0)

        tasks.append(CloudTask(
            id=i,
            task_type=ttype,
            estimated_runtime=round(rt_raw, 3),
            runtime=round(rt_raw * rng.uniform(0.85, 1.15), 3),
            cpu_need=round(cpu * scene["cpu_total"], 3),
            mem_need=round(mem * scene["mem_total"], 3),
            weight=weight,
            deadline=round(dl, 2),
            priority_class=pc,
        ))

    return {
        "tasks": tasks,
        "cpu_total": scene["cpu_total"],
        "mem_total": scene["mem_total"],
        "total_resources": scene["cpu_total"],   # primary for generic interface
        "objective": scene["objective"],
    }


def generate_instance_set(scene_name: str, n_instances: int, n_tasks: int,
                          base_seed: int = 0) -> List[dict]:
    return [generate_instance(scene_name, n_tasks, base_seed + i)
            for i in range(n_instances)]
