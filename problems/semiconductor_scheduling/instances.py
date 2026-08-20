"""
semiconductor_scheduling/instances.py
Semiconductor fabrication scheduling (wafer fab, multi-stage recipes).

场景对应论文描述：
  baseline : 正常配方（5阶段流水线，标准处理时间）
  F_D1     : Processing time shift: photolithography 步骤时间变长（新光刻胶）
  F_D2     : WIP mix shift: urgent wafers 占比从10%→35%
  F_S1     : Recipe dependency change: 增加检验步骤（拓扑图结构变化）
  F_S2     : Machine capacity reduction: 一台关键机台故障（C→0.7C）
  F_Sem1   : Objective shift: makespan → weighted tardiness
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class FabTask:
    id: int
    task_type: str        # stage name: 'litho'|'etch'|'cvd'|'inspect'|'test'
    lot_id: int           # which wafer lot
    estimated_runtime: float  # hours
    runtime: float
    resource_need: float  # machine capacity units needed
    weight: float         # lot urgency (1=standard, 3=hot, 10=emergency)
    deadline: float       # relative due date (hours)
    stage: int            # pipeline stage index (0-based)

SCENES: Dict[str, dict] = {
    "baseline": {
        "description": "Normal fab operation: 5-stage pipeline, standard recipes",
        "total_resources": 100.0,
        "stages": ["litho", "etch", "cvd", "inspect", "test"],
        "runtime_params": {
            "litho":   ("lognormal", np.log(8),  0.3),
            "etch":    ("lognormal", np.log(4),  0.4),
            "cvd":     ("lognormal", np.log(6),  0.3),
            "inspect": ("lognormal", np.log(2),  0.5),
            "test":    ("lognormal", np.log(3),  0.3),
        },
        "resource_params": {
            "litho":   ("uniform", 15, 25),
            "etch":    ("uniform", 10, 18),
            "cvd":     ("uniform", 12, 20),
            "inspect": ("uniform",  5, 10),
            "test":    ("uniform",  8, 15),
        },
        "urgent_fraction": 0.10,
        "objective": "makespan",
        "change_type": None,
    },
    "F_D1": {
        "description": "Litho runtime shift: new photoresist causes 2x slower litho steps",
        "total_resources": 100.0,
        "stages": ["litho", "etch", "cvd", "inspect", "test"],
        "runtime_params": {
            "litho":   ("lognormal", np.log(16), 0.5),  # doubled + higher variance
            "etch":    ("lognormal", np.log(4),  0.4),
            "cvd":     ("lognormal", np.log(6),  0.3),
            "inspect": ("lognormal", np.log(2),  0.5),
            "test":    ("lognormal", np.log(3),  0.3),
        },
        "resource_params": {
            "litho":   ("uniform", 15, 25),
            "etch":    ("uniform", 10, 18),
            "cvd":     ("uniform", 12, 20),
            "inspect": ("uniform",  5, 10),
            "test":    ("uniform",  8, 15),
        },
        "urgent_fraction": 0.10,
        "objective": "makespan",
        "change_type": "distributional",
    },
    "F_D2": {
        "description": "WIP urgency shift: hot lots increase from 10% to 35%",
        "total_resources": 100.0,
        "stages": ["litho", "etch", "cvd", "inspect", "test"],
        "runtime_params": {
            "litho":   ("lognormal", np.log(8),  0.3),
            "etch":    ("lognormal", np.log(4),  0.4),
            "cvd":     ("lognormal", np.log(6),  0.3),
            "inspect": ("lognormal", np.log(2),  0.5),
            "test":    ("lognormal", np.log(3),  0.3),
        },
        "resource_params": {
            "litho":   ("uniform", 15, 25),
            "etch":    ("uniform", 10, 18),
            "cvd":     ("uniform", 12, 20),
            "inspect": ("uniform",  5, 10),
            "test":    ("uniform",  8, 15),
        },
        "urgent_fraction": 0.35,   # increased
        "objective": "makespan",
        "change_type": "distributional",
    },
    "F_S1": {
        "description": "Recipe change: mandatory inline inspection step added after litho",
        "total_resources": 100.0,
        "stages": ["litho", "inline_inspect", "etch", "cvd", "inspect", "test"],
        "runtime_params": {
            "litho":          ("lognormal", np.log(8),  0.3),
            "inline_inspect": ("lognormal", np.log(1.5),0.4),  # new step
            "etch":           ("lognormal", np.log(4),  0.4),
            "cvd":            ("lognormal", np.log(6),  0.3),
            "inspect":        ("lognormal", np.log(2),  0.5),
            "test":           ("lognormal", np.log(3),  0.3),
        },
        "resource_params": {
            "litho":          ("uniform", 15, 25),
            "inline_inspect": ("uniform",  3,  8),
            "etch":           ("uniform", 10, 18),
            "cvd":            ("uniform", 12, 20),
            "inspect":        ("uniform",  5, 10),
            "test":           ("uniform",  8, 15),
        },
        "urgent_fraction": 0.10,
        "objective": "makespan",
        "change_type": "structural",
    },
    "F_S2": {
        "description": "Machine failure: litho capacity reduced 30% (C→0.7C)",
        "total_resources": 70.0,   # reduced
        "stages": ["litho", "etch", "cvd", "inspect", "test"],
        "runtime_params": {
            "litho":   ("lognormal", np.log(8),  0.3),
            "etch":    ("lognormal", np.log(4),  0.4),
            "cvd":     ("lognormal", np.log(6),  0.3),
            "inspect": ("lognormal", np.log(2),  0.5),
            "test":    ("lognormal", np.log(3),  0.3),
        },
        "resource_params": {
            "litho":   ("uniform", 15, 25),
            "etch":    ("uniform", 10, 18),
            "cvd":     ("uniform", 12, 20),
            "inspect": ("uniform",  5, 10),
            "test":    ("uniform",  8, 15),
        },
        "urgent_fraction": 0.10,
        "objective": "makespan",
        "change_type": "structural",
    },
    "F_Sem1": {
        "description": "Objective shift: minimize makespan → minimize weighted tardiness",
        "total_resources": 100.0,
        "stages": ["litho", "etch", "cvd", "inspect", "test"],
        "runtime_params": {
            "litho":   ("lognormal", np.log(8),  0.3),
            "etch":    ("lognormal", np.log(4),  0.4),
            "cvd":     ("lognormal", np.log(6),  0.3),
            "inspect": ("lognormal", np.log(2),  0.5),
            "test":    ("lognormal", np.log(3),  0.3),
        },
        "resource_params": {
            "litho":   ("uniform", 15, 25),
            "etch":    ("uniform", 10, 18),
            "cvd":     ("uniform", 12, 20),
            "inspect": ("uniform",  5, 10),
            "test":    ("uniform",  8, 15),
        },
        "urgent_fraction": 0.30,
        "objective": "weighted_tardiness",  # new objective
        "change_type": "semantic",
    },
}

WEIGHTED_COMPLETION_SCENES = {"F_Sem1"}

def _sample(spec, rng):
    name, *p = spec
    if name == "lognormal":
        return float(np.clip(rng.lognormal(p[0], p[1]), 0.1, 200.0))
    return float(rng.uniform(p[0], p[1]))

def generate_instance(scene_name: str, n_tasks: int, seed: int) -> dict:
    rng   = np.random.default_rng(seed)
    scene = SCENES[scene_name]
    stages = scene["stages"]
    n_stages = len(stages)

    # tasks_per_stage: distribute n_tasks roughly evenly
    tasks_per_stage = [n_tasks // n_stages] * n_stages
    for i in range(n_tasks % n_stages):
        tasks_per_stage[i] += 1

    tasks = []
    tid = 0
    for stage_idx, stage_name in enumerate(stages):
        for _ in range(tasks_per_stage[stage_idx]):
            rt_raw = _sample(scene["runtime_params"][stage_name], rng)
            res    = float(np.clip(_sample(scene["resource_params"][stage_name], rng),
                                   1.0, scene["total_resources"] * 0.5))
            urgent = rng.random() < scene["urgent_fraction"]
            weight = 10.0 if urgent else (3.0 if rng.random() < 0.2 else 1.0)
            dl     = rt_raw * rng.uniform(1.2, 3.0)

            tasks.append(FabTask(
                id=tid, task_type=stage_name, lot_id=tid // n_stages,
                estimated_runtime=round(rt_raw, 3),
                runtime=round(rt_raw * rng.uniform(0.9, 1.1), 3),
                resource_need=round(res, 2),
                weight=weight, deadline=round(dl, 2),
                stage=stage_idx,
            ))
            tid += 1

    return {
        "tasks": tasks,
        "total_resources": scene["total_resources"],
        "objective": scene["objective"],
    }

def generate_instance_set(scene_name: str, n_instances: int, n_tasks: int,
                          base_seed: int = 0) -> list:
    return [generate_instance(scene_name, n_tasks, base_seed + i)
            for i in range(n_instances)]
