"""
delivery_scheduling/instances.py
Last-mile delivery scheduling (vehicle routing with time windows, simplified).

对应论文：5–8 objective modifications per month, cost vs latency trade-offs.

场景：
  baseline : 标准配送（最小化makespan/最大完成时间）
  L_D1     : Delivery time distribution shift (高峰期配送时间变长)
  L_D2     : Order mix shift (大订单占比增加，urgent小订单减少)
  L_S1     : Fleet capacity reduction (一辆车故障，capacity减少)
  L_S2     : Geographic zone expansion (新增远郊区域，配送距离增加)
  L_Sem1   : Objective shift: makespan → weighted lateness (SLA penalty mode)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class DeliveryTask:
    id: int
    task_type: str        # 'standard'|'express'|'bulky'|'remote'
    estimated_runtime: float  # delivery time (minutes)
    runtime: float            # actual
    resource_need: float      # vehicle load units (capacity fraction * 100)
    weight: float             # SLA weight (1=standard, 3=express, 10=critical)
    deadline: float           # time window end (minutes from depot departure)
    zone: str                 # 'urban'|'suburban'|'rural'

SCENES: Dict[str, dict] = {
    "baseline": {
        "description": "Standard mixed delivery: urban/suburban, normal demand",
        "total_resources": 100.0,   # vehicle capacity units
        "task_types": ["standard", "express"],
        "type_weights": [0.70, 0.30],
        "runtime_params": {
            "standard": ("lognormal", np.log(25), 0.4),
            "express":  ("lognormal", np.log(15), 0.3),
        },
        "resource_params": {
            "standard": ("uniform", 8, 20),
            "express":  ("uniform", 3, 10),
        },
        "weight_map":  {"standard": 1.0, "express": 3.0},
        "deadline_multiplier": {"standard": 3.0, "express": 1.8},
        "objective": "makespan",
        "change_type": None,
    },
    "L_D1": {
        "description": "Peak-hour traffic surge: all delivery times 1.5x longer",
        "total_resources": 100.0,
        "task_types": ["standard", "express"],
        "type_weights": [0.70, 0.30],
        "runtime_params": {
            "standard": ("lognormal", np.log(38), 0.5),  # longer
            "express":  ("lognormal", np.log(22), 0.4),
        },
        "resource_params": {
            "standard": ("uniform", 8, 20),
            "express":  ("uniform", 3, 10),
        },
        "weight_map":  {"standard": 1.0, "express": 3.0},
        "deadline_multiplier": {"standard": 3.0, "express": 1.8},
        "objective": "makespan",
        "change_type": "distributional",
    },
    "L_D2": {
        "description": "Order mix shift: bulky orders surge (40% of orders)",
        "total_resources": 100.0,
        "task_types": ["standard", "express", "bulky"],
        "type_weights": [0.40, 0.20, 0.40],
        "runtime_params": {
            "standard": ("lognormal", np.log(25), 0.4),
            "express":  ("lognormal", np.log(15), 0.3),
            "bulky":    ("lognormal", np.log(40), 0.5),
        },
        "resource_params": {
            "standard": ("uniform",  8, 20),
            "express":  ("uniform",  3, 10),
            "bulky":    ("uniform", 25, 45),   # high load
        },
        "weight_map":  {"standard": 1.0, "express": 3.0, "bulky": 1.0},
        "deadline_multiplier": {"standard": 3.0, "express": 1.8, "bulky": 4.0},
        "objective": "makespan",
        "change_type": "distributional",
    },
    "L_S1": {
        "description": "Vehicle breakdown: fleet capacity reduced by 30% (C→0.7C)",
        "total_resources": 70.0,   # reduced
        "task_types": ["standard", "express"],
        "type_weights": [0.70, 0.30],
        "runtime_params": {
            "standard": ("lognormal", np.log(25), 0.4),
            "express":  ("lognormal", np.log(15), 0.3),
        },
        "resource_params": {
            "standard": ("uniform", 8, 20),
            "express":  ("uniform", 3, 10),
        },
        "weight_map":  {"standard": 1.0, "express": 3.0},
        "deadline_multiplier": {"standard": 3.0, "express": 1.8},
        "objective": "makespan",
        "change_type": "structural",
    },
    "L_S2": {
        "description": "New remote zone added: 25% of orders now require long-distance delivery",
        "total_resources": 100.0,
        "task_types": ["standard", "express", "remote"],
        "type_weights": [0.50, 0.25, 0.25],
        "runtime_params": {
            "standard": ("lognormal", np.log(25), 0.4),
            "express":  ("lognormal", np.log(15), 0.3),
            "remote":   ("lognormal", np.log(70), 0.4),  # long distance
        },
        "resource_params": {
            "standard": ("uniform",  8, 20),
            "express":  ("uniform",  3, 10),
            "remote":   ("uniform", 10, 22),
        },
        "weight_map":  {"standard": 1.0, "express": 3.0, "remote": 1.0},
        "deadline_multiplier": {"standard": 3.0, "express": 1.8, "remote": 5.0},
        "objective": "makespan",
        "change_type": "structural",
    },
    "L_Sem1": {
        "description": "Objective shift: makespan → weighted lateness (SLA penalty mode)",
        "total_resources": 100.0,
        "task_types": ["standard", "express"],
        "type_weights": [0.60, 0.40],
        "runtime_params": {
            "standard": ("lognormal", np.log(25), 0.4),
            "express":  ("lognormal", np.log(15), 0.3),
        },
        "resource_params": {
            "standard": ("uniform", 8, 20),
            "express":  ("uniform", 3, 10),
        },
        "weight_map":  {"standard": 1.0, "express": 5.0},   # express penalty 5x
        "deadline_multiplier": {"standard": 2.5, "express": 1.4},  # tighter windows
        "objective": "weighted_lateness",
        "change_type": "semantic",
    },
}

WEIGHTED_COMPLETION_SCENES = {"L_Sem1"}

def _sample(spec, rng):
    name, *p = spec
    if name == "lognormal":
        return float(np.clip(rng.lognormal(p[0], p[1]), 1.0, 500.0))
    return float(rng.uniform(p[0], p[1]))

def generate_instance(scene_name: str, n_tasks: int, seed: int) -> dict:
    rng   = np.random.default_rng(seed)
    scene = SCENES[scene_name]
    types = scene["task_types"]
    tw    = scene["type_weights"]

    tasks = []
    for i in range(n_tasks):
        ttype = rng.choice(types, p=tw)
        rt    = _sample(scene["runtime_params"][ttype], rng)
        res   = float(np.clip(_sample(scene["resource_params"][ttype], rng),
                               1.0, scene["total_resources"] * 0.45))
        w     = scene["weight_map"][ttype]
        dl    = rt * scene["deadline_multiplier"][ttype] * rng.uniform(0.8, 1.2)

        tasks.append(DeliveryTask(
            id=i, task_type=ttype,
            estimated_runtime=round(rt, 2),
            runtime=round(rt * rng.uniform(0.9, 1.1), 2),
            resource_need=round(res, 2),
            weight=w,
            deadline=round(dl, 2),
            zone="urban" if ttype != "remote" else "rural",
        ))

    return {
        "tasks": tasks,
        "total_resources": scene["total_resources"],
        "objective": scene["objective"],
    }

def generate_instance_set(scene_name, n_instances, n_tasks, base_seed=0):
    return [generate_instance(scene_name, n_tasks, base_seed + i)
            for i in range(n_instances)]
