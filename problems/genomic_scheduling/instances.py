"""
Genomic pipeline scheduling environment with scene changes.
Generates task instances for online scheduling experiments.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Task:
    id: int
    task_type: str
    runtime: float
    estimated_runtime: float
    resource_need: float
    weight: float
    deadline: float
    speed: float = 1.0
    base_runtime: float = None

    def __post_init__(self):
        if self.base_runtime is None:
            self.base_runtime = self.runtime

# ================================================================
# SCENE DEFINITIONS
# ================================================================

SCENES = {
    "baseline": {
        "description": "Standard WGS pipeline: moderate runtimes, balanced task mix",
        "runtime_dist": ("normal", 10.0, 2.0),
        "task_mix": {"alignment": 0.7, "variant_calling": 0.3},
        "total_resources": 100,
        "resource_choices": [4, 8, 16],
        "weight_range": (1.0, 1.0),
    },
    "S1_runtime_shift": {
        "description": "FFPE samples: heavy-tailed runtime distribution",
        "runtime_dist": ("lognormal", 2.3, 1.0),
        "task_mix": {"alignment": 0.7, "variant_calling": 0.3},
        "total_resources": 100,
        "resource_choices": [4, 8, 16],
        "weight_range": (1.0, 1.0),
    },
    "S2_resource_drop": {
        "description": "Resource pool drops from 100 to 60 CPU cores",
        "runtime_dist": ("normal", 10.0, 2.0),
        "task_mix": {"alignment": 0.7, "variant_calling": 0.3},
        "total_resources": 60,
        "resource_choices": [4, 8, 16],
        "weight_range": (1.0, 1.0),
    },
    "S3_task_mix_shift": {
        "description": "Task mix shifts to more variant_calling (panel-heavy)",
        "runtime_dist": ("normal", 10.0, 3.0),
        "task_mix": {"alignment": 0.3, "variant_calling": 0.7},
        "total_resources": 100,
        "resource_choices": [4, 8, 16, 32],
        "weight_range": (1.0, 1.0),
    },
    "S4_novel_task": {
        "description": "New SV calling tasks appear (very long, resource-heavy)",
        "runtime_dist": ("normal", 10.0, 2.0),
        "task_mix": {"alignment": 0.5, "variant_calling": 0.2, "sv_calling": 0.3},
        "total_resources": 100,
        "resource_choices": [4, 8, 16],
        "weight_range": (1.0, 1.0),
        "sv_runtime_dist": ("lognormal", 3.0, 0.8),
        "sv_resource": 32,
    },
    "S5_objective_change": {
        "description": "Urgent samples with high weights appear",
        "runtime_dist": ("normal", 10.0, 2.0),
        "task_mix": {"alignment": 0.7, "variant_calling": 0.3},
        "total_resources": 100,
        "resource_choices": [4, 8, 16],
        "weight_range": (0.5, 3.0),
    },
}


def generate_tasks(scene_name: str, n_tasks: int = 20, seed: int = None) -> List[Task]:
    """Generate a batch of tasks for the given scene."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    scene = SCENES[scene_name]
    tasks = []

    for i in range(n_tasks):
        # Task type
        types = list(scene["task_mix"].keys())
        probs = list(scene["task_mix"].values())
        task_type = rng.choice(types, p=probs)

        # Runtime
        if task_type == "sv_calling" and "sv_runtime_dist" in scene:
            dist_name, p1, p2 = scene["sv_runtime_dist"]
            resource_need = scene.get("sv_resource", 32)
        else:
            dist_name, p1, p2 = scene["runtime_dist"]
            resource_need = rng.choice(scene["resource_choices"])

        if dist_name == "normal":
            runtime = max(1.0, rng.normal(p1, p2))
        elif dist_name == "lognormal":
            runtime = max(1.0, rng.lognormal(p1, p2))
        else:
            runtime = max(1.0, rng.normal(p1, p2))

        # Noisy estimate
        estimated_runtime = runtime * rng.uniform(0.6, 1.4)

        # Weight
        w_lo, w_hi = scene["weight_range"]
        weight = rng.uniform(w_lo, w_hi)

        # Deadline
        deadline = runtime * rng.uniform(1.5, 4.0)

        # Speed tier (physical property, used only for speed-scaling energy
        # scenarios). Default speed=1.0 when 'speed_tiers' absent from scene
        # config (fully backward compatible no-op for all existing scenarios).
        base_runtime = runtime
        speed = 1.0
        if "speed_tiers" in scene:
            tier_speeds = scene["speed_tiers"]
            tier_probs = scene.get("speed_tier_probs",
                                    [1.0 / len(tier_speeds)] * len(tier_speeds))
            speed = float(rng.choice(tier_speeds, p=tier_probs))
            runtime = base_runtime / speed
            deadline = runtime * rng.uniform(1.5, 4.0)
        
        tasks.append(Task(
            id=i, task_type=task_type, runtime=runtime,
            estimated_runtime=estimated_runtime,
            resource_need=resource_need, weight=weight, deadline=deadline,
            speed=speed, base_runtime=base_runtime,
        ))

    return tasks, scene["total_resources"]


def generate_instance_set(scene_name: str, n_instances: int = 10,
                          n_tasks: int = 20) -> List:
    """Generate multiple instances for evaluation."""
    instances = []
    for i in range(n_instances):
        tasks, resources = generate_tasks(scene_name, n_tasks, seed=1000 + i)
        instances.append({
            "tasks": tasks,
            "total_resources": resources,
            "scene": scene_name,
        })
    return instances


# ============================================================
# Speed-scaling energy makespan scenario (for operator-coverage
# failure validation, Lemma III.2 in TICPS paper).
# ============================================================

if "S1_speedscale_energy" not in SCENES:
    SCENES["S1_speedscale_energy"] = dict(
        SCENES["S1_runtime_shift"],
        description=(
            "S1_runtime_shift with speed-scaling energy-aware makespan objective. "
            "Each task is assigned a fixed speed tier (0.6/1.0/1.5) at generation time; "
            "faster tiers finish sooner but consume cubically more energy (cube-root "
            "power law). Objective = makespan + gamma * sum(base_runtime * speed^3). "
            "The speed tier is a physical property NOT exposed in the base operator "
            "vocabulary V -- testing whether LAOSA can discover energy-aware scheduling "
            "without explicit speed-tier information."
        ),
        speed_tiers=[0.6, 1.0, 1.5],
        speed_tier_probs=[0.35, 0.40, 0.25],
    )
