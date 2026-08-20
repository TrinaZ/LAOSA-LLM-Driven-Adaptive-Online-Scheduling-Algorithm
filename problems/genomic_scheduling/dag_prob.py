"""
dag_prob.py — DAG-aware extension of GenomicScheduling.

Adds:
  1. DAGTask: Task with predecessors list
  2. DAGSchedulingSimulator: respects precedence constraints
  3. DAGGenomicScheduling: drop-in replacement for GenomicScheduling
     with DAG instances, supports graph_context in priority function
  4. DAG scene generation for S2 and S5

Usage:
    from problems.genomic_scheduling.dag_prob import (
        DAGGenomicScheduling, DAG_BASELINES, DAG_SCENES
    )
    prob = DAGGenomicScheduling("dag_S2_resource_drop", n_instances=15, n_tasks=20)
    fitness = prob.evaluate(code_string)
    makespan = prob.baseline_makespan * (1 - fitness)
"""
import numpy as np
import types
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ================================================================
# 1. DAGTask — extends Task with precedence info
# ================================================================

@dataclass
class DAGTask:
    id: int
    task_type: str
    runtime: float
    estimated_runtime: float
    resource_need: float
    weight: float
    deadline: float
    predecessors: List[int] = field(default_factory=list)  # list of task ids


# ================================================================
# 2. GraphContext — exposes topological features to LLM-generated code
# ================================================================

class GraphContext:
    """
    Pre-computed graph features callable from LLM-generated priority functions.
    The LLM can call graph_context.out_degree(task), etc.

    Computed once per instance before scheduling begins.
    """

    def __init__(self, tasks: List[DAGTask]):
        self._tasks = {t.id: t for t in tasks}
        # Build successor lists
        self._successors: Dict[int, List[int]] = {t.id: [] for t in tasks}
        for t in tasks:
            for pred_id in t.predecessors:
                if pred_id in self._successors:
                    self._successors[pred_id].append(t.id)

        # Pre-compute derived features
        self._out_degree = {t.id: len(self._successors[t.id]) for t in tasks}
        self._in_degree = {t.id: len(t.predecessors) for t in tasks}
        self._recursive_workload = self._compute_recursive_workload(tasks)
        self._critical_path_remaining = self._compute_critical_path(tasks)
        self._contention: Dict[str, float] = {}

    def _compute_recursive_workload(self, tasks) -> Dict[int, float]:
        """Sum of processing times of all downstream tasks (topological order)."""
        memo = {}
        task_map = {t.id: t for t in tasks}
        succ = self._successors

        def dfs(tid):
            if tid in memo:
                return memo[tid]
            t = task_map[tid]
            total = t.estimated_runtime
            for s in succ[tid]:
                total += dfs(s)
            memo[tid] = total
            return total

        for t in tasks:
            dfs(t.id)
        return memo

    def _compute_critical_path(self, tasks) -> Dict[int, float]:
        """Longest path from each task to any sink (critical path remaining)."""
        memo = {}
        task_map = {t.id: t for t in tasks}
        succ = self._successors

        def dfs(tid):
            if tid in memo:
                return memo[tid]
            t = task_map[tid]
            if not succ[tid]:
                memo[tid] = t.estimated_runtime
            else:
                memo[tid] = t.estimated_runtime + max(dfs(s) for s in succ[tid])
            return memo[tid]

        for t in tasks:
            dfs(t.id)
        return memo

    def update_contention(self, resource_type: str, value: float):
        """Called by simulator to update live contention index."""
        self._contention[resource_type] = value

    # ---- Public API for LLM-generated code ----

    def out_degree(self, task: DAGTask) -> int:
        return self._out_degree.get(task.id, 0)

    def in_degree(self, task: DAGTask) -> int:
        return self._in_degree.get(task.id, 0)

    def recursive_workload(self, task: DAGTask) -> float:
        return self._recursive_workload.get(task.id, task.estimated_runtime)

    def critical_path_remaining(self, task: DAGTask) -> float:
        return self._critical_path_remaining.get(task.id, task.estimated_runtime)

    def contention_index(self, resource: str = "cpu") -> float:
        return self._contention.get(resource, 0.0)


# ================================================================
# 3. DAG Simulator — respects precedence constraints
# ================================================================

def simulate_dag(tasks: List[DAGTask], total_resources: float,
                 priority_func, graph_context: GraphContext) -> float:
    """
    Event-driven DAG scheduling simulation.

    Rules:
      - Task can only start when ALL predecessors are complete.
      - Multiple tasks run in parallel if resources allow.
      - priority_func(task, graph_context) -> float (higher = scheduled first).

    Returns makespan (lower is better).
    """
    task_map = {t.id: t for t in tasks}
    completion_time: Dict[int, float] = {}
    current_time = 0.0

    # Track state
    pending = set(t.id for t in tasks)    # not yet started
    running: List[tuple] = []              # (task_id, finish_time, resource_used)
    done = set()

    MAX_ITER = len(tasks) * 50
    itr = 0

    while (pending or running) and itr < MAX_ITER:
        itr += 1

        # Determine eligible tasks: all predecessors done
        eligible = [
            task_map[tid] for tid in pending
            if all(p in done for p in task_map[tid].predecessors)
        ]

        # Update contention index
        used_resources = sum(r for _, _, r in running)
        ci = used_resources / max(total_resources, 1.0)
        graph_context.update_contention("cpu", ci)

        # Sort eligible by priority (descending)
        try:
            eligible.sort(key=lambda t: -priority_func(t, graph_context))
        except TypeError:
            # Fallback: priority function doesn't accept graph_context
            try:
                eligible.sort(key=lambda t: -priority_func(t))
            except Exception:
                pass

        # Greedily assign eligible tasks
        available = total_resources - used_resources
        newly_started = []
        for task in eligible:
            if task.resource_need <= available:
                finish = current_time + task.runtime
                running.append((task.id, finish, task.resource_need))
                newly_started.append(task.id)
                available -= task.resource_need
                pending.discard(task.id)

        # If nothing is running and nothing started (deadlock guard)
        if not running:
            if pending:
                # Force-start the first eligible task regardless of resources
                if eligible:
                    task = eligible[0]
                    finish = current_time + task.runtime
                    running.append((task.id, finish, task.resource_need))
                    pending.discard(task.id)
                else:
                    # No eligible tasks and nothing running — should not happen
                    # in a valid DAG, but guard anyway
                    break
            else:
                break

        # Advance time to next completion event
        next_finish = min(finish for _, finish, _ in running)
        current_time = next_finish

        # Collect completed tasks
        still_running = []
        for (tid, finish, res) in running:
            if abs(finish - current_time) < 1e-9:
                done.add(tid)
                completion_time[tid] = current_time
            else:
                still_running.append((tid, finish, res))
        running = still_running

    # Any stuck tasks get a penalty completion time
    for tid in pending:
        current_time += 1.0
        completion_time[tid] = current_time
    for (tid, finish, _) in running:
        completion_time[tid] = finish

    if not completion_time:
        return 0.0
    return max(completion_time.values())


# ================================================================
# 4. DAG Instance Generation
# ================================================================

DAG_SCENES = {
    # Calibrated so that resource constraints and DAG precedence BOTH bind.
    # resource_choices=[8,16,32] with capacity=64 means 2-4 concurrent tasks max.
    # dag_edge_prob=0.20 gives avg 2-3 predecessors per task (meaningful topology).
    # This produces 40-55% degradation under S2 resource drop — matching main Table 3.
    "dag_baseline": {
        "description": "DAG genomic pipeline: multi-layer dependencies, resource-heavy tasks",
        "runtime_dist": ("normal", 15.0, 5.0),
        "task_mix": {"alignment": 0.7, "variant_calling": 0.3},
        "total_resources": 64,
        "resource_choices": [8, 16, 32],
        "weight_range": (1.0, 1.0),
        "dag_edge_prob": 0.20,
    },
    "dag_S2_resource_drop": {
        "description": "DAG pipeline: resource pool drops from 64 to 40 cores (~38% reduction)",
        "runtime_dist": ("normal", 15.0, 5.0),
        "task_mix": {"alignment": 0.7, "variant_calling": 0.3},
        "total_resources": 40,
        "resource_choices": [8, 16, 32],
        "weight_range": (1.0, 1.0),
        "dag_edge_prob": 0.20,
    },
    "dag_S5_objective_change": {
        "description": "DAG pipeline: urgent samples with 3x weight appear",
        "runtime_dist": ("normal", 15.0, 5.0),
        "task_mix": {"alignment": 0.7, "variant_calling": 0.3},
        "total_resources": 64,
        "resource_choices": [8, 16, 32],
        "weight_range": (0.5, 3.0),
        "dag_edge_prob": 0.20,
    },
}


def generate_dag_tasks(scene_name: str, n_tasks: int = 20,
                       seed: int = None) -> tuple:
    """
    Generate a batch of DAG tasks.
    DAG topology: acyclic edges only from lower-index to higher-index nodes
    (Erdos-Renyi model, as in Appendix B.1 of the paper).

    Returns (List[DAGTask], total_resources)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    scene = DAG_SCENES[scene_name]
    edge_prob = scene.get("dag_edge_prob", 0.15)

    tasks = []
    for i in range(n_tasks):
        type_list = list(scene["task_mix"].keys())
        probs = list(scene["task_mix"].values())
        task_type = rng.choice(type_list, p=probs)

        dist_name, p1, p2 = scene["runtime_dist"]
        resource_need = rng.choice(scene["resource_choices"])

        if dist_name == "normal":
            runtime = max(1.0, rng.normal(p1, p2))
        elif dist_name == "lognormal":
            runtime = max(1.0, rng.lognormal(p1, p2))
        else:
            runtime = max(1.0, rng.normal(p1, p2))

        estimated_runtime = runtime * rng.uniform(0.6, 1.4)

        w_lo, w_hi = scene["weight_range"]
        weight = rng.uniform(w_lo, w_hi)
        deadline = runtime * rng.uniform(1.5, 4.0)

        # DAG edges: task i can only depend on tasks 0..i-1
        predecessors = []
        if i > 0:
            for j in range(i):
                if rng.random() < edge_prob:
                    predecessors.append(j)

        tasks.append(DAGTask(
            id=i, task_type=task_type, runtime=runtime,
            estimated_runtime=estimated_runtime,
            resource_need=resource_need, weight=weight, deadline=deadline,
            predecessors=predecessors,
        ))

    return tasks, scene["total_resources"]


def generate_dag_instance_set(scene_name: str, n_instances: int = 15,
                               n_tasks: int = 20) -> List[Dict]:
    """Generate multiple DAG instances for evaluation."""
    instances = []
    for i in range(n_instances):
        tasks, resources = generate_dag_tasks(scene_name, n_tasks,
                                              seed=2000 + i)
        gc = GraphContext(tasks)
        instances.append({
            "tasks": tasks,
            "total_resources": resources,
            "graph_context": gc,
            "scene": scene_name,
        })
    return instances


# ================================================================
# 5. DAGGenomicScheduling — drop-in replacement with DAG support
# ================================================================

class DAGGenomicScheduling:
    """
    DAG-aware scheduling evaluator.

    Identical interface to GenomicScheduling:
      - self.evaluate(code_string) -> fitness (float, higher is better)
      - self.baseline_makespan (float)

    The priority function in code_string can optionally accept a second
    argument `graph_context`:

        def priority(task, graph_context):
            return graph_context.out_degree(task) * task.estimated_runtime

    If it only accepts one argument, it degrades to topology-blind scheduling
    (same as original), allowing direct comparison.
    """

    def __init__(self, scene: str = "dag_baseline",
                 n_instances: int = 15, n_tasks: int = 20):
        self.scene = scene
        self.n_tasks = n_tasks
        self.instances = generate_dag_instance_set(scene, n_instances, n_tasks)
        self.baseline_makespan = self._eval_random_baseline()

    def _eval_random_baseline(self) -> float:
        makespans = []
        for inst in self.instances:
            ms = simulate_dag(
                inst["tasks"], inst["total_resources"],
                lambda t, gc: np.random.random(),
                inst["graph_context"]
            )
            makespans.append(ms)
        return np.mean(makespans)

    def _extract_priority_func(self, code_string: str):
        """
        Compile code_string and return the priority function.
        Detects whether function accepts (task) or (task, graph_context).
        """
        module = types.ModuleType("dag_heuristic")
        exec(f"import math\n{code_string}", module.__dict__)
        func = getattr(module, "priority", None)
        if func is None:
            return None

        # Detect arity
        import inspect
        try:
            sig = inspect.signature(func)
            n_params = len(sig.parameters)
        except (ValueError, TypeError):
            n_params = 1

        if n_params >= 2:
            return func  # accepts (task, graph_context)
        else:
            # Wrap single-arg function to accept graph_context silently
            def wrapped(task, graph_context):
                return func(task)
            return wrapped

    def evaluate(self, code_string: str) -> Optional[float]:
        """
        Evaluate a priority function code string on DAG instances.
        Returns fitness (higher = better), or None on failure.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                priority_func = self._extract_priority_func(code_string)
                if priority_func is None:
                    return None

                makespans = []
                for inst in self.instances:
                    # Re-create graph context fresh each evaluation
                    # (contention state is mutated during simulation)
                    gc = GraphContext(inst["tasks"])
                    ms = simulate_dag(
                        inst["tasks"], inst["total_resources"],
                        priority_func, gc
                    )
                    makespans.append(ms)

                avg_makespan = np.mean(makespans)
                fitness = (self.baseline_makespan - avg_makespan) / self.baseline_makespan
                return fitness

        except Exception:
            return None


# ================================================================
# 6. DAG Baselines — same rules but topology-blind (fair comparison)
# ================================================================

DAG_BASELINES = {
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
    # Topology-aware oracle baseline (uses graph_context)
    "CPF": """
def priority(task, graph_context):
    # Critical Path First: prioritize tasks on the longest remaining path
    return graph_context.critical_path_remaining(task)
""",
}
