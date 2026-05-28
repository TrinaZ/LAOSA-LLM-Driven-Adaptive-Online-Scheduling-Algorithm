# LAOSA-LLM-Driven-Adaptive-Online-Scheduling-Algorithm
# LAOSA — LLM-Driven Adaptive Online Scheduling Algorithm

> **论文**：*Autonomous Algorithm Adaptation for Non-Stationary Online Scheduling: An LLM-Driven Framework*  
> **投稿期刊**：Production and Operations Management (POMS)  
> **本文件用途**：供 LLM（大语言模型）快速理解整个项目，以便立即协助改进或续写代码

---

## 目录

1. [项目核心思想](#1-项目核心思想)
2. [完整架构图](#2-完整架构图)
3. [五阶段 Pipeline 详解](#3-五阶段-pipeline-详解)
4. [每个文件的作用](#4-每个文件的作用)
5. [场景定义（实验变量）](#5-场景定义实验变量)
6. [基线算法](#6-基线算法)
7. [关键数据结构和接口](#7-关键数据结构和接口)
8. [实验矩阵与当前结果](#8-实验矩阵与当前结果)
9. [如何运行](#9-如何运行)
10. [各文件整合关系（调用图）](#10-各文件整合关系调用图)
11. [当前问题与待改进方向](#11-当前问题与待改进方向)
12. [论文章节与代码的对应关系](#12-论文章节与代码的对应关系)

---

## 1. 项目核心思想

**问题**：基因组测序流水线（Genomic Pipeline）在运行中会发生"场景切换"（Scene Change）——任务分布改变、资源容量下降、出现新任务类型、优化目标改变。传统的固定调度算法（如 SPT、LPT）无法自适应，性能会大幅退化。

**方案**：LAOSA 是一个四阶段自适应框架：
1. **检测**（Detect）：用 CUSUM 统计检验发现性能退化
2. **诊断**（Diagnose）：分析性能瓶颈，生成对场景变化的自然语言描述
3. **生成**（Generate）：调用 LLM 进行进化搜索，生成新的调度优先级函数
4. **部署**（Deploy）：评估新函数，若改善超过阈值则替换当前算法

**核心创新**：
- 把调度算法表示为 `def priority(task) -> float` 的 Python 函数，LLM 可以直接生成、修改、优化
- 引入 NAV（Net Adaptation Value）门槛，避免在不必要的时候浪费 LLM 调用
- 进化搜索（Evolutionary Search）：每轮生成 5 个变体，选最优，连续 3 轮无改善停止，上限 20 次 LLM 调用
- 算法记忆库（AlgorithmArchive）：存储历史成功算法，下次遇到相似场景直接热启动

**应用域**：基因组测序任务调度（WGS pipeline），包含三类任务：`alignment`（比对）、`variant_calling`（变异检测）、`sv_calling`（结构变异检测）。

---

## 2. 完整架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LAOSA 系统架构                                    │
│                                                                         │
│  输入流：任务批次 → [调度算法 π] → makespan 观测值                          │
│                                      │                                  │
│                          ┌───────────▼──────────┐                      │
│                          │  SceneChangeDetector  │  laosa.py            │
│                          │  CUSUM 统计检验        │                      │
│                          └───────────┬──────────┘                      │
│                              检测到退化 │                                │
│                          ┌───────────▼──────────┐                      │
│                          │    NAV 门槛判断        │  run_experiments.py  │
│                          │  skip/tier1/full_evo  │                      │
│                          └───────────┬──────────┘                      │
│                           full_evo   │                                  │
│     ┌────────────────────┐           │                                  │
│     │  AlgorithmArchive   │◄──────── ▼                                  │
│     │  历史记忆库查询      │  ┌───────────────────────┐                  │
│     └────────┬───────────┘  │  LLMAlgorithmAdapter   │  laosa.py        │
│      热启动   │              │  进化搜索 (最多20次调用) │                  │
│              └─────────────►│  每轮5个候选 → 评估      │                  │
│                             │  → 选最优 → 3轮无改善停  │                  │
│                             └───────────┬───────────┘                  │
│                                         │                               │
│                             ┌───────────▼───────────┐                  │
│                             │  GenomicScheduling     │  prob.py         │
│                             │  evaluate(code_string) │                  │
│                             │  → fitness (0~1)       │                  │
│                             └───────────────────────┘                  │
│                                                                         │
│  LLM 调用链：llm_adapter.py → DeepSeek/OpenAI/Anthropic/Simulated       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 五阶段 Pipeline 详解

### Phase 1：基线测量
```python
prob_before = GenomicScheduling("baseline", n_instances=10, n_tasks=20)
baseline_fitness = prob_before.evaluate(BASELINES["SPT"])
baseline_makespan = prob_before.baseline_makespan * (1 - baseline_fitness)
```
建立正常场景下的 makespan 基准。

### Phase 2：场景切换后评估
```python
prob_after = GenomicScheduling("S2_resource_drop", n_instances=10, n_tasks=20)
degraded_fitness = prob_after.evaluate(BASELINES["SPT"])
degraded_makespan = prob_after.baseline_makespan * (1 - degraded_fitness)
degradation = (degraded_makespan / baseline_makespan - 1) * 100  # 退化百分比
```

### Phase 3：CUSUM 变化检测
```python
detector = SceneChangeDetector(warmup=10, threshold=2.5)
for ms in stream:
    result = detector.update(ms)
    if result["detected"]:
        # 触发适应
```
- `warmup=10`：前10步建立基线均值和方差
- `threshold=2.5`：CUSUM 超过此值触发报警
- 检测延迟通常 1~5 步

### Phase 4：NAV 门槛 + LLM 进化搜索

**NAV（Net Adaptation Value）三层决策**（在 `run_experiments.py`）：
- `skip`：退化 < 10%，不值得适应，继续用原算法
- `tier1`：最优固定基线 ρ ≥ 95%，直接换基线，不调 LLM
- `tier1_limited`：历史改善期望 < 2%（且有 ≥3 条历史记录），最多 5 次 LLM 调用
- `full_evo`：正常进化搜索，最多 20 次 LLM 调用

**进化搜索循环**（在 `LLMAlgorithmAdapter.generate()`）：
```
初始：incumbent = 当前最优代码（SPT 等基线）
每轮：
  1. 构建 prompt（包含当前 incumbent + 性能数据 + 过去失败尝试）
  2. 调用 LLM，要求生成 N_CANDIDATES=5 个变体
  3. 逐个编译、验证、在 prob_after 上评估
  4. 若最佳变体 makespan < incumbent makespan → 更新 incumbent
  5. 连续 PATIENCE=3 轮无改善 或 总调用 ≥ MAX_LLM_CALLS=20 → 停止
```

### Phase 5：评估与部署
最终算法与所有固定基线对比，记录：
- `improvement_pct`：相对退化后的改善百分比
- `recovery_pct`：向基线恢复的百分比 = `(degraded - adapted) / (degraded - baseline) * 100`
- `beats_best_baseline`：是否超过最优固定基线

---

## 4. 每个文件的作用

### 核心框架（修改必读）

#### `laosa.py` ★★★ 最核心文件

包含 3 个类：

**`ArchiveLogger`**：记录所有 LLM 生成的候选算法到 `generated_archive.pkl`，供消融实验 A 使用。用法：
```python
from laosa import set_archive_logger, save_archive, ArchiveLogger
logger = ArchiveLogger()
set_archive_logger(logger)
# ... 运行实验 ...
save_archive("generated_archive.pkl")
```

**`SceneChangeDetector`**：
- `__init__(warmup=15, threshold=3.0)`：warmup 步数内建立基线，threshold 是 CUSUM 报警阈值
- `update(makespan) -> dict`：输入一个观测值，返回 `{"detected": bool, "cusum_pos": float, "cusum_neg": float, "degradation_pct": float, ...}`

**`LLMAlgorithmAdapter`**：
- `build_prompt(current_code, perf_data, scene_diagnosis, past_attempts, iteration)`：构建发给 LLM 的 prompt，包含当前最优算法代码、性能数据、历史失败尝试
- `generate(current_code, perf_data, scene_diagnosis, prob_after=None)`：主入口。`prob_after=None` 为单次调用（旧模式），`prob_after` 为 GenomicScheduling 实例时进行进化搜索（新模式，与论文一致）
- `_parse_variants(raw_text)`：从 LLM 返回的文本中解析 `def priority(task)` 函数，去重
- `compile(code)`：编译检验，用 DummyTask 测试是否返回 float
- `_simulate_response(diagnosis)`：离线 fallback，根据 diagnosis 关键词返回预设算法

**`LAOSAExperiment`**：
- `run(scene_before, scene_after, change_description, ...)`：完整 5 阶段实验，从 Phase 1 开始，每次新建 prob 实例
- `run_with_probs(scene_before, scene_after, ..., prob_before, prob_after, baseline_ms, degraded_ms, ...)`：复用外部已建好的 prob 实例，从 Phase 3 开始，避免重复计算（供 run_experiments.py 在 NAV 检验后调用）

---

#### `llm_adapter.py` ★★★

LLM API 调用封装，自动检测可用 provider。

**provider 优先级**（按环境变量检测）：
1. `DEEPSEEK_API_KEY` → `deepseek-chat`（最便宜，推荐）
2. `OPENAI_API_KEY` → `gpt-4o-mini`
3. `ANTHROPIC_API_KEY` → `claude-sonnet-4-20250514`
4. `ZHIPU_API_KEY` → `glm-4-flash`（有免费额度）
5. 无 → `simulated`（使用 laosa.py 中的硬编码 fallback）

**主函数**：
```python
result = call_llm(prompt, max_tokens=2048)  # 返回提取后的代码字符串或 None
provider, key = detect_provider()           # 检测当前 provider
```

`_extract_code(raw_text)`：从 LLM 输出中提取 `def priority` 函数，处理 ````python`、```` ``` ````、裸函数等格式。

---

#### `algorithm_archive.py` ★★

跨场景记忆库，存储历史成功算法。

**`AlgorithmArchive`** 类：
- `store(scene_id, scene_features, algorithm_code, makespan, best_fixed_makespan)`：存储成功算法
- `retrieve(scene_id, scene_features, top_k=3) -> List[Dict]`：按场景特征相似度检索历史算法（作为进化搜索的热启动种子）

**`extract_scene_features(scene_name)`**（模块级函数）：从场景名称提取特征向量（distributional/structural/semantic）

文件持久化到 `results/algorithm_archive.json`。

---

### 问题定义层

#### `problems/genomic_scheduling/prob.py` ★★★

**`GenomicScheduling`** 类（EoH 兼容接口）：
```python
prob = GenomicScheduling(scene="S2_resource_drop", n_instances=10, n_tasks=20)
fitness = prob.evaluate(code_string)  # float，越高越好，用 baseline 归一化
makespan = prob.baseline_makespan * (1 - fitness)  # 实际 makespan
```

- `_simulate(tasks, total_resources, priority_func, weighted=False)`：核心仿真器。按优先级排序 → 贪婪分配资源 → 模拟时间推进 → 返回 makespan（或加权完成时间）
- `use_weighted`：S5 场景（`S5_objective_change`）自动切换为加权完成时间（`sum(weight × completion_time)`）作为评估指标，其他场景用 makespan
- **`BASELINES` 字典**：包含 SPT、LPT、EDF、WSJF、ResourceEfficient 五个基线的代码字符串，可直接传给 `evaluate()`

#### `problems/genomic_scheduling/instances.py` ★★★

**`Task` dataclass**：
```python
@dataclass
class Task:
    id: int
    task_type: str           # 'alignment' / 'variant_calling' / 'sv_calling'
    runtime: float           # 真实运行时间（仿真用）
    estimated_runtime: float # 预估运行时间（算法看到的，有噪声）
    resource_need: float     # CPU cores 需求
    weight: float            # 紧急程度权重
    deadline: float          # 截止时间
```

**`SCENES` 字典**：定义所有实验场景（详见第 5 节）

**`generate_instance_set(scene_name, n_instances, n_tasks)`**：生成用于评估的实例集合

#### `problems/genomic_scheduling/dag_prob.py` ★

DAG（有向无环图）依赖约束版本，扩展了 prob.py。

**新增内容**：
- `DAGTask`：在 Task 基础上增加 `predecessors: List[int]`（前驱任务 ID 列表）
- `GraphContext`：预计算图特征（出度、入度、关键路径剩余时间、递归工作量），可在 LLM 生成的优先级函数中调用：
  ```python
  def priority(task, graph_context=None):
      if graph_context:
          cp = graph_context.critical_path_remaining(task)
          od = graph_context.out_degree(task)
      ...
  ```
- `DAGGenomicScheduling`：尊重前驱约束的调度仿真器
- `DAG_SCENES`：两个 DAG 场景（`dag_baseline`、`dag_S2_resource_drop`、`dag_S5_objective_change`）
- `DAG_BASELINES`：DAG 场景用的基线算法

#### `problems/genomic_scheduling/prompts.py`

`GetPrompts` 类：提供 LLM prompt 模板（函数签名描述、任务字段说明）。由 `GenomicScheduling.__init__` 自动实例化。

---

### 实验脚本层

#### `run_experiments.py` ★★★ 主实验入口

**用法**：
```bash
python3 run_experiments.py mvp        # 单场景验证（S2+SPT，约10分钟）
python3 run_experiments.py full       # 完整 5×4 矩阵（20个实验）
python3 run_experiments.py baselines  # 所有基线在所有场景评估（不调 LLM）
python3 run_experiments.py nav        # NAV worthiness 分析（不调 LLM，约5分钟）
```

**NAV 决策函数** `compute_nav_decision()`：三门槛判断，见 Phase 4 详解。

**主循环** `run_full_matrix()`：遍历 5 个场景 × 4 个基线 = 20 个实验，每个实验先做 NAV 判断，再决定是否调 LLM。

**输出文件**：`results/full_matrix.json`（20个实验的完整结果）

#### `exp_5_3_collapse.py` ★★★ 论文 Figure 2 核心实验

生成 Table 1（大规模崩溃）和 Figure 2（时间序列）的数据。

- `LargeScaleSimulator`：1000 个任务，DAG 依赖结构，在完成 30% 时触发资源突变（CPU×0.6）
- 对照组设计：扰动前两条曲线（LAOSA / Non-adaptive）完全相同，扰动后分叉
- 输出：`figure2_raw_data.csv`（供 `compute_regret.py` 读取）

```bash
python exp_5_3_collapse.py
python compute_regret.py --data_path figure2_raw_data.csv --plot
```

#### `compute_regret.py` ★★

从 `figure2_raw_data.csv` 计算 5 个 regret 指标（论文 Section 6.4）：
- `[A]` 初始 regret spike（扰动后第一时刻）
- `[B]` 峰值 regret
- `[C]` 收敛速度（从峰值降到 5% 所需步数）
- `[D]` 稳态 regret（最后 20% 窗口均值）
- `[E]` 总累积 regret

```bash
python compute_regret.py --data_path figure2_raw_data.csv --plot
```

#### `exp_A_laosa_no_generation.py` ★★ 消融实验 A

**目的**：分离"更大的搜索空间"和"LLM 定向搜索策略"的贡献。

LAOSA-no-generation：用与 LAOSA 完全相同的候选算法池（`generated_archive.pkl`），但不调 LLM，只随机/相似度搜索已有候选。

**前提**：需要先运行完整 LAOSA 实验，且设置了 `ArchiveLogger` 保存候选池。

```bash
python exp_A_laosa_no_generation.py --archive_path generated_archive.pkl
```

#### `exp_5_2_optimality.py`

最优性 Gap 实验（Table 4）。需要 **Gurobi** 求解器（商业授权）。
- 用 Gurobi 求解小规模实例的精确最优解
- 计算 LAOSA 与最优解的 optimality gap

#### `run_dag_experiments.py` ★

在 DAG 场景上运行 LAOSA，验证对有向图依赖约束的泛化性。
```bash
python3 run_dag_experiments.py mvp   # 快速：S2+SPT
python3 run_dag_experiments.py       # 完整：S2+S5，各4个基线
```
使用 `DAGGenomicScheduling` 和 `MiniEvolutionV2`。

#### `run_reevo_comparison_FINAL.py` ★★

与 ReEvo 基线的直接对比（论文 Table 6）。ReEvo 使用 100 次 LLM 调用，LAOSA 使用 ≤20 次。

```bash
python3 run_reevo_comparison_FINAL.py --mode=both
```

#### `run_statistical_tests_fixed.py` ★

Wilcoxon 符号秩检验和 Mann-Whitney U 检验，验证统计显著性。
```bash
python run_statistical_tests_fixed.py
```

#### `cross_llm_experiment.py` ★

跨 LLM 鲁棒性实验（Table 7）：测试 DeepSeek / GPT-4o / Claude 3.5 / Gemini 1.5 的结果差异。需要同时配置多个 API key。

#### `memory_experiment.py` ★

15 次顺序场景冲击实验，验证记忆库积累效果（M-LAOSA vs NM-LAOSA）。

#### `gemini_experiments.py`

使用 Google Gemini API 的实验变体，接口与 `llm_adapter.py` 平行。

---

### 进化搜索组件

#### `mini_evolution.py`

早期原型，独立的进化搜索类 `MiniEvolution`（与 laosa.py 中的 `LLMAlgorithmAdapter` 平行）：
- `n_rounds=8`：进化轮数
- `candidates_per_round=3`：每轮候选数
- 支持 "thought + code"（EoH 风格）、reflect（反思失败原因）、crossover（交叉两个优秀算法）

#### `mini_evolution_v2.py`

`MiniEvolutionV2`：mini_evolution.py 的升级版，供 `run_dag_experiments.py` 使用。增加了 bottleneck 报告集成和 archive 热启动。

#### `bottleneck.py`

详细调度仿真器，带完整诊断信息：
- `simulate_detailed(tasks, total_resources, priority_func) -> dict`：返回 makespan + 详细瓶颈分析
- `format_bottleneck_report(diag_result) -> str`：将诊断结果格式化为给 LLM 的自然语言描述（包含资源利用率、队列长度、阻塞任务类型等）

#### `statistical_analyzer.py`

统计分析增强，提供分布检验和拟合（KS 检验、MLE）：
- `StatisticalAnalyzer.analyze_and_generate_hint(trace_data) -> str`：生成分布诊断的自然语言提示（如"运行时间从正态分布转变为对数正态分布，方差增大 3x"）

**注意**：目前此模块未与主流程连接，是待集成的增强功能。

---

### 辅助文件

#### `data_generate.py`
合成数据生成，用于预生成实验实例（非主路径，可独立运行）。

#### `5_4.py`
Section 5.4 消融实验：`AblationSimulator` 对比 4 种策略：
- `LPT (Baseline)`：纯时间排序
- `LAOSA-Sym`：符号算子（任务属性，无图拓扑）
- `LAOSA-GNN`：增加拓扑感知（出度）
- `LAOSA-Full`：完整版（拓扑 + 资源感知 + 动态调整）

#### `run_evolution.py` / `run_evolution_v2.py`
早期实验脚本，用于测试 mini_evolution.py 和 mini_evolution_v2.py。

#### `run_supplementary.py` / `run_supplementary_v2_enhanced.py`
补充实验（multiresource、scale、repeated、combined 等）。

#### `extreme_dag_generator.py`
生成极端 DAG 拓扑结构（high_fanout、deep_nested、complex_hybrid）用于压力测试。结果保存在 `dags/` 目录。

#### `theory_framework.md`
论文理论框架的数学记录：问题形式化（Definition 1-5）、CUSUM 理论（Definition 6-9）、LLM 搜索理论（Definition 10-12）、NAV 分析、复杂度分析。

---

### 数据文件

| 文件 | 内容 |
|------|------|
| `results/full_matrix.json` | 20 个实验（5 场景 × 4 基线）的完整结果 |
| `results/baselines.json` | 所有固定基线在所有场景的 makespan |
| `results/algorithm_archive.json` | 历史成功算法存档 |
| `results/cross_llm_results.json` | 跨 LLM 对比结果 |
| `results/dag_validation.json` | DAG 场景实验结果 |
| `results/statistical_tests.json` / `_table.tex` | 统计检验结果 |
| `results/reevo_vs_laosa_comparison.json` | LAOSA vs ReEvo 对比 |
| `generated_archive.pkl` | 所有 LLM 生成的候选算法（pkl，供实验 A 用） |
| `figure2_raw_data.csv` | Figure 2 时间序列原始数据（t, laosa_makespan, nonadaptive_makespan） |
| `figure2_metrics.json` | regret 指标 [A]-[E] |
| `regret_results.json` | compute_regret.py 的输出 |
| `memory_results.json` | 记忆实验结果 |
| `results_no_gen.json` | 实验 A 结果 |
| `dags/` | 极端 DAG 结构（JSON，供 extreme_dag_generator.py 使用）|

---

## 5. 场景定义（实验变量）

```python
# instances.py 中 SCENES 字典

"baseline"           # 标准WGS：正态分布(μ=10,σ=2)，alignment:70%，资源100核
"S1_runtime_shift"   # FFPE样本：重尾对数正态分布(μ=2.3,σ=1.0)，分布漂移
"S2_resource_drop"   # 资源突降：CPU 100核 → 60核，结构性变化
"S3_task_mix_shift"  # 任务比例变化：variant_calling 30% → 70%，分布漂移
"S4_novel_task"      # 新任务类型：sv_calling 出现（超长、超重资源），结构性
"S5_objective_change"# 目标改变：高权重紧急样本出现，评估指标切换为加权完成时间
```

场景变化类型（用于 NAV 和 archive 分类）：
- `distributional`：S1、S3（任务属性分布改变）
- `structural`：S2、S4（约束/任务类型结构改变）
- `semantic`：S5（优化目标改变）

---

## 6. 基线算法

```python
# prob.py 中 BASELINES 字典

"SPT"              # Shortest Processing Time：return -task.estimated_runtime
"LPT"              # Longest Processing Time：return task.estimated_runtime
"EDF"              # Earliest Deadline First：return -task.deadline
"WSJF"             # Weighted Shortest Job First：return task.weight / max(task.estimated_runtime, 0.01)
"ResourceEfficient"# 最小资源需求：return -task.resource_need
```

所有代码字符串可直接传给 `prob.evaluate()`。

---

## 7. 关键数据结构和接口

### Task 字段（LLM 生成算法可以访问的）
```python
task.estimated_runtime  # float：预估剩余时间（有噪声）
task.resource_need      # float：需要的 CPU 核数（4 / 8 / 16 / 32）
task.weight             # float：紧急程度权重（baseline=1.0，S5 场景 0.5~3.0）
task.deadline           # float：截止时间
task.task_type          # str：'alignment' / 'variant_calling' / 'sv_calling'
```

### priority 函数标准接口
```python
def priority(task):
    # 返回 float，越高 = 越高优先级 = 越先调度
    return task.weight / max(task.estimated_runtime, 0.01)
```

### evaluate() 返回值
```python
fitness = prob.evaluate(code_string)
# fitness > 0：相对 random baseline 有改善
# fitness < 0：比 random baseline 更差
# fitness = None：代码编译或运行失败
# makespan = prob.baseline_makespan * (1 - fitness)
```

### full_matrix.json 结果字段
```json
{
  "scene_before": "baseline",
  "scene_after": "S2_resource_drop",
  "baseline_algo": "SPT",
  "baseline_makespan": 25.06,
  "degraded_makespan": 38.42,
  "adapted_makespan": 37.15,
  "degradation_pct": 53.2,
  "improvement_pct": 3.3,
  "recovery_pct": 8.5,
  "detection_delay": 2,
  "generation_time_sec": 45.2,
  "best_fixed_baseline": "LPT",
  "best_fixed_makespan": 36.8,
  "beats_best_baseline": false,
  "adapted_code": "def priority(task): ...",
  "nav_action": "full_evo",
  "nav_rho": 0.97,
  "nav_delta": 0.532,
  "change_type": "structural"
}
```

---

## 8. 实验矩阵与当前结果

### 完整实验矩阵设计（5 场景 × 4 基线 = 20 个实验）

| 场景 | 变化类型 | 测试基线 |
|------|---------|---------|
| S1_runtime_shift | distributional | SPT, LPT, EDF, WSJF |
| S2_resource_drop | structural | SPT, LPT, EDF, WSJF |
| S3_task_mix_shift | distributional | SPT, LPT, EDF, WSJF |
| S4_novel_task | structural | SPT, LPT, EDF, WSJF |
| S5_objective_change | semantic | SPT, LPT, EDF, WSJF |

### 当前典型结果（2026-05-11，DeepSeek）

```
S1 × SPT：退化206%，改善9.6%，未超越最优基线（LPT），NAV=full_evo
S1 × LPT：退化190%，改善0.0%，NAV=tier1（自身已是最优基线）
S2 × SPT：退化53%，改善3.3%，未超越最优基线，NAV=full_evo
```

**主要挑战**：S1（重尾分布）场景改善不足，LLM 生成的算法收敛到类 LPT 方案，难以超越 LPT 基线本身。

---

## 9. 如何运行

### 环境配置
```bash
pip install numpy scipy networkx

# 选一个 LLM API（推荐 DeepSeek，最便宜）
export DEEPSEEK_API_KEY=sk-xxx
# 或
export OPENAI_API_KEY=sk-xxx
# 或
export ANTHROPIC_API_KEY=sk-xxx
```

### 快速验证（MVP，约10分钟，消耗约10次 API 调用）
```bash
python3 run_experiments.py mvp
```

### 完整实验矩阵（约2小时，消耗约200次 API 调用）
```bash
python3 run_experiments.py full
```

### 仅评估基线（不调 LLM，约2分钟）
```bash
python3 run_experiments.py baselines
```

### NAV 分析（不调 LLM，分析哪些场景值得适应）
```bash
python3 run_experiments.py nav
```

### Figure 2 大规模实验
```bash
python3 exp_5_3_collapse.py
python3 compute_regret.py --data_path figure2_raw_data.csv --plot
```

### DAG 实验
```bash
python3 run_dag_experiments.py mvp
```

---

## 10. 各文件整合关系（调用图）

```
run_experiments.py
  ├── laosa.py
  │   ├── LLMAlgorithmAdapter
  │   │   └── llm_adapter.py → DeepSeek/OpenAI/Anthropic API
  │   ├── SceneChangeDetector
  │   └── LAOSAExperiment
  │       └── problems/genomic_scheduling/prob.py
  │           └── problems/genomic_scheduling/instances.py (SCENES, Task)
  └── algorithm_archive.py

run_dag_experiments.py
  ├── mini_evolution_v2.py → llm_adapter.py
  ├── bottleneck.py → problems/genomic_scheduling/instances.py
  ├── algorithm_archive.py
  └── problems/genomic_scheduling/dag_prob.py
      └── problems/genomic_scheduling/instances.py

exp_5_3_collapse.py       （独立，不依赖 laosa.py，使用 networkx）
  └── → figure2_raw_data.csv

compute_regret.py         （独立，读取 figure2_raw_data.csv）

exp_A_laosa_no_generation.py  （独立，读取 generated_archive.pkl）

cross_llm_experiment.py   （依赖 openai, anthropic, google.generativeai）
  └── problems/genomic_scheduling/prob.py

exp_5_2_optimality.py     （依赖 Gurobi，独立）
```

---

## 11. 当前问题与待改进方向

### 已知问题

**1. S1 场景（重尾分布）改善不足**
LLM 生成的算法平均改善 < 5%，无法超越 LPT 基线。原因：重尾分布下 LPT 天然有优势，LLM 容易收敛到类 LPT 方案。
- 待改进：在 Phase 2 诊断中集成 `statistical_analyzer.py`，把"当前分布是对数正态、方差是基线的Nx"这个统计信息传给 LLM

**2. S5 场景 prompt 与评估指标不一致**
S5 使用加权完成时间（`sum(weight × completion_time)`）作为评估指标，但 prompt 中未明确告知 LLM，导致 LLM 优化方向与实际评估指标偏离。
- 待改进：在 `LLMAlgorithmAdapter.build_prompt()` 中根据场景类型动态修改任务说明

**3. 进化搜索 diversity 不足**
连续轮次中 LLM 容易生成重复或相似方案，`past_attempts` 提示效果有限。
- 待改进：引入 island model（多种群并行）或 prompt 中显式要求"与以下方案在结构上不同"

**4. ArchiveLogger 集成不稳定**
`generated_archive.pkl` 需要在调用 `run_experiments.py` 前手动设置 `set_archive_logger()`，容易遗漏导致实验 A 数据为空。
- 待改进：将 ArchiveLogger 集成到 `LAOSAExperiment.__init__` 中，默认开启

**5. NAV 门槛参数未经系统调优**
`NAV_DELTA_MIN = 0.10`（10% 退化门槛）和 `NAV_RHO_MAX = 0.95`（tier1 门槛）是经验设定，tier1 经常触发，跳过 LLM。
- 待改进：在 `run_experiments.py nav` 模式下系统扫描不同门槛的影响

### 待实现功能

- [ ] `statistical_analyzer.py` 与主流程连接（在 Phase 2 诊断时调用，将分布诊断作为额外 hint 传入 prompt）
- [ ] S5 场景的 prompt 增加"优化加权完成时间"的明确说明
- [ ] 跨 LLM 实验完整运行（目前 `cross_llm_experiment.py` 需要同时配置 4 个 API key）
- [ ] `memory_experiment.py` 连接真实 LAOSA runner（目前内部用 mock 数据填充）
- [ ] DAG 实验结果写入论文 Table 5b
- [ ] `run_statistical_tests_fixed.py` 验证所有 20 个实验结果的统计显著性

---

## 12. 论文章节与代码的对应关系

| 论文章节 | 代码位置 |
|---------|---------|
| Section 3 Problem Formulation | `theory_framework.md` §1、`instances.py` SCENES |
| Section 4.1 Scene Change Detection | `laosa.py` SceneChangeDetector |
| Section 4.2 Bottleneck Diagnosis | `bottleneck.py` simulate_detailed + format_bottleneck_report |
| Section 4.3 NAV Worthiness | `run_experiments.py` compute_nav_decision() |
| Section 4.4 LLM Evolutionary Search | `laosa.py` LLMAlgorithmAdapter.generate() |
| Section 4.5 Algorithm Archive | `algorithm_archive.py` |
| Section 5.4 Ablation (Sym/GNN/Full) | `5_4.py` AblationSimulator |
| Section 6.1 Main Results Table | `run_experiments.py` → `results/full_matrix.json` |
| Section 6.2 Optimality Gap (Table 4) | `exp_5_2_optimality.py`（需 Gurobi）|
| Section 6.3 Comparison with ReEvo (Table 6) | `run_reevo_comparison_FINAL.py` |
| Section 6.4 Large-Scale Collapse + Figure 2 | `exp_5_3_collapse.py` + `compute_regret.py` |
| Section 6.5 Memory Bank | `memory_experiment.py` |
| Section 6.6 Cross-LLM (Table 7) | `cross_llm_experiment.py` |
| Appendix A (No-Generation Ablation) | `exp_A_laosa_no_generation.py` |
| DAG Supplement (Table 5b) | `run_dag_experiments.py` + `dag_prob.py` |

---

*最后更新：2026-05-28*
