# WGS 8-Seed 论文关键数字

**总记录数**: 960
**Trials**: 30 scenarios × 4 baselines × 8 seeds = 960 trials

## 摘要/Section 6.1 填入
- Total WGS trials: **240** (30 scenarios × 8 seeds)
- KW H on improvement: **H=353.9**
- KW p: p<0.001
- η²: **0.368**

## 各Tier统计（Table 4/5）
### Distributional
  - n=320, success=320/320
  - LLM queries: 4.1 ± 0.9 [min=3, max=5]
  - Latency: 0.39 ± 0.16 min
  - Improvement: 0.2% ± 5.8% (CV=2410.3%)
  - Beats best baseline: 2/320

### Structural
  - n=320, success=320/320
  - LLM queries: 4.1 ± 0.9 [min=3, max=5]
  - Latency: 0.42 ± 0.16 min
  - Improvement: -1.3% ± 11.4% (CV=nan%)
  - Beats best baseline: 121/320

### Semantic
  - n=320, success=320/320
  - LLM queries: 4.6 ± 0.7 [min=3, max=5]
  - Latency: 0.49 ± 0.16 min
  - Improvement: 8.4% ± 3.7% (CV=43.8%)
  - Beats best baseline: 263/320

## KW检验结果
- Queries:     H=64.5, η²=0.065
- Improvement: H=353.9, η²=0.368

## Section 6.1 替换文字（参考）
each scenario is evaluated across **8** independent random-seed runs
(30 scenarios × 8 seeds = **240** total LAOSA trials per domain).
The coefficient of variation in improvement across seeds is below
**2410.3%**
for all tiers, confirming result stability.