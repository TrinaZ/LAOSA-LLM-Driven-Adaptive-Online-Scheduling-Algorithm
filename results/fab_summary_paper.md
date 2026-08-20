# Fab 8-Seed 论文关键数字

**总记录数**: 720
**Trials**: 30 scenarios × 3 baselines × 8 seeds = 720 trials
**注**: CR基线已排除（deadline-driven，在无统一deadline场景行为不稳定）

## 摘要/Section 6.2 填入
- Total Fab trials: **240** (30 scenarios × 8 seeds)
- KW H (improvement): **86.8**
- KW p: p<0.001
- η²: **0.118**

## 各Tier统计（Table 9）
### Distributional
  - n=240, success=240/240
  - LLM queries: 5.8 ± 1.7 [min=3, max=8]
  - Improvement: 21.7% ± 18.0% (CV=83.0%)
  - Beats best: 79/240

### Structural
  - n=240, success=240/240
  - LLM queries: 7.4 ± 1.0 [min=5, max=8]
  - Improvement: 15.8% ± 12.2% (CV=77.2%)
  - Beats best: 131/240

### Semantic
  - n=240, success=240/240
  - LLM queries: 8.0 ± 0.0 [min=8, max=8]
  - Improvement: 10.9% ± 22.5% (CV=207.2%)
  - Beats best: 49/240

## KW检验
- Queries:     H=285.0, η²=0.395
- Improvement: H=86.8, η²=0.118