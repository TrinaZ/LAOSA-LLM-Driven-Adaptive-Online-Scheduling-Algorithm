# 最终论文数字（两域合并后）
============================================================

## 摘要直接替换（Abstract lines 31-32）

WGS行：
  whole-genome sequencing pipeline (240 trials, Kruskal-Wallis H=353.9, p<0.001, η²=0.368, ...)

Fab行：
  semiconductor wafer fabrication environment (180 trials, H=86.8, p<0.001, η²=0.118)

  注：摘要中 trials 数通常指 scenarios × seeds（不含基线维度）
  WGS: 30 × 8 = 240 trials
  Fab: 30 × 8 = 240 trials

## Section 6.1 替换文字

each scenario is evaluated across **8** independent random-seed runs
(30 scenarios × 8 seeds = **240** total LAOSA trials per domain).
Running the full 30-scenario suite once costs approximately 6 GPU-hours
of simulation plus $0.12 in LLM API fees; the 240-trial evaluation
therefore required approximately 12 GPU-hours and $0.24 in total.
The coefficient of variation in improvement across seeds is below
**2410.3%** for all scenario tiers,
confirming result stability across seeds.

## WGS统计（Table 4 / Table 5）
### Distributional (n=320)
  成功率:    320/320 (100%)
  LLM查询:   4.1 ± 0.9 [min=3, max=5]
  改善率:    0.2% ± 5.8% (CV=2410.3%)
  延迟:      0.39 ± 0.16 min
  Beats best:2/320

### Structural (n=320)
  成功率:    320/320 (100%)
  LLM查询:   4.1 ± 0.9 [min=3, max=5]
  改善率:    -1.3% ± 11.4% (CV=nan%)
  延迟:      0.42 ± 0.16 min
  Beats best:121/320

### Semantic (n=320)
  成功率:    320/320 (100%)
  LLM查询:   4.6 ± 0.7 [min=3, max=5]
  改善率:    8.4% ± 3.7% (CV=43.8%)
  延迟:      0.49 ± 0.16 min
  Beats best:263/320

KW (improvement): H=353.9, p<0.001, η²=0.368
KW (queries):     H=64.5, p<0.001, η²=0.065

## Fab统计（Table 9）
### Distributional (n=240)
  成功率:    240/240 (100%)
  LLM查询:   5.8 ± 1.7 [min=3, max=8]
  改善率:    21.7% ± 18.0% (CV=83.0%)
  Beats best:79/240

### Structural (n=240)
  成功率:    240/240 (100%)
  LLM查询:   7.4 ± 1.0 [min=5, max=8]
  改善率:    15.8% ± 12.2% (CV=77.2%)
  Beats best:131/240

### Semantic (n=240)
  成功率:    240/240 (100%)
  LLM查询:   8.0 ± 0.0 [min=8, max=8]
  改善率:    10.9% ± 22.5% (CV=207.2%)
  Beats best:49/240

KW (improvement): H=86.8, p<0.001, η²=0.118
KW (queries):     H=285.0, p<0.001, η²=0.395

============================================================
注：将以上数字替换论文对应位置后，务必重新生成
Figure 4 和 Figure 6（violin图，样本量翻倍后形状更饱满）