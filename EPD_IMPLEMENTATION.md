# EPD (Entropy-Preservation Distillation) 实现文档

## 概述

EPD 是一种新的熵保护蒸馏方法，旨在解决 CV-SDPO 的核心问题：
- **保护思考位置**：防止 teacher 压制学生的探索性 token（如 "Wait", "Let me think"）
- **简化计算**：只需 1 次 teacher forward（与 vanilla SDPO 相同）
- **自调节机制**：保护强度随学生进步自动衰减

## 核心改进

### 1. 代码质量改进

#### 1.1 消除代码重复
- **新增辅助函数** `_compute_v2sqrt_joint_score()`
  - 统一 v4 和 v5_epd 的 fallback 逻辑
  - 消除 20+ 行重复代码
  - 提高可维护性

#### 1.2 完善文档
- **添加详细 docstring** 到 `apply_teacher_entropy_weighting()`
  - 说明所有 weighting version 的行为
  - 明确 v5_epd 的特殊参数（epd_lambda, epd_tau）
  - 描述 sigmoid(0) bug 的修复

### 2. Nebula 提交脚本

#### 2.1 参数化训练脚本
**文件**: `nebula_scripts/sdpo/sdpo_epd_parametric.sh`

功能：
- 从环境变量读取 EPD 超参数（EPD_LAMBDA, EPD_TAU）
- 构建 Hydra 配置覆盖
- 调用 `verl.trainer.main_ppo` 启动训练

必需环境变量：
```bash
DATASET                    # 数据集路径
MODEL_NAME                 # 模型路径
LR                         # 学习率
ALPHA                      # KL 散度方向 (0.0=forward, 0.5=JSD, 1.0=reverse)
DONT_REPROMPT_ON_SELF_SUCCESS  # 是否跳过自我成功的 reprompt
EPD_LAMBDA                 # EPD 最大保护强度 (0.5/0.8/1.0)
EPD_TAU                    # EPD Sigmoid 温度 (0.3/0.5/1.0)
TRAIN_BATCH_SIZE           # 训练 batch size
ROLLOUT_N                  # Rollout 采样数
```

#### 2.2 超参扫描提交脚本
**文件**: `nebula_scripts/submit_sdpo_epd_sweep.sh`

功能：
- 扫描 EPD_LAMBDA × EPD_TAU 的所有组合
- 支持多数据集并行提交
- 自动生成唯一的 job name

扫描范围：
```bash
EPD_LAMBDA_LIST=(0.5 0.8 1.0)    # 3 个值
EPD_TAU_LIST=(0.3 0.5 1.0)       # 3 个值
DATASETS=(biology chemistry material physics)  # 4 个数据集
```

总实验数：3 × 3 × 4 = **36 个实验**

使用方法：
```bash
# Dry-run 模式（只打印，不提交）
bash nebula_scripts/submit_sdpo_epd_sweep.sh --dry-run

# 正式提交
bash nebula_scripts/submit_sdpo_epd_sweep.sh
```

## EPD 算法详解

### 核心公式

```python
# 1. 计算熵
H_student = -sum(student_probs * log(student_probs))  # (B, T)
H_teacher = -sum(teacher_probs * log(teacher_probs))  # (B, T)

# 2. 计算坍缩风险
collapse_risk = max(H_student - H_teacher, 0)  # ≥0
normalized_collapse = collapse_risk / (H_student + ε)  # ∈ [0, 1]

# 3. 计算保护权重
sigmoid_weights = 1 - λ * sigmoid(normalized_collapse / τ)

# 4. 修复 sigmoid(0) bug
confidence_weights = where(
    collapse_risk > 0,
    sigmoid_weights,
    ones  # 当 H_teacher > H_student 时完全不保护
)

# 5. 应用到 loss
final_loss = per_token_loss * confidence_weights
```

### 超参数说明

#### EPD_LAMBDA (最大保护强度)
- **取值范围**: [0, 1]
- **默认值**: 0.8
- **含义**: 保护的最大强度
  - λ=0.5: 轻度保护（权重范围 [0.75, 1.0]）
  - λ=0.8: 平衡保护（权重范围 [0.6, 1.0]）
  - λ=1.0: 全保护（权重范围 [0.5, 1.0]）

#### EPD_TAU (Sigmoid 温度)
- **取值范围**: (0, +∞)
- **默认值**: 0.5
- **含义**: 保护函数的锐度
  - τ=0.3: 尖锐保护（快速从高保护过渡到低保护）
  - τ=0.5: 平滑保护（默认）
  - τ=1.0: 近似线性保护

### 与 v1-v4 的对比

| 特性 | v1-v4 (Entropy Weighting) | v5_epd (EPD) |
|------|---------------------------|--------------|
| **目标** | 提升重要 token 的蒸馏权重 | 降低思考位置的蒸馏权重 |
| **归一化** | Softmax（序列内相对） | Sigmoid（独立） |
| **权重范围** | [0, +∞) | [1-λ, 1] |
| **计算开销** | 1× teacher forward | 1× teacher forward |
| **自调节** | ❌ 无 | ✅ 有 |

## 关键修复：sigmoid(0) Bug

### 问题
当 `H_teacher > H_student`（teacher 比 student 更不确定）时：
- `collapse_risk = 0`
- `normalized_collapse = 0`
- `sigmoid(0) = 0.5`
- 导致 `confidence_weights = 1 - λ*0.5`（例如 λ=0.8 时为 0.6）

这违背了 EPD 的设计初衷：当 teacher 不比 student 更确定时，应该完全不保护（权重=1.0）。

### 修复
```python
confidence_weights = torch.where(
    collapse_risk > 0,
    sigmoid_weights,
    torch.ones_like(sigmoid_weights)  # 完全不保护
)
```

### 验证
测试 3（边界情况）验证了修复的正确性：
```
场景 A (Student = Teacher):
  Collapse risk: 0.000000
  权重均值: 1.0000  ✓

场景 B (Student < Teacher):
  Collapse risk: 0.000000
  权重均值: 1.0000  ✓
```

## 监控指标

EPD 会记录以下指标到 SwanLab：

| 指标 | 含义 | 期望行为 |
|------|------|----------|
| `epd/student_entropy_mean` | Student 的平均熵 | 随训练缓慢下降 |
| `epd/teacher_entropy_mean` | Teacher 的平均熵 | 相对稳定 |
| `epd/collapse_risk_mean` | 平均坍缩风险 | 随训练下降（自调节） |
| `epd/normalized_collapse_mean` | 归一化坍缩风险 | ∈ [0, 1] |
| `response_length/mean` | 平均响应长度 | 保持稳定（不下降） |

## 预期效果

### 成功标准
1. **学生熵下降更慢**：相比 vanilla SDPO
2. **响应长度稳定**：不随训练显著缩短
3. **思考词保留**："Wait", "Let me think" 等 token 的频率不下降
4. **准确率持平或提升**：在 SciKnowEval 测试集上

### 调参建议

#### 如果学生熵仍然快速下降
- 增大 `EPD_LAMBDA`（0.8 → 1.0）
- 减小 `EPD_TAU`（0.5 → 0.3）

#### 如果学习太慢或准确率下降
- 减小 `EPD_LAMBDA`（0.8 → 0.5）
- 增大 `EPD_TAU`（0.5 → 1.0）

#### 如果保护太激进（权重分布双峰）
- 增大 `EPD_TAU`（0.5 → 1.0）

## 文件清单

### 修改的文件
1. `verl/trainer/ppo/core_algos.py`
   - 新增 `_compute_v2sqrt_joint_score()` 辅助函数
   - 改进 `apply_teacher_entropy_weighting()` 的 docstring
   - 优化 v4/v5_epd 的 fallback 逻辑

2. `verl/trainer/config/actor/actor.yaml`
   - 添加 `epd_lambda` 和 `epd_tau` 配置

3. `verl/workers/config/actor.py`
   - `SelfDistillationConfig` 新增 EPD 字段

### 新增的文件
1. `nebula_scripts/sdpo/sdpo_epd_parametric.sh`
   - EPD 参数化训练脚本

2. `nebula_scripts/submit_sdpo_epd_sweep.sh`
   - EPD 超参扫描提交脚本

3. `test_epd.py`
   - EPD 单元测试（4 个测试用例）

## 快速开始

### 1. 运行单元测试
```bash
cd /home/kongyixian.kyx/TASD
python test_epd.py
```

### 2. 本地测试（单个数据集）
```bash
export DATASET="sciknoweval/biology"
export MODEL_NAME="Qwen3-8B"
export LR="1e-5"
export ALPHA="0.5"
export DONT_REPROMPT_ON_SELF_SUCCESS="True"
export EPD_LAMBDA="0.8"
export EPD_TAU="0.5"
export TRAIN_BATCH_SIZE="32"
export ROLLOUT_N="8"

bash nebula_scripts/sdpo/sdpo_epd_parametric.sh
```

### 3. 提交 Nebula 扫描
```bash
# 先 dry-run 检查
bash nebula_scripts/submit_sdpo_epd_sweep.sh --dry-run

# 正式提交
bash nebula_scripts/submit_sdpo_epd_sweep.sh
```

## 下一步

1. **小规模验证**：在 biology 数据集上跑 1-2 个配置，验证 EPD 指标正常
2. **全量扫描**：提交 36 个实验的超参扫描
3. **结果分析**：
   - 对比不同 λ/τ 的效果
   - 选择最优配置
   - 在其他数据集上验证泛化性

## 参考

- EPD 设计方案：`.claude/plans/5-3-2-lazy-badger.md`
- CV-SDPO 问题分析：之前的对话记录
- 熵加权（v1-v4）实现：`verl/trainer/ppo/core_algos.py:apply_teacher_entropy_weighting()`
