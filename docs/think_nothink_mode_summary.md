# DPO/KTO Think/NoThink 模式实施总结

## 概述

为 Qwen3 模型添加了 `think` 和 `nothink` 两种推理模式支持，通过 `apply_chat_template` 的 `enable_thinking` 参数控制模型是否在回答前进行深度思考。

## 关键发现

### Qwen3 Chat Template 行为

```python
# enable_thinking=True (默认)
prompt 结尾: ...assistant\n
# 模型会生成 <think>...</think> 内容

# enable_thinking=False
prompt 结尾: ...assistant\n<think>\n\n</think>\n\n
# 模型跳过思考，直接输出答案
```

### TRL 框架限制

**重要**：TRL 的 `DPOTrainer` 和 `KTOTrainer` 在内部调用 `apply_chat_template` 时**不传递** `enable_thinking` 参数，因此：

- ❌ 不能在训练时动态切换 think/nothink
- ✅ 必须在数据准备阶段决定，生成两份独立的数据文件

## 实施内容

### 1. 数据准备脚本增强

**文件**：
- `data/prepare_dpo_data.py`
- `data/prepare_kto_data.py`

**新增参数**：
```bash
--enable_thinking           # 启用思考模式（默认关闭）
--both_thinking_modes       # 一次生成 think + nothink 两份数据
```

**使用示例**：
```bash
# 生成两份数据（推荐）
python data/prepare_dpo_data.py \
  --input_path data/within_sample_pairs.jsonl \
  --output_path data/dpo_train.parquet \
  --model_name_or_path Qwen3-4B \
  --both_thinking_modes

# 输出：
# - data/dpo_train_think.parquet + data/dpo_train_think_test.parquet
# - data/dpo_train_nothink.parquet + data/dpo_train_nothink_test.parquet
```

### 2. 训练脚本适配

**本地训练脚本**：
- `run_local_dpo.sh` - 第二个参数选择 think/nothink
- `run_local_kto.sh` - 同上

**用法**：
```bash
# 默认 nothink
bash run_local_dpo.sh experiment_name

# 指定 think 模式
bash run_local_kto.sh experiment_name think
```

**实验命名**：
```
DPO-Qwen3-4B-beta0.1-lr5e-7-bs4x4-len4096-nothink-exp_name
DPO-Qwen3-4B-beta0.1-lr5e-7-bs4x4-len4096-think-exp_name
```

### 3. Nebula 集群脚本

**Sweep 脚本**：
- `nebula_scripts/submit_dpo_sweep.sh`
- `nebula_scripts/submit_kto_sweep.sh`

**新增配置**：
```bash
THINK_MODES=("think" "nothink")  # 自动 sweep 两种模式
```

**Parametric 脚本**：
- `nebula_scripts/dpo_kto/dpo_parametric.sh`
- `nebula_scripts/dpo_kto/kto_parametric.sh`

**环境变量**：
```bash
THINK_MODE=think|nothink  # 由 sweep 脚本注入
```

**数据路径构建**：
```bash
TRAIN_DATA="${DATASET}/dpo_train_${THINK_MODE}.parquet"
EVAL_DATA="${DATASET}/dpo_train_${THINK_MODE}_test.parquet"
```

## 生成的数据文件

```
/data/oss_bucket_0/ad/kongyixian.kyx/dpo/dataset_d769a815e7a5/
├── dpo_train_think.parquet          (6380 samples)
├── dpo_train_think_test.parquet     (709 samples)
├── dpo_train_nothink.parquet        (6380 samples)
├── dpo_train_nothink_test.parquet   (709 samples)
├── kto_train_think.parquet          (5889 samples)
├── kto_train_think_test.parquet     (655 samples)
├── kto_train_nothink.parquet        (5889 samples)
└── kto_train_nothink_test.parquet   (655 samples)
```

## Nebula Sweep 配置

### DPO Sweep（2 jobs）
```bash
Job 1: DPO-Qwen3-8B-think-lr5e_7-beta0.1-sigmoid
Job 2: DPO-Qwen3-8B-nothink-lr5e_7-beta0.1-sigmoid
```

### KTO Sweep（2 jobs）
```bash
Job 1: KTO-Qwen3-8B-think-beta0.1-lr5e_7
Job 2: KTO-Qwen3-8B-nothink-beta0.1-lr5e_7
```

**训练配置**（4-GPU FSDP）：
- Batch Size: 2 per device × 4 GPUs × 4 grad_accum = 32 effective
- Max Length: 4096 tokens
- Learning Rate: 5e-7
- Epochs: 3
- DPO steps: 600 (6380 / 32 × 3)
- KTO steps: 555 (5889 / 32 × 3)

## 使用指南

### 本地训练

```bash
# DPO with nothink mode (default)
bash run_local_dpo.sh test_run

# DPO with think mode
bash run_local_dpo.sh test_run think

# KTO with nothink mode (default)
bash run_local_kto.sh test_run

# KTO with think mode
bash run_local_kto.sh test_run think
```

### Nebula 集群训练

```bash
# Dry-run 测试
bash nebula_scripts/submit_dpo_sweep.sh --dry-run
bash nebula_scripts/submit_kto_sweep.sh --dry-run

# 提交训练（自动生成 think + nothink 两个 job）
bash nebula_scripts/submit_dpo_sweep.sh
bash nebula_scripts/submit_kto_sweep.sh
```

## 对比实验设计

通过 think/nothink 两种模式的对比，可以评估：

1. **推理深度 vs 准确性**：think 模式是否提升答案质量
2. **训练效率**：think 模式增加 token 数量，是否影响训练速度
3. **泛化能力**：两种模式在不同任务上的表现差异
4. **推理成本**：部署时是否需要 think 模式

## 技术细节

### 数据准备流程

```
原始数据 (JSONL)
  ↓
prepare_*_data.py --both_thinking_modes
  ↓
[think variant]
  - apply_chat_template(enable_thinking=True)
  - prompt 结尾: ...assistant\n
  ↓
dpo/kto_train_think.parquet

[nothink variant]
  - apply_chat_template(enable_thinking=False)
  - prompt 结尾: ...assistant\n<think>\n\n</think>\n\n
  ↓
dpo/kto_train_nothink.parquet
```

### 训练数据流

```
Sweep Script
  ↓ THINK_MODES=("think" "nothink")
  ↓
Parametric Script
  ↓ THINK_MODE=think|nothink
  ↓ 构建数据路径
  ↓
Training Script
  ↓ 加载对应的 parquet 文件
  ↓
Model Training
```

## 验证测试

### Dry-run 验证
```bash
✓ DPO sweep: 2 jobs (think + nothink)
✓ KTO sweep: 2 jobs (think + nothink)
✓ 本地脚本: think/nothink 参数正确切换
✓ 数据路径: 正确构建 _think/_nothink 后缀
```

### 本地训练验证
```bash
✓ DPO nothink: 成功启动，加载 dpo_train_nothink.parquet
✓ KTO think: 成功启动，加载 kto_train_think.parquet
```

## 文件清单

### 数据准备（2 个）
- `data/prepare_dpo_data.py` - 增加 --both_thinking_modes
- `data/prepare_kto_data.py` - 增加 --both_thinking_modes

### 本地训练（2 个）
- `run_local_dpo.sh` - 第二个参数选择 think/nothink
- `run_local_kto.sh` - 同上

### Nebula 集群（4 个）
- `nebula_scripts/submit_dpo_sweep.sh` - 添加 THINK_MODES sweep
- `nebula_scripts/submit_kto_sweep.sh` - 添加 THINK_MODES sweep
- `nebula_scripts/dpo_kto/dpo_parametric.sh` - 支持 THINK_MODE 环境变量
- `nebula_scripts/dpo_kto/kto_parametric.sh` - 支持 THINK_MODE 环境变量

### 数据文件（8 个）
- `dpo_train_think.parquet` + test
- `dpo_train_nothink.parquet` + test
- `kto_train_think.parquet` + test
- `kto_train_nothink.parquet` + test

## 下一步

1. 提交 Nebula 训练（4 jobs: DPO think/nothink + KTO think/nothink）
2. 监控训练指标（SwanLab）
3. 对比 think vs nothink 的：
   - 训练 loss 曲线
   - 验证集 accuracy
   - 推理速度
4. 根据结果决定最终部署模式

## 注意事项

1. **数据一致性**：think/nothink 数据使用相同的 train/test split（seed=42）
2. **模型兼容性**：当前配置针对 Qwen3-4B（本地）和 Qwen3-8B（Nebula）
3. **显存预算**：think 模式增加 ~10% token 数量，但 max_length=4096 截断后影响可控
4. **评估指标**：建议在评估时也测试两种模式，对比推理性能
