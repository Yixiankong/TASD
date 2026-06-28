# DPO/KTO 本地测试反思报告

**测试时间**: 2026-06-26  
**测试环境**: 单卡 H20 GPU (140GB)  
**测试模型**: Qwen3-4B  
**测试数据**: 6380 训练样本 + 709 验证样本

---

## 📋 发现的问题及修复

### 问题 1: SwanLab 看不到指标 ❌

**现象**:
- SwanLab 已连接成功，但训练 5 步后被杀掉，看不到任何指标
- 日志显示 `logging_steps=10`，`logging_first_step=False`（默认值）

**根本原因**:
- TRL 默认配置 `logging_first_step=False`，第一步不记录指标
- `logging_steps=10` 意味着每 10 步才记录一次
- 训练在第一步完成前就被终止，导致没有任何指标上传

**修复方案** ✅:
```python
# training/dpo_train.py & training/kto_train.py
logging_first_step=True,  # 记录第一步，避免长时间无指标
logging_steps=5,          # 降低日志频率
```

**对 Nebula 的影响**:
- ✅ Nebula 脚本需要同步此修复，否则长时间训练前也会看不到指标
- 建议 Nebula 脚本使用 `--logging_steps 5 --logging_first_step`

---

### 问题 2: 显存溢出 + 训练时间不可接受 ❌

**现象**:
- 显存占用: 142GB / 143GB（99%）
- 训练速度: ~2 分钟/step
- 总步数: 1197 steps
- 预估总时间: **40 小时** ❌

**根本原因分析**:

1. **序列长度过长**:
   - `max_length=10240` tokens
   - 实际数据 prompt 平均 ~4800 tokens
   - 长序列导致显存爆炸

2. **DPO 双模型架构**:
   - Policy model + Reference model（2× Qwen3-4B）
   - 模型参数: 16GB × 2 = 32GB
   - Optimizer states (fp32 AdamW): ~32GB
   - Gradients + Activations: ~40-60GB
   - 总计: ~104-124GB

3. **Batch size 过小**:
   - `batch_size=2`, `grad_accum=8`
   - Effective batch size = 16
   - 导致梯度累积步数过多，训练时间长

**修复方案** ✅:
```bash
# run_local_dpo.sh & run_local_kto.sh
BATCH_SIZE=4          # 增大 batch size
GRAD_ACCUM=4          # 减少梯度累积
MAX_LENGTH=4096       # 截断长序列（从 10240 降到 4096）
```

**预期效果**:
- 显存占用: 90-100GB（降低 ~40GB）
- 训练速度: ~1 分钟/step
- 总步数: 597 steps（减少 50%）
- 预估总时间: ~10 小时 ✅

**对 Nebula 的影响**:
- ⚠️ **Nebula 4 卡 FSDP 需要重新计算显存预算**
- 单卡 4B 模型占用 ~100GB，8B 模型在 4 卡 FSDP 下：
  - 模型分片: 32GB / 4 = 8GB/卡
  - Optimizer 分片: 64GB / 4 = 16GB/卡
  - Activations: ~40GB/卡（不分片）
  - 总计: ~64GB/卡 ✅ 可行
- **建议 Nebula 参数**:
  ```bash
  BATCH_SIZE=2          # 多卡可以降低
  GRAD_ACCUM=4
  MAX_LENGTH=4096
  ```

---

### 问题 3: 数据格式兼容性 ⚠️

**现象**:
- 原始 JSONL 数据中 `prompt` 是 chat messages 列表
- TRL 期望 prompt 是字符串格式

**修复方案** ✅:
```python
# data/prepare_dpo_data.py & data/prepare_kto_data.py
# 添加 --model_name_or_path 参数
tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
```

**对 Nebula 的影响**:
- ✅ 数据准备阶段已在本地处理完成
- Nebula 脚本使用预处理好的 parquet 文件，无需额外处理
- **注意**: 如果更换数据集，需要重新运行数据准备脚本

---

### 问题 4: TRL 1.6.0 API 变更 ⚠️

**现象**:
- `DPOConfig` 没有 `max_prompt_length` / `max_completion_length` 参数
- 只有 `max_length`（总长度 = prompt + completion）

**修复方案** ✅:
```python
# 使用 max_length 替代
max_length=4096  # 总长度限制
truncation_mode="keep_start"  # 截断模式
```

**对 Nebula 的影响**:
- ⚠️ **Nebula 环境必须安装 TRL >= 1.6.0**
- 检查 `requirements_nebula.txt` 是否包含 `trl>=1.6.0`
- 如果 Nebula 使用旧版 TRL，需要升级

---

### 问题 5: SwanLab 初始化时序 ⚠️

**现象**:
- 在 `trainer.train()` 之前调用 `upload_data_to_swanlab` 会失败
- SwanLab 尚未初始化

**修复方案** ✅:
```python
# 将数据上传移到训练完成后
trainer.train()
if args.upload_data_to_swanlab:
    upload_data_to_swanlab(train_dataset, eval_dataset)
```

**对 Nebula 的影响**:
- ✅ Nebula 脚本已同步此修复
- 数据上传会在训练完成后自动执行

---

### 问题 6: Accelerate 配置兼容性 ⚠️

**现象**:
- `main_training_port` 参数不被当前 Accelerate 版本支持

**修复方案** ✅:
```yaml
# configs/accelerate/single_gpu.yaml & multi_gpu_fsdp.yaml
# 移除 main_training_port 参数
```

**对 Nebula 的影响**:
- ✅ Nebula 使用的 Accelerate 版本可能不同
- 建议 Nebula 脚本使用 `accelerate config default` 生成配置
- 或者在 parametric 脚本中动态生成配置

---

## 🎯 Nebula 脚本需要的关键调整

基于本地测试，Nebula 多卡脚本需要以下调整：

### 1. 超参数调整（dpo_parametric.sh & kto_parametric.sh）

```bash
# 原参数（基于本地单卡测试）
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_LENGTH=${MAX_LENGTH:-4096}

# 建议 Nebula 4 卡参数
BATCH_SIZE=${BATCH_SIZE:-2}        # 多卡可以降低
GRAD_ACCUM=${GRAD_ACCUM:-4}        # 减少梯度累积
MAX_LENGTH=${MAX_LENGTH:-4096}     # 保持 4096
LOGGING_STEPS=${LOGGING_STEPS:-5}
LOGGING_FIRST_STEP=true            # 关键：记录第一步
```

### 2. SwanLab 项目名调整

```bash
# Nebula 使用独立项目名，便于区分本地和云端实验
--report_to swanlab \
--swanlab_project TASD-DPO-Nebula \
--swanlab_run_name ${JOB_NAME}
```

### 3. 检查点保存策略

```bash
# Nebula 训练时间长，需要更频繁的保存
--save_steps 100 \
--save_total_limit 5 \
--save_on_each_node false
```

### 4. 断点续训支持

```bash
# Nebula 脚本需要支持断点续训
RESUME_FROM=${RESUME_FROM:-""}
if [ -n "$RESUME_FROM" ]; then
    ARGS="$ARGS --resume_from_checkpoint $RESUME_FROM"
fi
```

### 5. TRL 版本检查

```bash
# 在 parametric 脚本开头添加版本检查
python -c "import trl; assert tuple(map(int, trl.__version__.split('.')[:2])) >= (1, 6), 'TRL >= 1.6.0 required'"
```

---

## 📊 性能对比预测

| 配置 | 显存占用 | 训练速度 | 总步数 | 预估时间 | 可行性 |
|------|---------|---------|--------|---------|--------|
| **本地单卡 4B (原)** | 142GB | 2 min/step | 1197 | 40h | ❌ |
| **本地单卡 4B (优化)** | ~100GB | 1 min/step | 597 | 10h | ✅ |
| **Nebula 4卡 8B (预测)** | ~65GB/卡 | 0.5 min/step | 597 | 5h | ✅ |
| **Nebula 4卡 8B + LoRA** | ~50GB/卡 | 0.4 min/step | 597 | 4h | ✅✅ |

---

## 🔧 建议的后续步骤

### Phase 1: 本地验证（当前）
- [x] 修复 SwanLab 指标记录问题
- [x] 优化显存和训练时间
- [ ] 运行完整本地训练（10小时）
- [ ] 验证模型收敛性和指标正常

### Phase 2: Nebula 单节点测试
- [ ] 提交单节点 4 卡 4B 模型测试
- [ ] 验证 FSDP 配置正确性
- [ ] 检查 SwanLab 指标上传
- [ ] 确认检查点保存正常

### Phase 3: Nebula 完整训练
- [ ] 提交 4 卡 8B 模型训练
- [ ] 监控显存使用和训练速度
- [ ] 评估模型性能
- [ ] 与本地结果对比

---

## 📝 关键代码修复清单

### 已修复 ✅

- [x] `training/dpo_train.py`: 添加 `logging_first_step=True`
- [x] `training/kto_train.py`: 添加 `logging_first_step=True`
- [x] `run_local_dpo.sh`: 优化参数 (batch=4, accum=4, max_length=4096)
- [x] `run_local_kto.sh`: 优化参数 (batch=4, accum=4, max_length=4096)
- [x] `data/prepare_dpo_data.py`: 添加 chat template 处理
- [x] `data/prepare_kto_data.py`: 添加 chat template 处理

### Nebula 脚本待修复 ⏳

- [ ] `nebula_scripts/dpo_kto/dpo_parametric.sh`: 同步参数优化
- [ ] `nebula_scripts/dpo_kto/kto_parametric.sh`: 同步参数优化
- [ ] `nebula_scripts/dpo_kto/dpo_parametric.sh`: 添加 TRL 版本检查
- [ ] `nebula_scripts/dpo_kto/kto_parametric.sh`: 添加 TRL 版本检查
- [ ] `nebula_scripts/dpo_kto/dpo_parametric.sh`: 添加断点续训支持
- [ ] `nebula_scripts/dpo_kto/kto_parametric.sh`: 添加断点续训支持
- [ ] `requirements_nebula.txt`: 确认 `trl>=1.6.0`

---

## 💡 经验总结

1. **本地测试的重要性**:
   - 单卡测试暴露了显存和时间问题
   - 避免了 Nebula 上浪费计算资源

2. **日志配置的细节**:
   - `logging_first_step=True` 是关键
   - 避免长时间训练看不到指标

3. **TRL 版本管理**:
   - 不同版本 API 差异大
   - 需要明确版本要求

4. **数据预处理**:
   - Chat template 处理必须在数据准备阶段完成
   - 避免训练时动态转换

5. **参数调优策略**:
   - 先优化显存（max_length）
   - 再优化速度（batch_size, grad_accum）
   - 最后优化日志频率（logging_steps）

---

**报告生成时间**: 2026-06-26 02:10  
**下一步**: 运行优化后的本地训练，验证修复效果
