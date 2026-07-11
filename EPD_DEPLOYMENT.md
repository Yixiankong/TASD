# EPD v5 部署验证报告

## ✅ 修复完成

### 发现的问题

1. **缺少环境变量设置**
   - PYTHONPATH, VLLM_USE_V1, WANDB_MODE, SWANLAB 配置等
   
2. **缺少关键训练配置**
   - `custom_reward_function.path` - 奖励函数路径
   - `actor_rollout_ref.rollout.val_kwargs.n=16` - 验证时采样数
   - `trainer.total_epochs=30` - 训练轮数
   - `trainer.total_training_steps=250` - 最大训练步数
   - `trainer.save_best_metric` - 最佳模型保存指标
   - `trainer.n_gpus_per_node=4` - GPU 数量
   - `algorithm.rollout_correction.rollout_is=token` - Token 级重要性采样修正
   
3. **缺少包安装步骤**
   - `pip install -e .` 安装当前项目

### 修复内容

重写了 `sdpo_epd_parametric.sh`，使其与 `sdpo_sciknoweval_parametric.sh` 保持一致，只增加 EPD 特有的 4 个配置项：

```bash
actor_rollout_ref.actor.self_distillation.entropy_weighting=True
actor_rollout_ref.actor.self_distillation.entropy_weighting_version=v5_epd
actor_rollout_ref.actor.self_distillation.epd_lambda=${EPD_LAMBDA}
actor_rollout_ref.actor.self_distillation.epd_tau=${EPD_TAU}
```

## 📊 配置对比

| 配置项 | Baseline | EPD | 说明 |
|--------|----------|-----|------|
| `entropy_weighting` | ❌ 未设置 | ✅ True | 启用熵加权 |
| `entropy_weighting_version` | ❌ 未设置 | ✅ v5_epd | 使用 EPD v5 算法 |
| `epd_lambda` | ❌ 未设置 | ✅ ${EPD_LAMBDA} | 保护强度超参 |
| `epd_tau` | ❌ 未设置 | ✅ ${EPD_TAU} | 保护时间超参 |
| 其他所有配置 | ✅ | ✅ | 与 baseline 完全一致 |

## 🧪 验证步骤

### 1. 单元测试（已完成）

```bash
cd /home/kongyixian.kyx/TASD
python test_epd.py
```

**预期输出**：
```
✓✓✓ 所有测试通过 ✓✓✓
```

### 2. 语法检查（已完成）

```bash
bash -n nebula_scripts/sdpo/sdpo_epd_parametric.sh
bash -n nebula_scripts/submit_sdpo_epd_sweep.sh
```

**预期输出**：无错误

### 3. 参数扫描预览

```bash
bash nebula_scripts/submit_sdpo_epd_sweep.sh --dry-run
```

**预期输出**：
- 36 个训练任务
- 4 数据集 × 3 λ值 × 3 τ值
- 每个任务显示正确的环境变量

### 4. 小规模验证（推荐先执行）

提交单个任务验证整个流程：

```bash
# 临时修改 submit_sdpo_epd_sweep.sh，只保留一个配置
DATASETS=("sciknoweval/biology")
EPD_LAMBDA_LIST=("0.8")
EPD_TAU_LIST=("0.5")

# 提交
bash nebula_scripts/submit_sdpo_epd_sweep.sh
```

**验证要点**：
- ✅ Nebula 任务成功提交
- ✅ 训练正常启动（无导入错误）
- ✅ SwanLab 中出现 EPD 指标：
  - `epd/student_entropy_mean`
  - `epd/teacher_entropy_mean`
  - `epd/collapse_risk_mean`
  - `epd/normalized_collapse_mean`
- ✅ 响应长度不会快速缩短
- ✅ 思考词（如 "Wait", "Let me think"）保留

### 5. 全量扫描

确认小规模验证通过后，恢复完整的参数扫描配置并提交：

```bash
# 恢复完整配置
DATASETS=("sciknoweval/biology" "sciknoweval/chemistry" "sciknoweval/material" "sciknoweval/physics")
EPD_LAMBDA_LIST=("0.5" "0.8" "1.0")
EPD_TAU_LIST=("0.3" "0.5" "1.0")

# 提交全部 36 个任务
bash nebula_scripts/submit_sdpo_epd_sweep.sh
```

## 📈 监控指标

### SwanLab 中应关注的关键指标

#### 1. EPD 核心指标

| 指标 | 含义 | 期望行为 |
|------|------|----------|
| `epd/student_entropy_mean` | 学生模型熵 | 缓慢下降或保持稳定 |
| `epd/teacher_entropy_mean` | 教师模型熵 | 相对稳定 |
| `epd/collapse_risk_mean` | 熵坍缩风险 | 随训练逐渐降低（自调节） |
| `epd/normalized_collapse_mean` | 归一化坍缩风险 | 在 [0, 1] 范围内 |

#### 2. 训练健康指标

| 指标 | 含义 | 期望行为 |
|------|------|----------|
| `critic/score/mean` | 平均得分 | 逐步上升 |
| `response_length/mean` | 平均响应长度 | 不会快速缩短 |
| `val/accuracy/*` | 验证集准确率 | 逐步上升 |

#### 3. 与 Baseline 对比

对比相同数据集上的 SDPO baseline 和 EPD 实验：
- EPD 的学生熵下降应更慢
- EPD 的响应长度应更稳定
- EPD 的最终准确率应 ≥ baseline

## 🔧 故障排查

### 问题 1：训练启动失败

**可能原因**：
- 缺少依赖包
- 模型路径错误
- GPU 资源不足

**排查步骤**：
```bash
# 检查 Nebula 日志
nebulactl logs <job_id>

# 本地测试参数脚本
bash nebula_scripts/sdpo/sdpo_epd_parametric.sh
```

### 问题 2：EPD 指标缺失

**可能原因**：
- `entropy_weighting_version` 未设置为 `v5_epd`
- `student_topk_log_probs` 未传递

**排查步骤**：
```bash
# 检查配置
grep "entropy_weighting" nebula_scripts/sdpo/sdpo_epd_parametric.sh

# 确认代码路径
grep "v5_epd" verl/trainer/ppo/core_algos.py
```

### 问题 3：学生熵仍然快速下降

**可能原因**：
- λ 值太小，保护不足
- τ 值太大，保护不够敏锐

**调整建议**：
```bash
# 尝试更强的保护
EPD_LAMBDA_LIST=("1.0")
EPD_TAU_LIST=("0.3")
```

## 📝 下一步行动

### 立即执行

1. ✅ **小规模验证**（1 个任务）
   ```bash
   # 修改配置为单点测试
   bash nebula_scripts/submit_sdpo_epd_sweep.sh
   ```

2. ✅ **检查结果**
   - 等待训练完成（约 2-4 小时）
   - 查看 SwanLab 指标
   - 对比 baseline

3. ✅ **全量扫描**（36 个任务）
   ```bash
   # 恢复完整配置
   bash nebula_scripts/submit_sdpo_epd_sweep.sh
   ```

### 后续分析

1. **超参数敏感性分析**
   - 哪个 λ 值效果最好？
   - 哪个 τ 值效果最好？
   - 不同数据集是否需要不同的超参数？

2. **消融实验**
   - EPD vs 无保护（baseline）
   - EPD vs 固定保护（不随训练调整）
   - EPD vs 其他保护策略

3. **论文撰写**
   - 方法描述
   - 实验设置
   - 结果分析
   - 消融实验

## 🎯 成功标准

### 短期目标（单点验证）

- [ ] 训练正常完成（无崩溃）
- [ ] EPD 指标正常记录
- [ ] 学生熵下降速度 < baseline
- [ ] 响应长度稳定（不快速缩短）

### 中期目标（全量扫描）

- [ ] 至少一个配置在 4 个数据集上都优于 baseline
- [ ] 找到最优的 (λ, τ) 超参数组合
- [ ] 验证 EPD 的跨数据集泛化性

### 长期目标（论文发表）

- [ ] EPD 显著提升推理任务性能
- [ ] 消融实验证明各组件的有效性
- [ ] 分析和解释 EPD 的工作机制

## 📚 相关文档

- [EPD 设计方案](docs/epd/entropy_preservation_distillation.md)
- [EPD 实现细节](docs/epd/implementation_details.md)
- [EPD 测试报告](docs/epd/test_report.md)
- [EPD 实现总结](EPD_IMPLEMENTATION.md)

---

**最后更新**: 2026-07-09  
**状态**: ✅ 准备部署
