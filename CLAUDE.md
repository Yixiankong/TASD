# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

TASD 是基于 [verl](https://github.com/volcengine/verl)（v0.7.0-dev）框架的 RLHF/RLVR 训练系统，实现了多种策略优化算法：
- **GRPO** — 基线 Group Relative Policy Optimization
- **SDPO** — Self-Distilled Policy Optimization（自蒸馏策略优化）
- **CV-SDPO** — 带答案泄漏去污染（Control Variate）的 SDPO
- **TASD** — Token-level Advantage Self-Distillation

以 editable mode 安装（`pip install -e .`），包名 `verl`。

## 关键约束

- **绝对不要在未经用户明确同意的情况下修改训练环境**（`pip install`、升级 transformers 等）。之前未经授权的 transformers 升级导致 Qwen3.5 兼容性问题。
- 当前硬件：4× NVIDIA H20-3e GPU（每卡 ~143GB VRAM）
- 当前模型：Qwen3-8B（36 decoder layers，8.19B params）

## 构建与运行

### 安装
```bash
pip install -e .  # editable mode，已安装
```

### 启动训练
所有训练通过 Hydra 配置系统启动，底层调用 `python -m verl.trainer.main_ppo`：
```bash
# 本地训练（通过 wrapper 脚本）
bash run_local_grpo.sh              # GRPO 基线
bash run_local_sdpo.sh              # SDPO
bash run_local_cv_sdpo.sh           # CV-SDPO（1 GPU, Qwen3-4B 验证用）
bash run_local_test.sh              # 测试

# 远程提交（Nebula 集群）
bash nebula_scripts/submit_cv_sdpo_sweep.sh [--dry-run]       # CV-SDPO 4卡
bash nebula_scripts/submit_sdpo_baseline_sweep.sh [--dry-run] # SDPO baseline 4卡
bash nebula_scripts/submit_grpo_baseline_sweep.sh [--dry-run] # GRPO baseline 4卡
```

训练脚本最终调用 `training/verl_training.sh`，它设置环境变量后执行：
```bash
python -m verl.trainer.main_ppo --config-name $CONFIG_NAME "$@"
```

### Hydra 配置
配置文件在 `verl/trainer/config/`，按算法分：
- `ppo_trainer.yaml` — 基础 PPO 配置（所有算法的 defaults）
- `baseline_grpo.yaml` — GRPO
- `sdpo.yaml` — SDPO
- `cv_sdpo.yaml` — CV-SDPO（含 leakage decontamination 配置）
- `tasd.yaml` — TASD

命令行覆盖用 Hydra 点号语法：`data.train_batch_size=32 actor_rollout_ref.rollout.n=8`

### Lint
```bash
ruff check .          # lint（配置在 pyproject.toml）
ruff format .         # 格式化
```

### 测试
```bash
pytest tests/                                    # 全部测试
pytest tests/test_protocol_on_cpu.py             # 单个测试文件
pytest tests/trainer/ -k "test_name"             # 按名称过滤
```

## 代码架构

### 训练循环核心

```
verl/trainer/main_ppo.py          ← Hydra 入口点
  └─ verl/trainer/ppo/ray_trainer.py  ← RayPPOTrainer：主训练循环
       ├─ .fit()                       每步：rollout → reward → advantage → update
       ├─ ._maybe_build_self_distillation_batch()  构建 SDPO/CV-SDPO teacher context
       └─ ._validate()                验证循环
```

### 训练步骤数据流（每个 training step）

1. **Rollout**：vLLM 引擎生成 on-policy responses
2. **Reward**：`verl/trainer/ppo/reward.py` 调用 reward function 计算分数
3. **Self-Distillation Batch**（SDPO/CV-SDPO）：
   - 识别 peer success rollouts → 构建 teacher prompt（含正确答案 context）
   - CV-SDPO 额外构建 base prompt 和 answer-only prompt
4. **Teacher Forward**（TASD/SDPO）：
   - TASD：`actor_rollout_wg.compute_teacher_log_probs()` 计算 teacher token-level rewards
   - SDPO：在 `update_actor` 之前 pre-compute 3× teacher forward（Plan B OOM fix）
5. **Advantage**：`verl/trainer/ppo/core_algos.py` 中的各算法（GRPO/TASD/GAE 等）
6. **Actor Update**：`verl/workers/actor/dp_actor.py` 反向传播

### 关键文件

| 文件 | 职责 |
|------|------|
| `verl/trainer/ppo/ray_trainer.py` | 主训练循环，编排所有组件 |
| `verl/trainer/ppo/core_algos.py` | 所有损失函数和 advantage 计算（GRPO、TASD、SDPO distillation loss、CV-SDPO clean target） |
| `verl/workers/actor/dp_actor.py` | Actor 前向/反向，含 teacher forward 和 self-distillation loss 计算 |
| `verl/workers/fsdp_workers.py` | FSDP worker 封装，`compute_teacher_log_probs` 的入口 |
| `verl/trainer/ppo/reward.py` | Reward function 加载和计算 |
| `verl/utils/reward_score/feedback/` | 各任务的 reward function 实现（mcq.py, math.py 等） |

### 算法实现细节

**SDPO（loss_mode="sdpo"）**：
- `dp_actor.py` 中 `self_distillation_enabled=True` 时，pg_loss = `compute_self_distillation_loss()`
- 损失是 KL 散度（student vs teacher 分布），**不使用 advantages/rewards**
- 支持 top-K distillation（`distillation_topk`）+ add_tail 或 renorm

**CV-SDPO（leakage decontamination）**：
- `compute_cv_sdpo_clean_target()`：用 control variate 从 teacher target 中去除答案泄漏
- 公式：`log_q_clean = log_p0 + γ × (Δ_full - β × Δ_ans)`
- 需要 3 次 teacher forward：full context / base (无 context) / answer-only
- 关键配置：`cv_gamma`、`beta_mode`（adaptive/fixed）、`beta_max`
- 诊断指标：`cv_sdpo/cos_full_ans`、`cv_sdpo/cos_clean_ans`、`cv_sdpo/beta_mean`

**TASD（loss_mode="tasd"）**：
- Token-level reward 由 `compute_tasd_token_rewards()` 计算（多种 reward_type）
- Advantage 由 `compute_tasd_advantage()` 计算（group mean normalization）
- PPO clipped loss 使用这些 advantages

### 命名约定

- `ref_module` = `teacher_module`（同一对象，只是别名）
- `_compute_ref_log_prob` → KL penalty（非 CV-SDPO 使用）
- `compute_teacher_log_probs` → teacher 蒸馏（SDPO/TASD Plan B 共用入口）
- `old_log_probs` = rollout 时的 policy log probs
- `token_level_scores` = verifier/reward 原始分数
- `token_level_rewards` = 经 KL penalty/TASD reward 处理后的最终 rewards

### 远程提交（Nebula）

```
nebula_scripts/
├── entry.py                         ← Nebula 作业入口
├── cluster_gpu_4.json              ← 4-GPU 集群配置
├── submit_*_sweep.sh               ← 各算法的超参扫描提交脚本
├── sdpo/
│   ├── cv_sdpo_sciknoweval_parametric.sh  ← CV-SDPO 实际训练脚本（独立）
│   └── sdpo_sciknoweval_parametric.sh     ← SDPO baseline 实际训练脚本
└── tasd/                           ← TASD 训练脚本
```

提交脚本通过 `nebulactl run mdl` 提交，环境变量传递超参数。

**CV-SDPO vs SDPO baseline 的区别**：
- CV-SDPO 使用独立脚本 `cv_sdpo_sciknoweval_parametric.sh`，显式启用 `leakage_decontamination`
- CV-SDPO sweep 额外传递 OOM 防护环境变量：`PPO_MINI_BATCH_SIZE=16`（SDPO 默认 32），`GPU_MEMORY_UTIL=0.75`（SDPO 默认 0.85）
- 这些 env vars 通过 `${VAR:-default}` 语法在 parametric 脚本中消费，SDPO baseline 不传则使用默认值

**关键环境变量**（通过 `--env=` 传递）：

| 变量 | SDPO baseline | CV-SDPO | 说明 |
|------|:---:|:---:|------|
| `CONFIG_NAME` | `sdpo` | `cv_sdpo` | Hydra config 名称 |
| `PPO_MINI_BATCH_SIZE` | 32（默认） | 16 | CV-SDPO 的 3× teacher forward 需更多显存余量 |
| `GPU_MEMORY_UTIL` | 0.85（默认） | 0.75 | 降低 10% 为 teacher activation 留余量 |
| `CV_GAMMA` | — | 0.5 | clean shift 强度 |
| `BETA_MODE` | — | adaptive | 自适应/固定 β |

### 日志和监控

训练日志输出到多个 logger：console, swanlab, wandb, tensorboard。
- SwanLab: `https://swanlab.cn/@kongyixian/TASD/`
- W&B: offline 模式，日志在 `$LOG_ROOT/logs/wandb/`
- TensorBoard: `$LOG_ROOT/logs/tensorboard/`
- Best checkpoint 按 `trainer.save_best_metric` 保存
- Rollout 数据保存在 `$LOG_ROOT/logs/rollout_data/` 下（JSONL 格式，每步一个文件）

### 工具脚本

| 脚本 | 用途 |
|------|------|
| `scripts/diagnose.py` | 训练诊断工具 |
| `scripts/rollout_viewer.py` | Rollout 数据查看器 |
| `scripts/print_cfg.py` | 打印 Hydra 配置 |

## CV-SDPO 已知问题与分析

### 4 卡并行策略
- **FSDP FULL_SHARD**：4 GPU 组成 1D device_mesh，参数/optimizer/梯度各切 4 份
- **vLLM**：`tensor_model_parallel_size=1`，每 GPU 独立推理（与 FSDP 交替使用显存）
- **共址部署**：Actor + Rollout + Ref/Teacher 在同一组 4 GPU 上（Ray `max_colocate_count=3`）
- **Teacher CPUOffload**：ref_module FSDP 配置了 `CPUOffload(offload_params=True)`

### Qwen3-8B 每 GPU 内存预算 (FULL_SHARD, 假设 96GB)
| 项目 | 每 GPU |
|------|--------|
| 模型参数 (bf16, sharded) | 3.8 GB |
| Optimizer (fp32, sharded) | 15.2 GB |
| 梯度 (bf16, sharded) | 3.8 GB |
| 基础合计 | **22.8 GB** |
| 可用于 activation + KV cache | **~73 GB** |

### 实验分析发现（2026-06）
- **Peer solution 幻觉**：同一题目的不同成功 rollout 可能给出相互矛盾的推理路径（如蛋白质 ID 识别不一致），过程信号不可靠
- **空洞推理**：部分 peer solution 承认"无法确定答案"但仍输出正确答案，Δ_full 几乎全是泄漏
- **Answer-only prompt 信号弱**：`"The answer is C."` 对需要专业知识的题目几乎不携带有用信息，导致 `cos_full_ans` 偏低 (~0.34)
- **clean target 接近 base**：`H_clean ≈ H_base`，说明去污染后蒸馏信号弱
- **诊断指标参考**：`cos_clean_ans` 应接近 0（残差正交），实际 ~0.15-0.18（偏高，说明 β 不够或 CV 设计需调整）
