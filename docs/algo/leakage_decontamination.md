# CV-SDPO: Answer-Leakage Decontaminated Self-Distillation Policy Optimization

> 基于 SDPO（arXiv:2601.20802）框架的答案泄漏去污染方法。
> 面向 SciKnowEval L3 推理子集及 ToolUse 任务。

---

## 1. SDPO 基础框架

### 1.1 核心思想

SDPO（Self-Distillation Policy Optimization）在标准 GRPO 的基础上，额外利用 **同一模型在不同 context 下的分布差异** 作为 dense token-level 监督信号。核心流程：

1. 当前策略 π_θ 采样 on-policy rollout
2. Verifier 打分，识别成功/失败 rollout
3. 用成功 rollout 构建 **self-teacher prompt**（将正确答案/推理过程注入 context）
4. Teacher 在 feedback-conditioned context 下重新评估 rollout 的 token log-prob
5. Teacher-Student 分布差距形成 dense 蒸馏损失

### 1.2 实现文件

| 文件 | 职责 |
|------|------|
| `verl/trainer/ppo/ray_trainer.py` | 主训练循环、teacher context 构建、答案提取 |
| `verl/workers/actor/dp_actor.py` | Student/Teacher forward pass、梯度更新 |
| `verl/trainer/ppo/core_algos.py` | `compute_cv_sdpo_clean_target()`、`compute_self_distillation_loss()` |
| `verl/workers/fsdp_workers.py` | FSDP 分布式 worker：rollout/training 模式切换 |
| `verl/workers/config/actor.py` | `SelfDistillationConfig` dataclass |
| `verl/trainer/config/cv_sdpo.yaml` | CV-SDPO Hydra 配置 |

---

## 2. 训练 Pipeline 详解

### 2.1 总体流程

```
ray_trainer._train_step()
│
├── 1. Rollout (vLLM)
│     gen_batch_output = actor_rollout_wg.generate_sequences(gen_batch)
│
├── 2. Reward
│     reward_tensor, reward_extra_infos_dict = _compute_or_extract_reward(batch)
│
├── 3. 构建 Self-Distillation Batch
│     self_distillation_data = _maybe_build_self_distillation_batch(batch, reward_tensor)
│     → batch.union(self_distillation_batch)  # 追加 teacher 相关 tensor
│
├── 4. Recompute old_log_probs (FSDP)
│     old_log_prob = actor_rollout_wg.compute_log_prob(batch)
│
├── 5. Compute Advantage (GRPO)
│     batch = compute_advantage(batch, adv_estimator="grpo", ...)
│
├── 5.5 Pre-compute Teacher Log-Probs (Plan B, FSDP)
│     sdpo_teacher_result = actor_rollout_wg.compute_teacher_log_probs(batch)
│     → batch["precomputed_*"] = teacher_result
│     （teacher activation 在此释放，不与梯度共存）
│
└── 6. Update Actor (FSDP)
      actor_output = actor_rollout_wg.update_actor(batch)
      → dp_actor.update_policy()
        ├── Student forward（有梯度, topk_indices=precomputed）
        ├── 读取 precomputed teacher tensors（跳过 teacher forward）
        ├── CV 去污染
        ├── Loss 计算
        ├── Backward + optimizer step
        └── Teacher EMA 更新
```

### 2.2 Step 1: vLLM Batch Rollout

**代码路径**: `ray_trainer.py:1997-2001` → `fsdp_workers.py:983-1029`

Hybrid Engine 架构下，Actor 和 Rollout 共享同一组 GPU。生成阶段：

1. `fsdp_workers.rollout_mode()` — 将 FSDP actor 权重 offload，腾出显存给 vLLM
2. `self.rollout.generate_sequences(prompts)` — vLLM 引擎执行 batch 生成，每个 prompt 采样 `n=8` 条 response
3. `fsdp_workers.trainer_mode()` — 切回训练模式，重新加载 FSDP 权重

```python
# ray_trainer.py:1990-2001
gen_batch_output = gen_batch.repeat(repeat_times=n, interleave=True)  # 每题复制 n 份
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
```

**关键配置**:
- `rollout.n = 8` — 每题采样数
- `rollout.temperature` — 采样温度
- `rollout.calculate_log_probs = True` — 生成时同步计算 rollout log-prob

### 2.3 Step 2: Reward 计算

**代码路径**: `ray_trainer.py:2063-2081`

支持两种模式：
- **Rule-based reward**：通过 `reward_fn` 计算（SciKnowEval 用 exact match）
- **Reward model**：通过 `rm_wg.compute_rm_score(batch)` 计算

输出 `reward_tensor: (batch_size, response_length)`，其中只在最后一个 token 位置有非零值（sequence-level reward）。

### 2.4 Step 3: 构建 Self-Distillation Batch

**代码路径**: `ray_trainer.py:786-1014`

#### 2.4.1 收集成功 Rollout

```python
# ray_trainer.py:717-724
success_by_uid = _collect_solutions_by_uid(batch, reward_tensor, threshold=1.0)
# 输出: {uid: [成功rollout的index列表]}
```

同一 `uid`（同一题）下所有 score >= threshold 的 rollout 索引。

#### 2.4.2 选择 Peer Solution

```python
# ray_trainer.py:748-784
solution = _get_solution(idx, success_by_uid, uids, response_texts, ...)
```

对每个 rollout `idx`：
- 从同 uid 的成功 rollout 中 **随机选一条** peer rollout 作为 solution
- `dont_reprompt_on_self_success=True` 时排除自身
- 无成功 peer → 返回 `None`（该样本不参与 SDPO loss）

#### 2.4.3 构建 3 种 Teacher Context（CV-SDPO）

对每个 rollout，构建 3 种不同 context 的 teacher 输入：

| Context | 构建方法 | 包含信息 |
|---------|----------|----------|
| **Full teacher** | `_build_teacher_message()` | 原始题目 + 成功 peer rollout 的完整解答 |
| **Answer-only teacher** | `_build_answer_only_message()` | 原始题目 + 仅答案字母（如 "The answer is C."） |
| **Base teacher** | `_build_base_message()` | 原始题目（无任何特权信息） |

**答案提取** (`_extract_answer_letter_from_response`, `ray_trainer.py:732-746`):
1. 优先 XML 匹配: `<answer>\s*([A-D])\s*</answer>`
2. Fallback: 文本中最后一个独立的 A/B/C/D

**Tokenize & 拼接** (`ray_trainer.py:875-969`):
```python
# 将 teacher prompt 与 rollout response 拼接
teacher_input_ids = cat([teacher_prompt["input_ids"], responses], dim=1)
teacher_attention_mask = cat([teacher_prompt["attention_mask"], response_mask], dim=1)
teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)
```

三种 context 各自生成一组 `(input_ids, attention_mask, position_ids)` tensors。

#### 2.4.4 Self-Distillation Mask

```python
# ray_trainer.py:977-981
self_distillation_mask = tensor([
    solution_strs[i] is not None or feedback_used[i]
    for i in range(batch_size)
])
# shape: (batch_size,)，1.0 = 有 teacher context，参与 SDPO loss
#                        0.0 = 无成功 peer，跳过 SDPO loss
```

### 2.5 Step 4: Recompute old_log_probs (FSDP)

**代码路径**: `ray_trainer.py:2097-2115` → `fsdp_workers.py:1033-1079`

用 FSDP actor 模型重新计算当前策略的 log-prob（而非复用 vLLM rollout 时的 log-prob），作为 PPO clip 的 anchor：

```python
old_log_prob = actor_rollout_wg.compute_log_prob(batch)  # FSDP forward, no_grad
batch = batch.union(old_log_prob)  # 添加 old_log_probs, entropys
```

**为什么不直接用 rollout log-prob？**
- vLLM rollout 用的是 rollout engine（可能有精度差异）
- Recompute 保证 old_log_probs 和训练时的 forward 完全一致（同一 FSDP 模型）

### 2.6 Step 5: Advantage 计算

**代码路径**: `ray_trainer.py:2442-2454`

使用 GRPO advantage estimator：

```python
batch = compute_advantage(batch, adv_estimator="grpo", num_repeat=n, ...)
```

GRPO 按 uid 分组，在同一题的 n 条 rollout 内做 sequence-level 归一化：
- `advantages = (scores - mean) / std`（每个 uid group 内）
- `norm_adv_by_std_in_grpo = False` 时不除以 std

### 2.7 Step 6: Actor 更新（FSDP Training）

**代码路径**: `ray_trainer.py:2510-2513` → `fsdp_workers.py:937-979` → `dp_actor.py:769-1070`

这是最核心的步骤，包含 student/teacher forward、loss 计算和参数更新。

---

## 3. Loss 计算详解

### 3.1 Forward Pass 架构

在 `dp_actor.update_policy()` 中，每个 micro-batch 依次执行：

```
┌─────────────────────────────────────────────────────┐
│               dp_actor.update_policy()               │
│                                                      │
│  1. Student forward (有梯度)                          │
│     → log_prob, student_topk_logps, topk_indices     │
│                                                      │
│  2. Full teacher forward (no_grad)                   │
│     → teacher_log_prob, teacher_topk_logps           │
│     （复用 student 的 topk_indices 做 gather）         │
│                                                      │
│  3. Base teacher forward (no_grad, CV-SDPO only)     │
│     → base_teacher_topk_logps                        │
│     （复用 student 的 topk_indices 做 gather）         │
│                                                      │
│  4. Answer-only teacher forward (no_grad, CV-SDPO)   │
│     → answer_teacher_topk_logps                      │
│     （复用 student 的 topk_indices 做 gather）         │
│                                                      │
│  5. compute_cv_sdpo_clean_target()                   │
│  6. compute_self_distillation_loss()                 │
│  7. loss.backward() + optimizer.step()               │
│  8. _update_teacher()  (EMA)                         │
└─────────────────────────────────────────────────────┘
```

### 3.2 Top-K 选择策略

**Top-K 在 student 上选取，teacher 复用 student 的 indices。**

遵循 SDPO 论文 Appendix A.3 的设计（`dp_actor.py:681` 注释说明）。

**原理**: 蒸馏目标是修正 student 当前概率最高的那些 token 的分布。在 student 高概率区域做对齐，梯度信号最有效。在 teacher 高概率但 student 低概率的区域做对齐，student 难以学到。

**实现** (`dp_actor.py:176-562`, `_forward_micro_batch`):

```python
# Student 路径: distill_topk=100, topk_indices=None → 自己选 topk
if topk_indices is None:
    topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)

# Teacher 路径: distill_topk=None, topk_indices=student_topk_indices → 只做 gather
else:
    topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
```

**配置**: `distillation_topk: 100`（在 vocab 的 ~152K tokens 中取 top-100）

### 3.3 KL 散度选择：Reverse KL

CV-SDPO 使用 **Reverse KL**（`alpha=1.0`）：

$$\mathcal{L} = \text{KL}(\text{student} \| \text{teacher}) = \sum_v S(v) \log \frac{S(v)}{T(v)}$$

**代码** (`core_algos.py:1633-1636`):
```python
if self_distillation_config.alpha == 1.0:
    kl_loss = F.kl_div(
        teacher_distill_log_probs,   # input (被当作 log Q)
        student_distill_log_probs,   # target (被当作 log P)
        reduction="none", log_target=True
    )
    # 注: PyTorch kl_div(input, target) = target * (log_target - input)
    # 即 KL(target || input) = KL(student || teacher)
```

**alpha 参数的三种模式**:

| alpha | 散度 | 行为 | 适用场景 |
|-------|------|------|----------|
| 0.0 | Forward KL: KL(T ‖ S) | Mode-covering：student 覆盖 teacher 所有模式 | 传统知识蒸馏 |
| 0.5 | JSD | 对称混合 | 平衡 |
| **1.0** | **Reverse KL: KL(S ‖ T)** | **Mode-seeking：student 聚焦自身高概率区域** | **CV-SDPO** |

**CV-SDPO 选择 Reverse KL 的理由**:
- Student 只在自己已有高概率的 token 上对齐 teacher → 避免被 teacher 中的噪声分布（尤其是去污染后可能引入的伪影）拉偏
- 与 top-K on student 的策略一致：student 主导对齐方向
- Mode-seeking 特性适合 RL 场景：强化 student 已有的正确模式，而非强迫覆盖所有模式

### 3.4 Top-K 尾部处理 (add_tail)

**配置**: `distillation_add_tail: true`

由于只取 top-K token，剩余 vocab 的概率质量需要处理。`add_tail` 将剩余概率打包为一个尾部桶：

```python
# core_algos.py:1585-1591
def add_tail(log_probs):  # log_probs: (batch, seq_len, K)
    log_s = logsumexp(log_probs, dim=-1, keepdim=True)  # top-K 概率之和
    log_s = clamp(log_s, max=-1e-7)                     # 确保 < 1
    tail_log = log(1 - exp(log_s))                       # 尾部概率 = 1 - sum(topK)
    return cat([log_probs, tail_log], dim=-1)             # (batch, seq_len, K+1)
```

最终 KL 散度在 K+1 维分布上计算，保留了完整的概率质量。

### 3.5 IS Ratio Clipping

**配置**: `is_clip: 2.0`

当 student 策略在 mini-batch 训练中偏离 old policy 时，使用 importance sampling ratio 修正：

```python
# core_algos.py:1659-1667
ratio = exp(student_log_probs - old_log_probs).clamp(max=is_clip)
per_token_loss = per_token_loss * ratio
```

- `ratio > 1` 时说明 student 比 old policy 更倾向该 token → 加大该 token 的蒸馏力度
- `clamp(max=2.0)` 防止 ratio 爆炸

### 3.6 Loss 聚合

**配置**: `loss_agg_mode: "token-mean"`（默认）

```python
# core_algos.py:1232-1235
loss = masked_sum(per_token_loss, loss_mask) / batch_num_tokens * dp_size
```

其中 `loss_mask = response_mask * self_distillation_mask`：
- `response_mask`: 屏蔽 padding token
- `self_distillation_mask`: 屏蔽无成功 peer 的样本

---

## 4. CV-SDPO 去污染算法

### 4.1 问题描述

SDPO 的 full teacher context 包含完整的成功 rollout（如 `<answer>C</answer>`）。在 MCQ 任务中，teacher 后验分布混杂了两种信号：

1. **过程信号**（有价值）：知道正确解法后，更倾向于生成正确的推理步骤
2. **答案泄漏**（有害）：仅因知道答案是 C，就提升 shortcut tokens（`therefore`, `thus`, `option C`）并抑制 deliberation tokens（`Wait`, `Maybe`, `Alternatively`）

### 4.2 三种后验分布

所有分布使用 **同一 EMA teacher 权重** $\bar{\theta}$，仅 context 不同：

| 符号 | Context | 描述 |
|------|---------|------|
| $p_t^0(v)$ | 原始题目 | Base 分布，无特权信息 |
| $q_t^{\text{full}}(v)$ | 题目 + 完整成功解答 | Full teacher，包含过程信号 + 答案泄漏 |
| $q_t^{\text{ans}}(v)$ | 题目 + 仅答案字母 | Answer-only teacher，只包含答案泄漏 |

### 4.3 Control Variate 去污染

**核心思想**: 用 answer-only teacher 作为 control variate，估计并减去答案泄漏分量。

**Step 1: 计算 log-prob shift**

$$\Delta_t^{\text{full}}(v) = \log q_t^{\text{full}}(v) - \log p_t^0(v)$$
$$\Delta_t^{\text{ans}}(v) = \log q_t^{\text{ans}}(v) - \log p_t^0(v)$$

**Step 2: 自适应投影系数 $\beta_t$**

$$\beta_t = \text{clip}\left(\frac{\langle \Delta_t^{\text{full}}, \Delta_t^{\text{ans}} \rangle_w}{|\Delta_t^{\text{ans}}|_w^2 + \epsilon},\ 0,\ \beta_{\max}\right)$$

其中 $w_i = p_t^0(i)$ 是 Fisher-style 权重。

- $\beta_t$ 高 → full teacher 的偏移主要是答案泄漏，需要大力修正
- $\beta_t$ 低 → full teacher 包含超越答案知识的过程信号，保留更多

**Step 3: 构建 clean target**

$$\Delta_t^{\text{clean}} = \Delta_t^{\text{full}} - \beta_t \cdot \Delta_t^{\text{ans}}$$
$$\log q_t^{\text{clean}}(v) = \log p_t^0(v) + \gamma \cdot \Delta_t^{\text{clean}}(v)$$

**实现细节** (`core_algos.py:1429-1553`):
- Delta 计算在 **raw log-prob space**（非 renormalized），保留尾部概率质量信息
- $\beta_t$ 计算在 **renormalized space**（条件分布），确保加权内积的数值正确
- 最终 clean target 经过 `add_tail` 或 `renorm` 处理，匹配蒸馏 loss 的输入格式
- Safety clamp: 防止 top-K 概率之和超过 1（`core_algos.py:1544-1545`）

### 4.4 具体示例

**题目**: "Which organelle produces ATP? A) Ribosome B) Nucleus C) Mitochondria D) Golgi"

假设在生成 "therefore" 这个 token 的位置，top-3 token 的 log-prob：

| Token | $p^0$ (base) | $q^{\text{full}}$ | $q^{\text{ans}}$ |
|-------|-------------|-------------------|------------------|
| therefore | -2.0 | -0.5 | -0.3 |
| however | -1.5 | -3.0 | -3.2 |
| maybe | -1.8 | -4.0 | -4.5 |

**Delta**:
- $\Delta^{\text{full}}$("therefore") = -0.5 - (-2.0) = **+1.5** ← full teacher 大幅提升
- $\Delta^{\text{ans}}$("therefore") = -0.3 - (-2.0) = **+1.7** ← 仅靠答案字母也提升了！

这两个 delta 高度相关 → $\beta_t \approx 0.9$

**去污染后**:
- $\Delta^{\text{clean}}$("therefore") = 1.5 - 0.9×1.7 = **-0.03** ← 泄漏被去除

"therefore" 不再被异常提升，deliberation tokens 的抑制也被纠正。

---

## 5. Teacher 模型管理

### 5.1 EMA Teacher

**配置**: `teacher_regularization: "ema"`, `teacher_update_rate: 0.05`

Teacher 是一个独立的模型副本，通过 EMA（Exponential Moving Average）跟踪 student 权重：

```python
# dp_actor.py:132-153
def _update_teacher(self):
    with torch.no_grad():
        for teacher_param, student_param in zip(
            self.teacher_module.parameters(),
            self.actor_module.parameters(),
        ):
            teacher_param.data.mul_(1 - update_rate).add_(
                student_param.data, alpha=update_rate
            )
```

$$\theta_{\text{teacher}} \leftarrow (1 - \tau) \cdot \theta_{\text{teacher}} + \tau \cdot \theta_{\text{student}}$$

**更新时机**: 每个 training step 结束后（`dp_actor.py:1068-1069`），且仅在至少一次成功的梯度更新后才触发。

### 5.2 Trust-Region Teacher（可选）

**配置**: `teacher_regularization: "trust-region"`

将 ref 和 student logits 做线性插值：

```python
# dp_actor.py:51-64
class TrustRegionTeacher(nn.Module):
    def forward(self, *args, **kwargs):
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)
```

### 5.3 Teacher Forward 的执行位置

Teacher forward 在 **FSDP actor worker** 上执行（非 vLLM）：

- **预计算 teacher forward**（CV-SDPO / TASD 共用路径）: `ray_trainer.fit()` 中调用 `actor_rollout_wg.compute_teacher_log_probs()` → `fsdp_workers.py:1084-1108` → `dp_actor.compute_teacher_log_probs()`。在 `update_actor` **之前**执行，结果以 tensor 传入 update_actor。
- **Trainer 内 teacher forward**（fallback，向后兼容）: 当无预计算结果时，`dp_actor.update_policy()` 中仍可 inline 执行 teacher forward。

所有 teacher forward 都在 `torch.no_grad()` 下执行，不参与梯度计算。

### 5.4 Teacher Forward 预计算优化（Plan B）

#### 5.4.1 OOM 问题

在原始实现中，3× teacher FSDP forward（full/base/answer-only）在 `dp_actor.update_policy()` 的 micro-batch 循环**内**执行。此时 GPU 上同时驻留：

| 组件 | 显存估算 |
|------|---------|
| Student forward activation（有梯度） | ~20GB |
| 3× Teacher FSDP forward activation（no_grad） | ~30GB |
| 梯度（FP16） | ~16GB |
| Optimizer states（Adam，FP32） | ~32GB |
| 模型参数（FSDP sharded） | ~14GB |
| **合计** | **~112-142GB** |

在 80GB GPU 上 OOM。

#### 5.4.2 解决方案

将 3× teacher forward 移到 `update_actor` **之前**执行（`ray_trainer.py`），预计算结果以 tensor 写入 batch（`precomputed_*` 前缀），传入 update_actor。

**显存时序分离**：
```
Before:  [teacher activation + gradients + optimizer] → OOM
After:   [teacher activation] → 释放 → [gradients + optimizer] → OK
```

**代码复用**：复用 TASD 已有的 `compute_teacher_log_probs` 入口（`dp_actor.py:678-810`），扩展支持 CV-SDPO 的 base/answer teacher forward。

#### 5.4.3 语义等价论证

Teacher 权重在整个 micro-batch 循环中**不变**：
- EMA 更新 `_update_teacher()` 在 `optimizer.step()` 之后（`dp_actor.py:1068-1069`）
- `optimizer.step()` 在所有 micro-batch 的梯度累积之后
- 因此，提前计算 teacher forward 与循环内计算结果**完全一致**

#### 5.4.4 Top-K 索引近似

SDPO loss 使用 top-K logit 蒸馏，要求 `student_topk_log_probs` 和 `teacher_topk_log_probs` 在相同 vocab 位置上对齐。

预计算流程（两阶段）：
1. **预计算阶段**（update_actor 之前）：student forward (no_grad) → `student_topk_indices`；teacher forward 使用相同 indices gather
2. **训练阶段**（update_actor 内）：student forward (有梯度) 接收 pre-computed `topk_indices` → `torch.gather(logits, dim=-1, index=topk_indices)` 在相同位置 gather，**保留梯度**

**近似**：topk_indices 基于 pre-gradient student 权重，而非 in-loop 更新后的权重。Top-K 集合在小梯度步内变化极小，TASD 已验证此近似的可行性。

#### 5.4.5 预计算 Pipeline

```
ray_trainer.fit()
│
├── _maybe_build_self_distillation_batch()
│     → 构建 teacher/base/answer input_ids
│
├── [Plan B] compute_teacher_log_probs()    ← 新增
│     → student forward (no_grad) → topk_indices
│     → teacher forward × 3 (full/base/answer)
│     → 结果写入 batch.batch["precomputed_*"]
│     → teacher activation 释放
│
└── update_actor()
      → dp_actor.update_policy()
        ├── 检测 "precomputed_teacher_log_probs" in model_inputs
        ├── student forward (有梯度, topk_indices=precomputed)
        ├── 直接读取 precomputed teacher tensors（跳过 teacher forward）
        └── compute_self_distillation_loss()
```

---

## 6. 梯度更新流程

### 6.1 Student 梯度更新

```python
# dp_actor.py:841-1069 (update_policy)
for epoch in range(ppo_epochs):       # 通常 1
    for mini_batch in mini_batches:     # 通常 1（on-policy）
        optimizer.zero_grad()
        for micro_batch in micro_batches:
            # Student forward（有梯度）
            outputs = _forward_micro_batch(model_inputs, ...)

            # Teacher forward（无梯度）
            with torch.no_grad():
                teacher_outputs = _forward_micro_batch(teacher_inputs, ..., module=teacher_model)

            # CV 去污染 + Loss 计算
            pg_loss, pg_metrics = compute_self_distillation_loss(...)

            # Gradient accumulation
            loss = pg_loss * loss_scale_factor
            loss.backward()

        # Gradient clipping + optimizer step
        grad_norm = actor_module.clip_grad_norm_(max_norm=grad_clip)
        optimizer.step()

    # EMA teacher update（每个 training step 结束）
    _update_teacher()
```

### 6.2 关键设计

- **On-policy**: `ppo_mini_batch_size=32` = `train_batch_size=32`，且 `ppo_epochs=1` → 每个 batch 只做一次更新
- **Gradient accumulation**: 通过 micro-batch 实现，`loss_scale_factor = 1 / gradient_accumulation`
- **梯度只流过 student**: teacher forward 在 `no_grad()` 下，CV 去污染的 clean target 也是 detached
- **LR scheduler**: 在每个 training step 后 step（`fsdp_workers.py:965`）

---

## 7. 配置详解

### 7.1 CV-SDPO 完整配置 (`cv_sdpo.yaml`)

```yaml
defaults:
  - ppo_trainer
  - user
  - _self_

max_model_len: 18944  # 3x teacher context 需要足够空间

actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 32
    policy_loss:
      loss_mode: sdpo
    self_distillation:
      alpha: 1.0                # Reverse KL: KL(student || teacher)
      max_reprompt_len: 10240
      is_clip: 2.0              # IS ratio clamp 上限
      entropy_weighting: false
      entropy_temperature: 1.0
      entropy_weighting_version: "v4"
      # CV-SDPO leakage decontamination
      leakage_decontamination_enabled: true
      leakage_decontamination_mode: "control_variate"
      answer_context_source: "peer_success"
      cv_gamma: 0.5             # clean shift 强度
      beta_mode: "adaptive"     # 自适应 vs 固定 beta
      beta_fixed: 0.5           # beta_mode="fixed" 时使用
      beta_max: 1.0             # beta 上限
      stop_gradient_clean_target: true
      fallback_when_no_success: "no_sdpo"
    optim:
      lr: 1e-5
  rollout:
    n: 8                        # 每题采样数
    calculate_log_probs: True

algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: False
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: 2.0

data:
  train_batch_size: 32

trainer:
  val_before_train: False
```

### 7.2 Self-Distillation 默认参数 (`actor.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `full_logit_distillation` | `True` | 使用 top-K logit 蒸馏（而非仅 token log-ratio） |
| `alpha` | `0.0` | KL 方向（CV-SDPO 覆盖为 1.0） |
| `success_reward_threshold` | `1.0` | 判定成功 rollout 的最低分数 |
| `teacher_regularization` | `"ema"` | Teacher 更新方式 |
| `teacher_update_rate` | `0.05` | EMA 更新率 τ |
| `distillation_topk` | `None` | Top-K 数量（通常设为 100） |
| `distillation_add_tail` | `True` | 是否添加尾部概率桶 |
| `dont_reprompt_on_self_success` | `False` | 成功 rollout 是否排除自身作为 peer |

### 7.3 Pipeline Change (CV-SDPO vs SDPO)

```
Vanilla SDPO:
  rollout → reward → full reprompt → q_full → loss

CV-SDPO:
  rollout → reward
          → full reprompt        → q_full   (teacher_topk_logps)
          → answer-only reprompt → q_ans    (answer_teacher_topk_logps)
          → base prompt          → p0       (base_teacher_topk_logps)
          → compute_cv_sdpo_clean_target()
          → q_clean replaces q_full → loss
```

额外 batch tensors（`ray_trainer.py` 添加）:
- `base_teacher_input_ids`, `base_teacher_attention_mask`, `base_teacher_position_ids`
- `answer_teacher_input_ids`, `answer_teacher_attention_mask`, `answer_teacher_position_ids`

---

## 8. Context 构建模板

### 8.1 Full Teacher（现有 SDPO）

```
{prompt}
Correct solution:
{successful_previous_rollout}

Correctly solve the original question.
```

### 8.2 Answer-Only Teacher（CV-SDPO）

```
{prompt}
The answer is {answer_letter}.
Correctly solve the original question.
```

### 8.3 Base Teacher

原始题目，无任何附加信息。

### 8.4 Fallback 规则

- 无成功 rollout → `self_distillation_mask=0`，该样本跳过 SDPO loss，仅走 GRPO
- 答案提取失败 → answer-only context 回退为原始 prompt（delta_ans ≈ 0，CV 修正变为 no-op）

### 8.5 数据集兼容性

| 数据集 | MCQ 格式 | 答案提取 | CV-SDPO 有效 |
|--------|----------|----------|-------------|
| sciknoweval/biology | Yes | Works | Yes |
| sciknoweval/chemistry | Yes | Works | Yes |
| sciknoweval/material | Yes | Works | Yes |
| sciknoweval/physics | Yes | Works | Yes |
| tooluse | No | Fails（无 A/B/C/D） | No-op（退化为 vanilla SDPO） |

---

## 9. 诊断 Metrics

所有 CV-SDPO metrics 在 `compute_cv_sdpo_clean_target()` 中内联计算，输出到 SwanLab。

### 9.1 后验相似度

$$\text{cos\_full\_ans} = \frac{\langle \Delta^{\text{full}}, \Delta^{\text{ans}} \rangle_{p^0}}{|\Delta^{\text{full}}|_{p^0} \cdot |\Delta^{\text{ans}}|_{p^0}}$$

高值 → full teacher 偏移主要来自答案泄漏。

### 9.2 残差正交性

$$\text{cos\_clean\_ans} = \frac{\langle \Delta^{\text{clean}}, \Delta^{\text{ans}} \rangle_{p^0}}{|\Delta^{\text{clean}}|_{p^0} \cdot |\Delta^{\text{ans}}|_{p^0}}$$

CV 修正后应接近 0。

### 9.3 熵跟踪

| Metric | 含义 |
|--------|------|
| `cv_sdpo/H_base` | Base teacher 熵 |
| `cv_sdpo/H_full` | Full teacher 熵 |
| `cv_sdpo/H_ans` | Answer-only teacher 熵 |
| `cv_sdpo/H_clean` | Clean target 熵 |

泄漏热点：$H(q^{\text{full}}) \ll H(p^0)$ 且 $q^{\text{full}} \approx q^{\text{ans}}$ 的位置。

### 9.4 自适应 Beta

- `cv_sdpo/beta_mean` — 有效位置上 $\beta_t$ 的均值

### 9.5 Self-Distillation 覆盖率

| Metric | 含义 |
|--------|------|
| `self_distillation/success_group_fraction` | 有至少一条成功 rollout 的题目比例 |
| `self_distillation/success_sample_fraction` | 有成功 peer solution 的样本比例 |
| `self_distillation/answer_extracted_fraction` | 成功提取答案字母的样本比例 |
| `self_distillation/reprompt_sample_fraction` | 实际参与 SDPO loss 的样本比例 |

### 9.6 Token Mass（可选，`leakage_diagnostics.py`）

- Search tokens: `Wait, Maybe, Alternatively, Consider, Check, Suppose, Let`
- Shortcut tokens: `Therefore, Thus, Hence, answer, option, boxed, final`

---

## 10. 训练脚本

### 10.1 Nebula 提交

```bash
bash nebula_scripts/submit_cv_sdpo_sweep.sh [--dry-run]
```

关键环境变量:
- `CONFIG_NAME=cv_sdpo`
- `CV_GAMMA=0.5`
- `BETA_MODE=adaptive`

训练脚本: `nebula_scripts/sdpo/sdpo_sciknoweval_parametric.sh`

### 10.2 本地调试

```bash
bash run_local_cv_sdpo.sh [suffix]
```

使用 `datasets/sciknoweval/biology`（本地），设置绝对路径。

---

## 11. 实验设计

### 11.1 配置

| 参数 | 值 |
|------|-----|
| Model | Qwen3-8B |
| Datasets | sciknoweval/{biology,chemistry,material,physics}, tooluse |
| n_rollouts | 8 |
| val_n | 16 |
| train_batch_size | 32 |
| distillation_topk | 100 |
| alpha (RKL) | 1.0 |
| cv_gamma | 0.5 |
| beta_mode | adaptive |
| beta_max | 1.0 |
| teacher_update_rate (EMA τ) | 0.05 |
| total_steps | 250 |
| save_best_metric | val-core/sciknoweval/acc/mean@16 |

### 11.2 Baselines

| # | Method | Description |
|---|--------|-------------|
| 1 | GRPO | On-policy GRPO（无蒸馏） |
| 2 | SDPO | Vanilla SDPO，用 $q^{\text{full}}$ |
| 3 | CV-SDPO (fixed β) | $\beta = 0.5$，固定投影系数 |
| 4 | CV-SDPO (adaptive β) | 加权投影，自适应 β |

### 11.3 Ablation Plan

1. **Random control**: 用随机答案的 $q^{\text{ans}}$ 替代 → 验证 answer-specific 后验的必要性
2. **Fixed vs adaptive β**: $\beta \in \{0.25, 0.5, 0.75, 1.0\}$ vs adaptive
3. **γ sweep**: $\gamma \in \{0.25, 0.5, 1.0\}$（clean shift 强度）
4. **KL 方向**: Forward KL vs Reverse KL vs JSD
5. **Oracle diagnostic**: 用 gold answer 替代 peer-extracted answer

---

## 12. Known Issues & TODOs

### 已实现但不完整:
- [ ] `fallback_when_no_success` 配置已定义但未在代码中使用（总是回退到原始 prompt）
- [ ] `answer_context_source` 配置已定义但未使用（总是用 peer_success）
- [ ] `stop_gradient_clean_target` 配置已定义但未使用（行为已由 `no_grad` 保证正确）
- [ ] `tooluse` 数据集：CV-SDPO 为 no-op（答案提取总是失败）

### 已完成:
- [x] Teacher forward 预计算优化（Plan B）：解决 update_actor 阶段 OOM

### 未来优化:
- [ ] 将 3 种 teacher context 合并为单次 forward（减少 3x 顺序开销）
- [ ] 当 base context 在同一 step 内相同时缓存 base teacher logits
- [ ] 实现 ALOP-SDPO（`leakage_decontamination_mode: "logit_projection"`）

---

## 13. 代码复用与命名说明

> **阅读指南**：CV-SDPO 代码中存在多个名称相似但用途不同的函数和概念。本节旨在澄清它们的关系，避免在梳理实现时被误导。

### 13.1 ref_module vs teacher_module

| 概念 | 定义位置 | 实际指向 | 说明 |
|------|---------|---------|------|
| `ref_module_fsdp` | `fsdp_workers.py:857-870` | 独立的 FSDP 模型副本 | 初始化时从同一 checkpoint 加载 |
| `teacher_module` | `fsdp_workers.py:910` | **→ ref_module_fsdp** | 仅一个引用，不是独立模型 |
| `actor_module_fsdp` | `fsdp_workers.py:830-855` | Student 模型 | 接受梯度更新 |

**关键**：CV-SDPO 中 `teacher_module` 和 `ref_module` 是**同一个对象**。EMA 更新作用于 `ref_module`（通过 `teacher_module` 别名），使其平滑跟踪 student 权重。

### 13.2 三个 "compute log_prob" 函数

| 函数 | 文件 | 使用模型 | 用途 | CV-SDPO 是否调用 |
|------|------|---------|------|----------------|
| `compute_log_prob()` | `dp_actor.py:592` | **actor_module** (student) | 计算 `old_log_probs` | **是**（重算 student log_prob） |
| `_compute_ref_log_prob()` | `ray_trainer.py:1709` | **ref_module** | KL penalty 中的 ref 基准 | **否**（`use_kl_loss=False`） |
| `compute_teacher_log_probs()` | `dp_actor.py:678` | **teacher_module** (= ref_module) | Teacher 蒸馏 + CV-SDPO 去污染 | **是**（Plan B 预计算） |

**容易混淆的点**：
- `_compute_ref_log_prob` 和 `compute_teacher_log_probs` 用的是**同一个模型**（ref_module），但用途完全不同
- CV-SDPO 配置 `use_kl_loss=False` + `use_kl_in_reward=False`，所以 `_compute_ref_log_prob` **从不被调用**
- `compute_teacher_log_probs` 最初为 TASD 实现（用于 reward 计算），CV-SDPO Plan B 复用同一入口（用于 loss 计算中的 teacher 分布）

### 13.3 Teacher forward 的两条路径

```
路径 A (TASD):
  ray_trainer.fit()
    → compute_teacher_log_probs()  → teacher_result
    → compute_tasd_token_rewards(teacher_result)  → token_level_rewards
    → compute_advantage()
    → update_actor()  ← teacher 结果已折叠为 reward，不传入

路径 B (CV-SDPO Plan B):
  ray_trainer.fit()
    → compute_teacher_log_probs()  → teacher_result
    → batch["precomputed_*"] = teacher_result  ← 以 tensor 传入
    → update_actor()
      → dp_actor.update_policy()
        → 检测 precomputed_* → 跳过 inline teacher forward
        → compute_self_distillation_loss(teacher_topk=precomputed)
```

**相同入口，不同消费方式**：TASD 在 update_actor 之前消费 teacher 结果（变成 reward），CV-SDPO 在 update_actor 内部消费（作为 SDPO loss 的 teacher 分布）。

### 13.4 预计算 tensor 命名规范

所有预计算结果使用 `precomputed_` 前缀，与 inline 计算的变量区分：

| Tensor key | 来源 | 用途 |
|-----------|------|------|
| `precomputed_teacher_log_probs` | teacher forward (per-token) | SDPO loss 的 teacher 分布 |
| `precomputed_teacher_topk_log_probs` | teacher forward (top-K gather) | SDPO top-K 蒸馏 |
| `precomputed_student_topk_log_probs` | student forward (no_grad, top-K) | 仅用于校验 |
| `precomputed_student_topk_indices` | student forward (no_grad, top-K) | 训练时 student gather 使用相同 indices |
| `precomputed_base_teacher_topk_log_probs` | base teacher forward | CV 去污染 |
| `precomputed_answer_teacher_topk_log_probs` | answer teacher forward | CV 去污染 |

---

## References

1. [Reinforcement Learning via Self-Distillation (SDPO)](https://arxiv.org/abs/2601.20802)
2. [SDPO GitHub](https://github.com/lasgroup/SDPO)
3. [Anti-Self-Distillation for Reasoning RL via PMI](https://arxiv.org/abs/2605.11609)
4. [From Generic Correlation to Input-Specific Credit (CREDIT)](https://arxiv.org/abs/2605.11613)
5. [SciKnowEval](https://arxiv.org/abs/2406.09098)
