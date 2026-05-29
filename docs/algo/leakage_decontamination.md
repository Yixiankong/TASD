# Answer-Leakage Decontaminated SDPO

> Design document for CV-SDPO and ALOP-SDPO. Based on arXiv:2601.20802 (SDPO) framework,
> targeting SciKnowEval L3 reasoning subsets and ToolUse.

---

## 1. Background: SDPO Framework

### 1.1 Core Mechanism

SDPO flow:

1. Sample on-policy rollouts from current policy
2. Score rollouts (scalar reward or rich feedback)
3. Build self-teacher prompt with feedback/successful rollout in context
4. Teacher re-evaluates original rollout's token log-probs under feedback-conditioned context
5. Teacher-student distribution gap forms dense token-level supervision

In "without rich feedback" mode (SciKnowEval), SDPO uses successful peer rollouts as implicit feedback.

### 1.2 Code Implementation (Current State)

Key config switches (`verl/trainer/config/sdpo.yaml`):

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: sdpo
    self_distillation:
      full_logit_distillation: true
      alpha: 0.5              # 0.0=FKL, 0.5=JSD, 1.0=RKL
      teacher_regularization: ema
      teacher_update_rate: 0.05
      distillation_topk: 100
      distillation_add_tail: true
      is_clip: 2.0
```

Implementation files:
- `verl/trainer/ppo/ray_trainer.py` — teacher context construction, answer extraction
- `verl/workers/actor/dp_actor.py` — teacher forward passes (base/full/answer)
- `verl/trainer/ppo/core_algos.py` — `compute_cv_sdpo_clean_target()`, `compute_self_distillation_loss()`
- `verl/workers/config/actor.py` — config dataclass with CV-SDPO fields
- `verl/trainer/config/cv_sdpo.yaml` — Hydra config for CV-SDPO

---

## 2. Problem Statement

SDPO's full privileged context typically contains:

```
Correct solution: {successful_previous_rollout}
Correctly solve the original question.
```

In SciKnowEval MCQ tasks, the successful rollout contains the final answer (e.g., `<answer>C</answer>`). This causes the self-teacher's posterior to be a **hindsight posterior** that:

- Boosts shortcut tokens: `therefore`, `thus`, `answer is`, `option C`, `final`
- Suppresses deliberation tokens: `Wait`, `Maybe`, `Alternatively`, `Consider`

**Goal**: Decontaminate the teacher logits before they enter the loss, without changing SDPO's rollout/reprompt/top-k distillation pipeline.

---

## 3. Three Posteriors

All methods build on three distributions at each token position $t$:

| Symbol | Context | Description |
|--------|---------|-------------|
| $p_t^0(v)$ | Original prompt only | Base distribution (no privileged info) |
| $q_t^{\text{full}}(v)$ | Full SDPO reprompt | Teacher with complete solution context |
| $q_t^{\text{ans}}(v)$ | Answer-only reprompt | Teacher with only the final answer letter |

All three use the **same EMA teacher weights** $\bar{\theta}$, differing only in context.

The decontaminated target:

$$q_t^{\text{clean}} = \text{DeLeak}(p_t^0, q_t^{\text{full}}, q_t^{\text{ans}})$$

replaces $q_t^{\text{full}}$ in the SDPO loss.

---

## 4. Method 1: CV-SDPO (Control Variate)

### 4.1 Motivation

The answer-only teacher serves as a control variate estimating answer leakage:

$$\Delta_t^{\text{full}}(v) = \log q_t^{\text{full}}(v) - \log p_t^0(v)$$

$$\Delta_t^{\text{ans}}(v) = \log q_t^{\text{ans}}(v) - \log p_t^0(v)$$

### 4.2 Adaptive $\beta_t$

Per-position weighted projection coefficient:

$$\beta_t = \text{clip}\left(\frac{\langle \Delta_t^{\text{full}}, \Delta_t^{\text{ans}} \rangle_w}{|\Delta_t^{\text{ans}}|_w^2 + \epsilon},\ 0,\ \beta_{\max}\right)$$

where $w_i = p_t^0(i)$ (Fisher-style weights on conditional top-K support).

- High $\beta_t$ → full teacher is mostly answer leakage
- Low $\beta_t$ → full teacher contains process signal beyond answer knowledge

### 4.3 Clean Target Construction

$$\Delta_t^{\text{clean}} = \Delta_t^{\text{full}} - \beta_t \cdot \Delta_t^{\text{ans}}$$

$$\log q_t^{\text{CV}}(v) = \log p_t^0(v) + \gamma \cdot \Delta_t^{\text{clean}}(v)$$

**Implementation detail**: The computation operates in **raw log-prob space** (not renormalized), so that `add_tail` correctly captures the remaining probability mass outside top-K. A safety clamp prevents the top-K sum from exceeding 1.

### 4.4 Config (`cv_sdpo.yaml`)

```yaml
actor_rollout_ref:
  actor:
    self_distillation:
      leakage_decontamination_enabled: true
      leakage_decontamination_mode: "control_variate"
      answer_context_source: "peer_success"
      cv_gamma: 0.5
      beta_mode: "adaptive"    # or "fixed"
      beta_fixed: 0.5
      beta_max: 1.0
      fallback_when_no_success: "no_sdpo"
```

### 4.5 Pipeline Change

```
Original SDPO:
  rollout → reward → full reprompt → q_full → loss

CV-SDPO:
  rollout → reward
          → full reprompt        → q_full   (teacher_topk_logps)
          → answer-only reprompt → q_ans    (answer_teacher_topk_logps)
          → base prompt          → p0       (base_teacher_topk_logps)
          → compute_cv_sdpo_clean_target()
          → loss
```

Extra batch tensors added by `ray_trainer.py`:
- `base_teacher_input_ids`, `base_teacher_attention_mask`, `base_teacher_position_ids`
- `answer_teacher_input_ids`, `answer_teacher_attention_mask`, `answer_teacher_position_ids`

---

## 5. Method 2: ALOP-SDPO (Logit-Space Orthogonal Projection)

> **Status**: Not yet implemented. Design only.

### 5.1 Motivation

CV-SDPO subtracts in log-prob PMI space (per-token scalar). ALOP treats answer leakage as a **direction** in logit space, more suitable for capturing correlated shortcut clusters.

### 5.2 Fisher-Weighted Projection

On support $S_t$:

$$\Delta z_t^{\text{full}} = z_t^{\text{full}} - z_t^{\text{base}}$$

$$\Delta z_t^{\text{ans}} = z_t^{\text{ans}} - z_t^{\text{base}}$$

$$\alpha_t = \text{clip}\left(\frac{\sum_{i \in S_t} p_t^0(i) \cdot \Delta z_{t,i}^{\text{full}} \cdot \Delta z_{t,i}^{\text{ans}}}{\sum_{i \in S_t} p_t^0(i) \cdot (\Delta z_{t,i}^{\text{ans}})^2 + \epsilon},\ 0,\ \alpha_{\max}\right)$$

$$\Delta z_t^{\text{clean}} = \Delta z_t^{\text{full}} - \alpha_t \cdot \Delta z_t^{\text{ans}}$$

### 5.3 Clean Teacher

$$z_t^{\text{clean}} = z_t^{\text{base}} + \gamma \cdot \Delta z_t^{\text{clean}}$$

$$q_t^{\text{ALOP}} = \text{softmax}_{S_t}(z_t^{\text{clean}})$$

### 5.4 Key Difference from CV-SDPO

| Aspect | CV-SDPO | ALOP-SDPO |
|--------|---------|-----------|
| Space | Log-prob (PMI) | Raw logit |
| Normalization | Renormalized on support | Softmax on support |
| Captures | Token-level leakage | Cluster-level directional leakage |
| Implementation | Done | Planned |

---

## 6. Data & Context Construction

### 6.1 Answer Extraction

From peer successful rollouts, extract the answer letter via:
1. XML match: `<answer>\s*([A-D])\s*</answer>`
2. Fallback: last standalone `A/B/C/D` in text

**Implementation**: `ray_trainer.py:_extract_answer_letter_from_response()`

### 6.2 Context Templates

**Full teacher** (existing SDPO):
```
{prompt}
Correct solution:
{successful_previous_rollout}

Correctly solve the original question.
```

**Answer-only teacher** (CV-SDPO):
```
{prompt}
The answer is {answer_letter}.
Correctly solve the original question.
```

**Base teacher**: original prompt only (no privileged context).

### 6.3 Fallback Rules

When no successful rollout exists in the group:
- `"no_sdpo"` (default): skip SDPO loss for that sample, use GRPO only
- Answer-only prompt falls back to original prompt (delta_ans ≈ 0, CV correction becomes no-op)

### 6.4 Dataset Compatibility

| Dataset | MCQ format | Answer extraction | CV-SDPO effective |
|---------|-----------|-------------------|-------------------|
| sciknoweval/biology | Yes | Works | Yes |
| sciknoweval/chemistry | Yes | Works | Yes |
| sciknoweval/material | Yes | Works | Yes |
| sciknoweval/physics | Yes | Works | Yes |
| tooluse | No | Fails (no A/B/C/D) | No-op (degrades to vanilla SDPO) |

---

## 7. Training Scripts

### 7.1 Nebula Submission

```bash
bash nebula_scripts/submit_cv_sdpo_sweep.sh [--dry-run]
```

Key env vars passed to training script:
- `CONFIG_NAME=cv_sdpo`
- `CV_GAMMA=0.5`
- `BETA_MODE=adaptive`

Training script: `nebula_scripts/sdpo/sdpo_sciknoweval_parametric.sh`
- Uses `--config-name ${CONFIG_NAME:-sdpo}` (parametric)
- Conditionally passes `CV_GAMMA` and `BETA_MODE` as Hydra overrides

### 7.2 Local Debug

```bash
bash run_local_cv_sdpo.sh [suffix]
```

Uses `datasets/sciknoweval/biology` (local), sets absolute paths for `data.train_files`/`data.val_files`.

---

## 8. Diagnostic Metrics

All metrics are computed inline in `compute_cv_sdpo_clean_target()` and logged to SwanLab.

### 8.1 Posterior Similarity

$$\text{cos\_full\_ans} = \frac{\langle \Delta^{\text{full}}, \Delta^{\text{ans}} \rangle_{p^0}}{|\Delta^{\text{full}}|_{p^0} \cdot |\Delta^{\text{ans}}|_{p^0}}$$

High value → full teacher shift is mostly answer leakage.

### 8.2 Residual Orthogonality

$$\text{cos\_clean\_ans} = \frac{\langle \Delta^{\text{clean}}, \Delta^{\text{ans}} \rangle_{p^0}}{|\Delta^{\text{clean}}|_{p^0} \cdot |\Delta^{\text{ans}}|_{p^0}}$$

Should be near 0 after CV correction.

### 8.3 Entropy Tracking

- `cv_sdpo/H_base` — base teacher entropy
- `cv_sdpo/H_full` — full teacher entropy
- `cv_sdpo/H_ans` — answer-only teacher entropy
- `cv_sdpo/H_clean` — clean target entropy

Leakage hotspots: positions where $H(q^{\text{full}}) \ll H(p^0)$ and $q^{\text{full}} \approx q^{\text{ans}}$.

### 8.4 Adaptive Beta

- `cv_sdpo/beta_mean` — average $\beta_t$ across valid positions

### 8.5 Token Mass (Optional, via `leakage_diagnostics.py`)

Search tokens: `Wait, Maybe, Alternatively, Consider, Check, Suppose, Let`

Shortcut tokens: `Therefore, Thus, Hence, answer, option, boxed, final`

---

## 9. Experiment Design

### 9.1 Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-8B |
| Datasets | sciknoweval/{biology,chemistry,material,physics}, tooluse |
| n_rollouts | 8 |
| val_n | 16 |
| train_batch_size | 32 |
| distillation_topk | 100 |
| alpha (JSD) | 0.5 |
| cv_gamma | 0.5 |
| beta_mode | adaptive |
| beta_max | 1.0 |
| total_steps | 250 |
| save_best_metric | val-core/sciknoweval/acc/mean@16 |

### 9.2 Baselines

| # | Method | Description |
|---|--------|-------------|
| 1 | GRPO | On-policy GRPO (no distillation) |
| 2 | SDPO | Vanilla SDPO with $q^{\text{full}}$ |
| 3 | CV-SDPO (fixed $\beta$) | $\beta = 0.5$ |
| 4 | CV-SDPO (adaptive $\beta$) | Weighted projection |
| 5 | ALOP-SDPO | Logit-space projection (planned) |

### 9.3 Main Results Table

| Method | Bio | Chem | Mat | Phys | ToolUse | Avg | Len |
|--------|-----|------|-----|------|---------|-----|-----|
| GRPO | | | | | | | |
| SDPO | | | | | | | |
| CV-SDPO | | | | | | | |

Primary metric: avg@16

---

## 10. Ablation Plan

1. **Random control**: replace $q^{\text{ans}}$ with random-answer posterior → verify answer-specific posterior matters
2. **Fixed vs adaptive $\beta$**: $\beta \in \{0.25, 0.5, 0.75, 1.0\}$ vs adaptive
3. **$\gamma$ sweep**: $\gamma \in \{0.25, 0.5, 1.0\}$ (clean shift strength)
4. **CV-SDPO vs ALOP-SDPO**: log-prob vs logit-space correction
5. **Oracle diagnostic**: use gold answer instead of peer-extracted answer

---

## 11. Known Issues & TODOs

### Implemented but incomplete:
- [ ] `fallback_when_no_success` config defined but not used in code (always falls back to original prompt)
- [ ] `answer_context_source` config defined but not used (always uses peer_success)
- [ ] `stop_gradient_clean_target` config defined but not used (behavior is already correct due to `no_grad`)
- [ ] `tooluse` dataset: CV-SDPO is a no-op since answer extraction always fails

### Future optimizations:
- [ ] Batch 3 teacher contexts into single forward (reduce 3x sequential overhead)
- [ ] Cache base teacher logits when base context is identical within a step
- [ ] Implement ALOP-SDPO (`leakage_decontamination_mode: "logit_projection"`)

---

## References

1. [Reinforcement Learning via Self-Distillation (SDPO)](https://arxiv.org/abs/2601.20802)
2. [SDPO GitHub](https://github.com/lasgroup/SDPO)
3. [Anti-Self-Distillation for Reasoning RL via PMI](https://arxiv.org/abs/2605.11609)
4. [From Generic Correlation to Input-Specific Credit (CREDIT)](https://arxiv.org/abs/2605.11613)
5. [SciKnowEval](https://arxiv.org/abs/2406.09098)
