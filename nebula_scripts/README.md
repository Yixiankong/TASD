# nebula_scripts 使用说明

本目录包含在 **Nebula 平台**上提交和运行 SDPO 系列训练实验的所有脚本。

---

## 目录结构

```
nebula_scripts/
├── submit_tasd_ema_sweep.sh        # TASD EMA Teacher 超参扫描提交脚本
├── submit_grpo_baseline_sweep.sh   # GRPO Baseline 超参扫描提交脚本
├── submit_sdpo_baseline_sweep.sh   # SDPO Baseline 超参扫描提交脚本
├── submit_sdpo_ew_sweep.sh         # SDPO + Entropy Weighting 超参扫描提交脚本
├── submit_job.sh                   # 通用单 job 提交脚本
├── launch_ray_cluster.sh           # Nebula 容器内启动 Ray 集群并执行训练
├── entry.py                        # Nebula 任务入口（调用 launch_ray_cluster.sh）
├── cluster.json                    # 8 GPU 集群配置
├── cluster_gpu_4.json              # 4 GPU 集群配置
├── prepare_data.sh                 # 数据集上传到 OSS 的辅助脚本
├── tasd/
│   ├── tasd_sciknoweval_parametric.sh    # TASD 参数化训练脚本（sweep 调用）
│   └── tasd_sciknoweval_qwen3_8B.sh      # TASD 单次固定超参训练脚本
├── grpo/
│   └── grpo_sciknoweval_parametric.sh    # GRPO 参数化训练脚本（sweep 调用）
└── sdpo/
    ├── sdpo_sciknoweval_parametric.sh    # SDPO Baseline 参数化训练脚本
    └── sdpo_entropy_weighting_parametric.sh  # SDPO+EW 参数化训练脚本
```

---

## 前置条件：环境变量

提交任务前，需在本地 shell 中 export 以下变量：

```bash
export OPENLM_TOKEN="..."         # Nebula 平台认证 token
export OSS_ACCESS_ID="..."        # 阿里云 OSS Access Key ID
export OSS_ACCESS_KEY="..."       # 阿里云 OSS Access Key Secret
export SWANLAB_API_KEY="..."      # SwanLab 实验跟踪 API Key（可选，有内置 fallback）
```

---

## 实验说明

### 1. TASD EMA Teacher 超参扫描

**提交脚本**：`submit_tasd_ema_sweep.sh`  
**训练脚本**：`tasd/tasd_sciknoweval_parametric.sh`  
**SwanLab 项目**：`TASD_param_search`

TASD（Teacher-Aware Self-Distillation）是本项目的核心算法。模型在每步 rollout 后，以当前或 EMA 维护的历史模型作为 teacher，对失败（或全部）rollout 进行蒸馏。

#### 可扫描超参

| 变量 | 含义 | 典型取值 |
|------|------|---------|
| `REWARD_TYPE` | Teacher 信号类型 | `teacher_prob`（直接用 teacher 概率）、`log_teacher_prob` |
| `LR` | 学习率 | `1e-5`、`5e-6` |
| `ENTROPY_COEFF` | 熵正则系数，防止策略过早坍缩 | `0.0`、`0.05`、`0.1`、`1.0` |
| `TEACHER_REG` | Teacher 更新方式 | `none`（固定初始模型）、`ema`（指数移动平均） |
| `TEACHER_UPDATE_RATE` | EMA 更新速率（`TEACHER_REG=ema` 时有效） | `0.0`（极慢）、`0.05`、`0.1` |
| `INCLUDE_SUCCESSFUL_ROLLOUTS` | 成功 rollout 是否也参与 TASD reward | `True`（全量）、`False`（仅失败） |

#### 固定参数

| 参数 | 值 | 含义 |
|------|----|------|
| `NORM_ADV_BY_STD` | `False` | advantage 不除以标准差（teacher_prob ∈ [0,1] 天然有界） |
| `CLIP_ADV` | `True` | 开启 advantage clipping |
| `CLIP_ADV_VALUE` | `5.0` | advantage clip 阈值 |
| `ROLLOUT_IS` | `token` | token-level importance sampling |
| `TRAIN_BATCH_SIZE` | `32` | 训练 batch size |
| `MINI_BATCH_SIZE` | `32` | PPO mini-batch size |
| `ROLLOUT_N` | `8` | 每个 prompt 采样的 rollout 数量 |

#### 实验名格式

```
TASD-bio-lr{LR}-rt{REWARD_TYPE}-nostd-clip5.0-ent{ENTROPY_COEFF}-rctoken-isr{0/1}-ema{TEACHER_UPDATE_RATE}-Qwen3-8B-{时间戳}
```

---

### 2. GRPO Baseline

**提交脚本**：`submit_grpo_baseline_sweep.sh`  
**训练脚本**：`grpo/grpo_sciknoweval_parametric.sh`  
**SwanLab 项目**：`TASD_para_search`

GRPO（Group Relative Policy Optimization）是强化学习基线方法，不使用 teacher 信号，仅依赖组内 reward 的相对优劣计算 advantage。

#### 可扫描超参

| 变量 | 含义 | 典型取值 |
|------|------|---------|
| `DATASET` | 数据集（OSS `datasets/` 下相对路径） | `sciknoweval/biology`、`sciknoweval/chemistry` 等 |
| `MODEL_NAME` | 基底模型（OSS `base_models/` 下目录名） | `Qwen3-8B`、`Olmo-3-7B-Instruct` |
| `LR` | 学习率 | `1e-5`、`1e-6` |
| `MINI_BATCH_SIZE` | PPO mini-batch size | `8`、`32` |

#### 实验名格式

```
GRPO-{数据集短名}-mbs{MINI_BATCH_SIZE}-train{TRAIN_BATCH_SIZE}-lr{LR}-{MODEL_NAME}-{时间戳}
```

---

### 3. SDPO Baseline

**提交脚本**：`submit_sdpo_baseline_sweep.sh`  
**训练脚本**：`sdpo/sdpo_sciknoweval_parametric.sh`  
**SwanLab 项目**：`TASD_para_search`

SDPO（Self-Distillation Policy Optimization）基线，使用 JS 散度（alpha=0.5）在成功/失败 rollout 之间做自蒸馏。

#### 可扫描超参

| 变量 | 含义 | 典型取值 |
|------|------|---------|
| `DATASET` | 数据集 | `sciknoweval/biology` 等 |
| `MODEL_NAME` | 基底模型 | `Qwen3-8B` |
| `LR` | 学习率 | `1e-5` |
| `ALPHA` | 蒸馏散度权重（0=forward KL, 0.5=JS, 1=reverse KL） | `0.5` |
| `DONT_REPROMPT_ON_SELF_SUCCESS` | 成功 rollout 不重新提示 teacher | `True`、`False` |

#### 实验名格式

```
SDPO-{数据集短名}-train{TRAIN_BATCH_SIZE}-alpha{ALPHA}-lr{LR}-dross{DONT_REPROMPT}-{MODEL_NAME}-{时间戳}
```

---

### 4. SDPO + Entropy Weighting

**提交脚本**：`submit_sdpo_ew_sweep.sh`  
**训练脚本**：`sdpo/sdpo_entropy_weighting_parametric.sh`  
**SwanLab 项目**：`TASD_para_search`

在 SDPO 基础上，根据 token 级别的熵对蒸馏 loss 加权，鼓励在高不确定性位置学习更多。

#### 可扫描超参（在 SDPO 基础上新增）

| 变量 | 含义 | 典型取值 |
|------|------|---------|
| `ENTROPY_WEIGHTING` | 是否开启熵加权 | `True`、`False` |
| `ENTROPY_TEMPERATURE` | 熵加权 softmax 温度（EW=True 时有效） | `0.5`、`1.0`、`2.0` |
| `ENTROPY_WEIGHTING_VERSION` | 加权算法版本 | `v1`（基础版）、`v4`（改进版） |

#### 实验名格式

```
SDPO-{数据集短名}-alpha{ALPHA}-lr{LR}-dross{DONT_REPROMPT}-{EW_VER}-ew{True/False}-et{TEMPERATURE}-{MODEL_NAME}-{时间戳}
```

---

## 使用方式

### 提交超参扫描

```bash
# 先 export 必要环境变量
export OPENLM_TOKEN="..."
export OSS_ACCESS_ID="..."
export OSS_ACCESS_KEY="..."

# dry-run：只打印 job 列表，不实际提交
bash nebula_scripts/submit_tasd_ema_sweep.sh --dry-run

# 正式提交
bash nebula_scripts/submit_tasd_ema_sweep.sh
```

### 修改扫描范围

直接编辑对应 `submit_*.sh` 文件中的超参数组，注释掉不需要的取值即可：

```bash
ENTROPY_COEFF_LIST=(
    # "0.05"   # 注释掉不跑
    "0.1"
    "1.0"
)
```

### 调整 Docker 镜像

每个 submit 脚本顶部有 `CUSTOM_DOCKER_IMAGE` 变量，默认使用 `sdpo_env` 自定义镜像：

```bash
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
```

留空则回退到 Nebula 默认 `pytorch260` 镜像。

---

## 执行流程

```
submit_*.sh
  └─ nebulactl run mdl → Nebula 平台
       └─ entry.py --script_path=xxx --world_size=N
            └─ launch_ray_cluster.sh
                 ├─ 激活 sdpo_env conda 环境
                 ├─ 设置 PYTHONPATH / CUDA / vLLM 等环境变量
                 ├─ ray start（head 或 worker）
                 └─ [Rank 0] bash <训练脚本>.sh
                      └─ python -m verl.trainer.main_ppo --config-name tasd/grpo/sdpo
```

---

## 数据集与模型路径（OSS）

| 类型 | OSS 路径 |
|------|---------|
| sciknoweval/biology | `datasets/sciknoweval/biology/{train,test}.parquet` |
| sciknoweval/chemistry | `datasets/sciknoweval/chemistry/{train,test}.parquet` |
| 基底模型 Qwen3-8B | `base_models/Qwen3-8B/` |
| checkpoint 保存 | `models/{JOB_NAME}/` |
| SwanLab 日志 | `logs/swanlab_logs/` |

OSS 根路径：`/data/oss_bucket_0/ad/loujieming.ljm`
