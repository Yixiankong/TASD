# nebulactl 任务提交指南

本文档是 nebulactl 任务提交的一站式操作手册，覆盖从环境准备到任务管理的全流程。

> 各算法的实验说明、超参数定义和 SwanLab 项目信息，请参阅 [README.md](README.md)。

---

## 1. 概述

**nebulactl** 是 Nebula 平台的 CLI 工具（当前版本 1.1.25），用于提交、管理和监控分布式训练任务。

本项目使用的核心子命令：

```bash
nebulactl run mdl    # 提交 MDL（Machine Deep Learning）任务
```

任务提交流程概览：

```
本地 shell 脚本 → nebulactl run mdl → Nebula 平台容器 → 入口脚本 → 训练
```

---

## 2. 前置准备

### 2.1 环境变量

提交前需在本地 shell 中 export 以下变量：

```bash
export OPENLM_TOKEN="..."         # Nebula 平台认证 token
export OSS_ACCESS_ID="..."        # 阿里云 OSS Access Key ID
export OSS_ACCESS_KEY="..."       # 阿里云 OSS Access Key Secret
export SWANLAB_API_KEY="..."      # SwanLab 实验跟踪 API Key（可选，有内置 fallback）
```

### 2.2 用户配置文件（可选）

`~/.nebulactl/config.ini` 可存储默认配置，避免每次传参：

```ini
user_id=xxx
user_name=xxx
nebula_project=xxx
nebula_workspace=xxx
access_id=xxx
access_key=xxx
```

优先级：CLI 参数 > 环境变量（`NEBULA_PROJECT` 等） > 配置文件。

### 2.3 Python 依赖

`requirements_nebula.txt` 在任务启动时由 Nebula 自动 `pip install`。仅包含镜像中未预装的依赖（如 `vllm>=0.9.0`、`swanlab`、`hydra-core`、`trl>=1.6.0` 等）。

---

## 3. `nebulactl run mdl` 命令详解

### 3.1 完整命令模板

```bash
nebulactl run mdl \
    --force \
    --engine=xdl \
    --queue=${QUEUE} \
    --entry=nebula_scripts/entry.py \
    --user_params="${options}" \
    --worker_count=${WORLD_SIZE} \
    --file.cluster_file=${CLUSTER_FILE} \
    --job_name=${JOB_NAME} \
    --access_id=${access_id} \
    --access_key=${access_key} \
    --env=OPENLM_TOKEN=${OPENLM_TOKEN} \
    --env=SWANLAB_API_KEY=${SWANLAB_API_KEY} \
    --custom_docker_image=${IMAGE} \
    --requirements_file_name=requirements_nebula.txt \
    --oss_access_id=${OSS_ACCESS_ID} \
    --oss_access_key=${OSS_ACCESS_KEY} \
    --oss_bucket=${OSS_BUCKET} \
    --oss_endpoint=${OSS_ENDPOINT}
```

### 3.2 参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--force` | 覆盖同名已有任务；跳过工作目录 >1GB 的警告 | — |
| `--engine=xdl` | 使用 XDL 分布式训练引擎（本项目固定值） | `xdl` |
| `--queue` | GPU 集群队列名称 | `lazada_llm_ad_h20`（H20）、`ae_h100`（H100） |
| `--entry` | Nebula 容器内的 Python 入口脚本 | `nebula_scripts/entry.py` 或 `nebula_scripts/dpo_kto_entry.py` |
| `--user_params` | 传递给入口脚本的参数字符串 | `"--script_path=... --world_size=1 --job_name=..."` |
| `--worker_count` | 节点数（每个节点 GPU 数由 cluster_file 决定） | `1`（单节点）、`2`（多节点 16 卡） |
| `--file.cluster_file` | 集群资源配置 JSON 文件路径 | `nebula_scripts/cluster.json`（8 GPU）或 `nebula_scripts/cluster_gpu_4.json`（4 GPU） |
| `--job_name` | 任务唯一名称 | `TASD-bio-lr1e-5-rtteacher_prob-...-20260628_143000` |
| `--access_id` / `--access_key` | Nebula 平台身份认证凭据 | — |
| `--env=KEY=VALUE` | 注入容器环境变量（可多次使用） | `--env=LR=1e-5 --env=ENTROPY_COEFF=0.1` |
| `--custom_docker_image` | 自定义 Docker 镜像地址 | `hub.docker.alibaba-inc.com/mdl/notebook_saved:...` |
| `--algo_name` | 默认镜像名（与 `--custom_docker_image` 二选一） | `pytorch260` |
| `--requirements_file_name` | 容器启动时安装的 pip 依赖文件 | `requirements_nebula.txt` |
| `--oss_access_id` / `--oss_access_key` | 阿里云 OSS 认证凭据 | — |
| `--oss_bucket` | OSS Bucket 名称 | `lazada-ai-model` |
| `--oss_endpoint` | OSS Endpoint 地址 | `oss-cn-hangzhou-zmf.aliyuncs.com` |

### 3.3 Docker 镜像选择

```bash
# 使用自定义镜像（推荐，包含 sdpo_env conda 环境）
CUSTOM_DOCKER_IMAGE="hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942"

# 使用默认镜像（留空 CUSTOM_DOCKER_IMAGE 时）
--algo_name=pytorch260
```

选择逻辑：`CUSTOM_DOCKER_IMAGE` 非空时用 `--custom_docker_image`，否则用 `--algo_name=pytorch260`。

---

## 4. 执行流程

### 4.1 RL 训练（GRPO / SDPO / TASD / CV-SDPO）

```
┌─────────────────────────────────────────────────────────────────┐
│  本地机器                                                        │
│                                                                 │
│  submit_*.sh                                                    │
│    └─ nebulactl run mdl                                         │
│         ├─ 打包当前目录为 zip（排除 .git）                         │
│         ├─ 上传到 OSS                                            │
│         └─ 提交到 NOP（Nebula One Proxy）                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Nebula 容器                                                     │
│                                                                 │
│  entry.py                                                       │
│    ├─ 解析 --script_path, --world_size, --job_name               │
│    ├─ 将 --env KEY=VALUE 注入 os.environ                         │
│    └─ 调用 launch_ray_cluster.sh                                 │
│         ├─ 激活 sdpo_env conda 环境                               │
│         ├─ 设置 PYTHONPATH、CUDA_VISIBLE_DEVICES                  │
│         ├─ 设置 vLLM/PyTorch 环境变量                              │
│         ├─ 禁用 DeepSpeed triton ops（避免 Ray worker 报错）       │
│         ├─ 清理 PATH/LD_LIBRARY_PATH 重复项（防止 std::length_error）│
│         ├─ ray start --head（Rank 0）或 --address（其他节点）      │
│         └─ [Rank 0] bash <训练脚本>.sh                            │
│              └─ python -m verl.trainer.main_ppo                   │
│                  --config-name tasd/grpo/sdpo/cv_sdpo             │
│                  <Hydra 覆盖参数（来自环境变量）>                    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 离线训练（DPO / KTO）

与 RL 训练的区别：

- 入口脚本为 `dpo_kto_entry.py`（不启动 Ray）
- 直接激活 conda → 执行训练脚本 → `accelerate launch` 处理分布式

```
submit_dpo_sweep.sh / submit_kto_sweep.sh
  └─ nebulactl run mdl --entry=nebula_scripts/dpo_kto_entry.py
       └─ dpo_kto_entry.py
            ├─ 注入环境变量
            ├─ 激活 sdpo_env conda
            ├─ 清理 PATH/LD_LIBRARY_PATH
            └─ bash <训练脚本>.sh
                 └─ accelerate launch ...
```

---

## 5. 集群配置

### 5.1 配置文件

**`cluster.json`** — 8 GPU / 节点：

```json
{
    "worker": {
      "cpu": 4000,
      "gpu": 800,
      "memory": 120000
    }
}
```

**`cluster_gpu_4.json`** — 4 GPU / 节点：

```json
{
    "worker": {
      "cpu": 1600,
      "gpu": 400,
      "memory": 160000
    }
}
```

### 5.2 资源单位

| 字段 | 单位 | 说明 |
|------|------|------|
| `cpu` | millicore | 4000 = 4 CPU 核心 |
| `gpu` | millicore | 800 = 8 GPU、400 = 4 GPU |
| `memory` | MB | 120000 = 120 GB |

### 5.3 多节点配置

总 GPU 数 = `worker_count` × 每节点 GPU 数：

| 场景 | worker_count | cluster_file | 总 GPU |
|------|:---:|------|:---:|
| 单节点 4 GPU | 1 | `cluster_gpu_4.json` | 4 |
| 单节点 8 GPU | 1 | `cluster.json` | 8 |
| 双节点 16 GPU | 2 | `cluster.json` | 16 |

> `submit_job.sh` 默认逻辑：`WORLD_SIZE > 1` 时使用 `cluster.json`，`WORLD_SIZE = 1` 时使用 `cluster_gpu_4.json`。

---

## 6. 超参数传递机制

超参数通过 `--env=KEY=VALUE` 在 nebulactl 命令行注入，容器内的训练脚本通过 shell 变量消费。

### 6.1 传递链路

```
submit_*.sh                          entry.py                        parametric.sh
───────────                          ────────                        ─────────────
--env=LR=1e-5          →    env["LR"] = "1e-5"       →    LR=${LR:-1e-5}
--env=ENTROPY_COEFF=0.1       env["ENTROPY_COEFF"]="0.1"     hydra override: entropy_coeff=$ENTROPY_COEFF
```

### 6.2 默认值语法

训练脚本使用 `${VAR:-default}` 为未传递的变量提供默认值：

```bash
# parametric.sh 中
LR="${LR:-1e-5}"                    # 未传 LR 时默认 1e-5
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"   # CV-SDPO 传 16，SDPO 不传则默认 32
```

### 6.3 可选参数语法

使用 `${VAR:+...}` 仅在变量非空时生成 Hydra 覆盖：

```bash
# 仅当 CV_GAMMA 有值时才传递
${CV_GAMMA:+algorithm.cv_gamma=$CV_GAMMA}
```

### 6.4 完整示例

```bash
# submit 脚本中定义扫描变量
LR_LIST=("1e-5" "5e-6")
ENTROPY_COEFF_LIST=("0.0" "0.1")

for LR in "${LR_LIST[@]}"; do
  for ENTROPY_COEFF in "${ENTROPY_COEFF_LIST[@]}"; do
    # 组装 nebulactl 命令
    nebulactl run mdl \
        --env=LR=${LR} \
        --env=ENTROPY_COEFF=${ENTROPY_COEFF} \
        --env=CONFIG_NAME=tasd \
        ...
  done
done
```

容器内 `tasd_sciknoweval_parametric.sh` 消费：

```bash
LR="${LR:-1e-5}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
CONFIG_NAME="${CONFIG_NAME:-tasd}"

python -m verl.trainer.main_ppo --config-name ${CONFIG_NAME} \
    actor_rollout_ref.actor.optim.lr=$LR \
    algorithm.entropy_coeff=$ENTROPY_COEFF
```

---

## 7. 入口脚本对比

| 特性 | `entry.py` | `dpo_kto_entry.py` |
|------|-----------|-------------------|
| 适用算法 | GRPO、SDPO、CV-SDPO、TASD | DPO、KTO |
| 分布式框架 | Ray + vLLM | HuggingFace Accelerate |
| 启动流程 | 调用 `launch_ray_cluster.sh` 启动 Ray 集群 | 直接执行训练脚本 |
| 多节点支持 | Ray head/worker 自动协商 | Accelerate 自动配置 |
| conda 环境 | 在 launch_ray_cluster.sh 中激活 | 在 entry 脚本中直接激活 |
| GPU 检测 | `launch_ray_cluster.sh` 中 `nvidia-smi` | entry 脚本中 `nvidia-smi` |

---

## 8. 任务管理常用命令

### 8.1 提交任务

```bash
# 通过封装脚本提交（推荐）
bash nebula_scripts/submit_tasd_ema_sweep.sh

# dry-run：只打印配置，不实际提交
bash nebula_scripts/submit_tasd_ema_sweep.sh --dry-run

# 通用单任务提交
bash nebula_scripts/submit_job.sh \
    nebula_scripts/tasd/tasd_sciknoweval_qwen3_8B.sh \
    1 \
    lazada_llm_ad_h20
```

### 8.2 停止任务

```bash
nebulactl stop <task_id> --task_type=train
```

### 8.3 查看模型信息

```bash
# 查看模型详情
nebulactl desc mdl --name <model_name>

# 列出模型版本
nebulactl list model_version --name <model_name>

# 列出 checkpoint
nebulactl list model_ckpt --name <model_name>
```

### 8.4 下载 checkpoint

```bash
nebulactl download model_ckpt --name <model_name> --ckpt_id <id>
```

---

## 9. 提交脚本速查表

### 通用脚本

| 脚本 | 说明 |
|------|------|
| `submit_job.sh` | 通用单任务提交（接受脚本路径、节点数、队列作为参数） |

### RL 训练（入口：`entry.py`）

| 脚本 | 算法 | 说明 |
|------|------|------|
| `submit_tasd_ema_sweep.sh` | TASD | EMA Teacher 超参扫描 |
| `submit_tasd_lcb_sweep.sh` | TASD | LiveCodeBench 数据集 |
| `submit_tasd_stability_sweep.sh` | TASD | 稳定性实验 |
| `submit_tasd_relative_sweep.sh` | TASD | Relative reward 扫描 |
| `submit_tasd_future_kl_sweep.sh` | TASD | Future-KL 调制扫描 |
| `submit_tasd_hybrid_adv_sweep.sh` | TASD | Hybrid advantage 扫描 |
| `submit_tasd_hybrid_adv_lcb_sweep.sh` | TASD | Hybrid advantage + LCB |
| `submit_tasd_distill_temp_sweep.sh` | TASD | 蒸馏温度扫描 |
| `submit_tasd_teacher_gae_sweep.sh` | TASD | Teacher-GAE 扫描 |
| `submit_tasd_p0_feasibility.sh` | TASD | Phase 0 可行性验证 |
| `submit_tasd_p1_comparison.sh` | TASD | Phase 1 对比实验 |
| `submit_tasd_p2_ablation.sh` | TASD | Phase 2 消融实验 |
| `submit_grpo_baseline_sweep.sh` | GRPO | GRPO 基线扫描 |
| `submit_grpo_lcb_sweep.sh` | GRPO | GRPO + LiveCodeBench |
| `submit_sdpo_baseline_sweep.sh` | SDPO | SDPO 基线扫描 |
| `submit_sdpo_ew_sweep.sh` | SDPO+EW | SDPO + Entropy Weighting |
| `submit_sdpo_lcb_sweep.sh` | SDPO | SDPO + LiveCodeBench |
| `submit_cv_sdpo_sweep.sh` | CV-SDPO | Control Variate SDPO |
| `submit_cv_sdpo_gamma1_sweep.sh` | CV-SDPO | CV-SDPO γ=1.0 |

### 离线训练（入口：`dpo_kto_entry.py`）

| 脚本 | 算法 | 说明 |
|------|------|------|
| `submit_dpo_sweep.sh` | DPO | DPO 超参扫描 |
| `submit_kto_sweep.sh` | KTO | KTO 超参扫描 |

---

## 10. 常见问题排查

### 提交失败

| 现象 | 原因 | 解决方案 |
|------|------|---------|
| `OPENLM_TOKEN not set` | 未 export 环境变量 | `export OPENLM_TOKEN="..."` |
| 工作目录 >1GB 警告 | 代码目录过大 | 添加 `--force` 参数 |
| 队列无可用资源 | 队列繁忙 | 换队列（`lazada_llm_ad_h20` ↔ `ae_h100`）或稍后重试 |
| 提交无报错但任务未启动 | 认证失败 | 检查 `access_id` / `access_key` 是否正确 |

### 容器内运行失败

| 现象 | 原因 | 解决方案 |
|------|------|---------|
| `std::length_error` | PATH/LD_LIBRARY_PATH 过长（重复条目） | 已在 `launch_ray_cluster.sh` 中通过 `clean_path()` 自动去重 |
| DeepSpeed `0 active drivers` | Ray worker 导入时无法检测 GPU | 已在 `launch_ray_cluster.sh` 中设置 `DS_BUILD_OPS=0` 和 `DS_SKIP_CUDA_CHECK=1` |
| `ModuleNotFoundError: verl` | PYTHONPATH 未设置 | 已在 `launch_ray_cluster.sh` 中 `export PYTHONPATH=$(pwd):$PYTHONPATH` |
| conda 环境未激活 | 容器中找不到正确的 Python | 确认自定义镜像包含 `sdpo_env` 环境，或使用 `--algo_name=pytorch260` |
| OOM（显存不足） | CV-SDPO 3× teacher forward 占用过多 | 降低 `PPO_MINI_BATCH_SIZE`（如 16）和 `GPU_MEMORY_UTIL`（如 0.75） |

### 任务命名建议

```
{算法}-{数据集}-{关键超参}-{模型}-{时间戳}
```

示例：
- `TASD-bio-lr1e-5-rtteacher_prob-nostd-clip5.0-ent0.1-ema0.05-Qwen3-8B-20260628_143000`
- `GRPO-bio-mbs32-train32-lr1e-5-Qwen3-8B-20260628_150000`
- `SDPO-bio-train32-alpha0.5-lr1e-5-drossTrue-Qwen3-8B-20260628_153000`

---

## 附录：OSS 路径约定

| 类型 | OSS 路径 |
|------|---------|
| 数据集 | `datasets/<dataset_name>/{train,test}.parquet` |
| 基底模型 | `base_models/<model_name>/` |
| Checkpoint | `models/<JOB_NAME>/` |
| SwanLab 日志 | `logs/swanlab_logs/` |

OSS 根路径：`/data/oss_bucket_0/ad/loujieming.ljm`  
Bucket：`lazada-ai-model`  
Endpoint：`oss-cn-hangzhou-zmf.aliyuncs.com`
