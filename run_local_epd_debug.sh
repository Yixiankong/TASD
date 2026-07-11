#!/bin/bash

# Usage: ./run_local_epd_debug.sh [experiment_name_suffix]
# =============================================================================
# EPD 本地 debug 训练脚本
# 使用小模型 + 缩小规模，快速验证 EPD 训练流程
# =============================================================================

CONFIG_NAME="sdpo"

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
LOG_ROOT="/data/oss_bucket_0/ad/kongyixian.kyx/TASD"

# 小模型
MODEL_NAME="Qwen3-4B"
MODEL_PATH="${OSS_ROOT}/base_models/${MODEL_NAME}"

# 数据集：biology（MCQ，450 条训练数据）
DATASET="sciknoweval/biology"

# ── 缩小规模 ──────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE=4          # 生产用 32
ROLLOUT_N=2                 # 生产用 8
PPO_MINI_BATCH_SIZE=4       # 生产用 32
VAL_N=2                     # 生产用 16
MAX_PROMPT_LENGTH=512       # 生产用 2048
MAX_RESPONSE_LENGTH=1024    # 生产用 8192
MAX_MODEL_LEN=4096          # 生产用 18944
TOTAL_TRAINING_STEPS=5      # 只跑 5 步 debug
TOTAL_EPOCHS=1

# ── EPD 超参 ──────────────────────────────────────────────────────────
EPD_LAMBDA=0.8
EPD_TAU=0.5

# ── SDPO 超参 ─────────────────────────────────────────────────────────
LR=1e-5
ALPHA=0.5
DONT_REPROMPT=True
TOPK=100

SUFFIX=${1:-"local_epd_debug"}
N_GPUS=1

# =============================================================================
# SETUP
# =============================================================================

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

# 激活训练环境
source activate sdpo_env 2>/dev/null || conda activate sdpo_env 2>/dev/null || true

export WANDB_MODE=offline
export WANDB_DIR=${LOG_ROOT}/logs/wandb
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-3sKfdi20C8rYk5JQs0fOJ}"
export SWANLAB_LOG_DIR=${LOG_ROOT}/logs/swanlab

# =============================================================================
# EXECUTION
# =============================================================================

EXP_NAME="LOCAL-EPD-DEBUG-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.train_files=${OSS_ROOT}/datasets/${DATASET}/train.parquet \
data.val_files=${OSS_ROOT}/datasets/${DATASET}/test.parquet \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
max_model_len=$MAX_MODEL_LEN \
custom_reward_function.path=${PROJECT_ROOT}/verl/utils/reward_score/feedback/__init__.py \
trainer.group_name=EPD-debug-local \
trainer.n_gpus_per_node=$N_GPUS \
trainer.total_epochs=$TOTAL_EPOCHS \
trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
trainer.val_before_train=False \
trainer.test_freq=100 \
trainer.save_freq=-1 \
actor_rollout_ref.rollout.n=$ROLLOUT_N \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.max_num_seqs=256 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.optim.lr_warmup_steps=2 \
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
actor_rollout_ref.actor.self_distillation.distillation_topk=$TOPK \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONT_REPROMPT} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.entropy_weighting=True \
actor_rollout_ref.actor.self_distillation.entropy_weighting_version=v5_epd \
actor_rollout_ref.actor.self_distillation.epd_lambda=${EPD_LAMBDA} \
actor_rollout_ref.actor.self_distillation.epd_tau=${EPD_TAU} \
algorithm.rollout_correction.rollout_is=token \
trainer.logger=[console,swanlab]"

LOG_DIR="${LOG_ROOT}/logs/training_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "EPD Debug Training (小规模快速验证)"
echo "============================================================"
echo "Experiment:    $EXP_NAME"
echo "Model:         $MODEL_PATH"
echo "Dataset:       $DATASET"
echo "Batch size:    $TRAIN_BATCH_SIZE (production: 32)"
echo "Rollout n:     $ROLLOUT_N (production: 8)"
echo "Max seq len:   $MAX_MODEL_LEN (production: 18944)"
echo "Total steps:   $TOTAL_TRAINING_STEPS"
echo "EPD lambda:    $EPD_LAMBDA"
echo "EPD tau:       $EPD_TAU"
echo "GPUs:          $N_GPUS"
echo "Log file:      $LOG_FILE"
echo "============================================================"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATASET" $ARGS 2>&1 | tee "$LOG_FILE"
