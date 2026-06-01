#!/bin/bash

# Usage: ./run_local_cv_sdpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="cv_sdpo"

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# Default to sciknoweval/biology (MCQ format, compatible with CV-SDPO answer extraction)
DATASET="sciknoweval/biology"

# Hyperparameters (aligned with SDPO baseline)
TRAIN_BATCH_SIZE=8
ROLLOUT_BATCH_SIZE=4
LR=1e-5
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
MODEL_NAME="Qwen3-4B"
MODEL_PATH="${OSS_ROOT}/base_models/${MODEL_NAME}"
N_GPUS=1

# CV-SDPO specific
CV_GAMMA=0.5
BETA_MODE="adaptive"

# Allow overriding experiment name suffix
SUFFIX=${1:-"local_cv_sdpo"}

# =============================================================================
# SETUP
# =============================================================================

source activate sdpo_env 2>/dev/null || conda activate sdpo_env

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}
export WANDB_MODE=offline
export WANDB_DIR=/data/oss_bucket_0/ad/kongyixian.kyx/logs/wandb
export TENSORBOARD_DIR=/data/oss_bucket_0/ad/kongyixian.kyx/logs/tensorboard
export SWANLAB_LOG_DIR=/data/oss_bucket_0/ad/kongyixian.kyx/logs/swanlab

# =============================================================================
# EXECUTION
# =============================================================================

EXP_NAME="LOCAL-CV_SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-gamma${CV_GAMMA}-beta${BETA_MODE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.train_files=${OSS_ROOT}/datasets/${DATASET}/train.parquet \
data.val_files=${OSS_ROOT}/datasets/${DATASET}/test.parquet \
custom_reward_function.path=${PROJECT_ROOT}/verl/utils/reward_score/feedback/__init__.py \
trainer.group_name=CV-SDPO-local \
trainer.n_gpus_per_node=$N_GPUS \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=8 \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.leakage_decontamination_enabled=True \
actor_rollout_ref.actor.self_distillation.leakage_decontamination_mode=control_variate \
actor_rollout_ref.actor.self_distillation.cv_gamma=$CV_GAMMA \
actor_rollout_ref.actor.self_distillation.beta_mode=$BETA_MODE \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16 \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.max_num_seqs=256 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
trainer.logger=[console,swanlab,wandb,tensorboard] \
trainer.rollout_data_dir=/data/oss_bucket_0/ad/kongyixian.kyx/logs/rollout_data/${SUFFIX}"

LOG_DIR="/data/oss_bucket_0/ad/kongyixian.kyx/logs/training_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "----------------------------------------------------------------"
echo "Starting Local CV-SDPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATASET"
echo "Model: $MODEL_PATH"
echo "CV-SDPO: gamma=$CV_GAMMA, beta_mode=$BETA_MODE"
echo "Log file: $LOG_FILE"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATASET" $ARGS 2>&1 | tee "$LOG_FILE"
