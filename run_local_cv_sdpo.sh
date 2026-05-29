#!/bin/bash

# Usage: ./run_local_cv_sdpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="cv_sdpo"

# Default to sciknoweval/biology (MCQ format, compatible with CV-SDPO answer extraction)
DATA_PATH="datasets/sciknoweval/biology"

# Hyperparameters (aligned with SDPO baseline)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-5
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
N_GPUS=1

# CV-SDPO specific
CV_GAMMA=0.5
BETA_MODE="adaptive"

# Allow overriding experiment name suffix
SUFFIX=${1:-"local_cv_sdpo"}

# =============================================================================
# SETUP
# =============================================================================

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="LOCAL-CV_SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-gamma${CV_GAMMA}-beta${BETA_MODE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.train_files=[${PROJECT_ROOT}/${DATA_PATH}/train.parquet] \
data.val_files=[${PROJECT_ROOT}/${DATA_PATH}/test.parquet] \
custom_reward_function.path=${PROJECT_ROOT}/verl/utils/reward_score/feedback/__init__.py \
trainer.group_name=CV-SDPO-local \
trainer.n_gpus_per_node=$N_GPUS \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.leakage_decontamination_enabled=True \
actor_rollout_ref.actor.self_distillation.leakage_decontamination_mode=control_variate \
actor_rollout_ref.actor.self_distillation.cv_gamma=$CV_GAMMA \
actor_rollout_ref.actor.self_distillation.beta_mode=$BETA_MODE \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16"

echo "----------------------------------------------------------------"
echo "Starting Local CV-SDPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "CV-SDPO: gamma=$CV_GAMMA, beta_mode=$BETA_MODE"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
