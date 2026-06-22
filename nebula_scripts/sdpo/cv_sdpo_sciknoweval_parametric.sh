#!/usr/bin/env bash
# =============================================================================
# CV-SDPO 参数化训练脚本（供 Nebula sweep 调用）
# 基于 sdpo_sciknoweval_parametric.sh，增加 CV-SDPO 特有参数
# 所有超参通过 nebulactl --env 注入
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
LOG_ROOT="${LOG_ROOT:-/data/oss_bucket_0/ad/kongyixian.kyx/TASD}"

# ── 从环境变量读取超参 ────────────────────────────────────────────────
check_env() { val=$(eval echo "\$$1"); [ -n "$val" ] || { echo "ERROR: $1 is not set. Aborting."; exit 1; }; }
check_env DATASET
check_env LR
check_env ALPHA
check_env DONT_REPROMPT_ON_SELF_SUCCESS
check_env TRAIN_BATCH_SIZE
check_env ROLLOUT_N
check_env MODEL_NAME

# 数据集路径（使用 DATASET 环境变量）
train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
model_path="${OSS_ROOT}/base_models/${MODEL_NAME}"
save_path="${LOG_ROOT}/models/${JOB_NAME:-cv_sdpo_sweep}"

# ── 环境 ──────────────────────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
unset VLLM_ATTENTION_BACKEND  # 与 verl_training.sh 行为一致，避免平台注入的值影响 attention 计算
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team
export WANDB_DIR="${LOG_ROOT}/logs/wandb"
export TENSORBOARD_DIR="${LOG_ROOT}/logs/tensorboard"
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal}"
export SWANLAB_LOG_DIR="${LOG_ROOT}/logs/swanlab"
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

pip install -e . --no-deps --no-build-isolation --quiet 2>/dev/null || true

python -m verl.trainer.main_ppo \
    --config-name ${CONFIG_NAME:-cv_sdpo} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32} \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.alpha=${ALPHA} \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONT_REPROMPT_ON_SELF_SUCCESS} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.self_distillation.leakage_decontamination_enabled=True \
    actor_rollout_ref.actor.self_distillation.leakage_decontamination_mode=control_variate \
    ${CV_GAMMA:+actor_rollout_ref.actor.self_distillation.cv_gamma=${CV_GAMMA}} \
    ${BETA_MODE:+actor_rollout_ref.actor.self_distillation.beta_mode=${BETA_MODE}} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL:-0.85} \
    algorithm.rollout_correction.rollout_is=token \
    trainer.total_epochs=30 \
    trainer.total_training_steps=250 \
    trainer.save_freq=-1 \
    trainer.save_best_metric="val-core/sciknoweval/acc/mean@16" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-CV_SDPO_v1}" \
    trainer.experiment_name="${JOB_NAME:-cv_sdpo_sweep}" \
    trainer.group_name="CV-SDPO-generalization" \
    trainer.rollout_data_dir="${LOG_ROOT}/logs/rollout_data/${JOB_NAME:-cv_sdpo_sweep}" \
    "trainer.logger=[console,swanlab,wandb,tensorboard]"
