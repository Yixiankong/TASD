#!/bin/bash
# KTO Nebula parametric training script
# Called by submit_kto_sweep.sh

set -eo pipefail

# ── 动态检测 REPO_ROOT（Nebula CWD 通常是代码仓库根目录）──────────────
REPO_ROOT="$(pwd)"
echo "REPO_ROOT = ${REPO_ROOT}"

# Verify required environment variables
: "${DATASET:?DATASET is not set}"
: "${MODEL:?MODEL is not set}"
: "${LR:?LR is not set}"
: "${BETA:?BETA is not set}"
: "${NUM_EPOCHS:?NUM_EPOCHS is not set}"
: "${BATCH_SIZE:?BATCH_SIZE is not set}"
: "${LOSS_TYPE:?LOSS_TYPE is not set}"

# Optional parameters with defaults (optimized based on local testing)
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_LENGTH=${MAX_LENGTH:-4096}
LOGGING_STEPS=${LOGGING_STEPS:-5}
SAVE_STEPS=${SAVE_STEPS:-100}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
DESIRABLE_WEIGHT=${DESIRABLE_WEIGHT:-1.0}
UNDESIRABLE_WEIGHT=${UNDESIRABLE_WEIGHT:-1.0}
THINK_MODE=${THINK_MODE:-"nothink"}
RESUME_FROM=${RESUME_FROM:-""}

echo "================================================================"
echo "KTO Nebula Training"
echo "================================================================"
echo "Dataset:      ${DATASET}"
echo "Model:        ${MODEL}"
echo "LR:           ${LR}"
echo "Beta:         ${BETA}"
echo "Epochs:       ${NUM_EPOCHS}"
echo "Batch Size:   ${BATCH_SIZE}"
echo "Grad Accum:   ${GRAD_ACCUM}"
echo "Max Length:   ${MAX_LENGTH}"
echo "Loss Type:    ${LOSS_TYPE}"
echo "Think Mode:   ${THINK_MODE}"
echo "Desirable W:  ${DESIRABLE_WEIGHT}"
echo "Undesirable W:${UNDESIRABLE_WEIGHT}"
echo "Logging Step: ${LOGGING_STEPS}"
echo "Save Steps:   ${SAVE_STEPS}"
if [ -n "$RESUME_FROM" ]; then
    echo "Resume From:  ${RESUME_FROM}"
fi
echo "================================================================"

# Activate conda environment
CONDA_ENV_NAME="dpo_env"
CONDA_ENV_BIN="/opt/conda/envs/${CONDA_ENV_NAME}/bin"
if [ -d "${CONDA_ENV_BIN}" ]; then
    export PATH="${CONDA_ENV_BIN}:${PATH}"
    echo "Activated conda env: ${CONDA_ENV_NAME}"
fi

# TRL version check (>= 1.6.0)
python3 -c "
import trl
version = tuple(map(int, trl.__version__.split('.')[:2]))
if version < (1, 6):
    print(f'ERROR: TRL >= 1.6.0 required, got {trl.__version__}')
    exit(1)
print(f'TRL version: {trl.__version__} ✓')
"

# Data paths (use THINK_MODE to select think/nothink data)
TRAIN_DATA="${DATASET}/kto_train_${THINK_MODE}.parquet"
EVAL_DATA="${DATASET}/kto_train_${THINK_MODE}_test.parquet"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found: $TRAIN_DATA"
    exit 1
fi

# Extract model basename for naming
MODEL_SHORT=$(basename "$MODEL")

# ── NCCL / 分布式训练环境变量（参考 launch_ray_cluster.sh）─────────────
export NCCL_DEBUG=WARN                    # 诊断多卡通信问题
export NCCL_TIMEOUT=1800                  # 30 分钟超时，防止多卡同步卡死
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0 # 抑制无用警告
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal}"

export WANDB_MODE=offline
export WANDB_DIR=${REPO_ROOT}/logs/wandb
export SWANLAB_LOG_DIR=${REPO_ROOT}/logs/swanlab
export TENSORBOARD_DIR=${REPO_ROOT}/logs/tensorboard

# Output directory
OUTPUT_DIR="${REPO_ROOT}/checkpoints/KTO-${MODEL_SHORT}-${THINK_MODE}-beta${BETA}-lr${LR}-bs${BATCH_SIZE}-${LOSS_TYPE}"
mkdir -p "$OUTPUT_DIR"

# Determine GPU count
N_GPUS=${N_GPUS:-1}
if command -v nvidia-smi &> /dev/null; then
    N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
fi

echo "Using ${N_GPUS} GPUs"

# Build training arguments
ARGS="--model_name_or_path ${MODEL} \
--train_data_path ${TRAIN_DATA} \
--eval_data_path ${EVAL_DATA} \
--beta ${BETA} \
--desirable_weight ${DESIRABLE_WEIGHT} \
--undesirable_weight ${UNDESIRABLE_WEIGHT} \
--loss_type ${LOSS_TYPE} \
--learning_rate ${LR} \
--num_train_epochs ${NUM_EPOCHS} \
--per_device_train_batch_size ${BATCH_SIZE} \
--gradient_accumulation_steps ${GRAD_ACCUM} \
--max_length ${MAX_LENGTH} \
--warmup_ratio 0.1 \
--output_dir ${OUTPUT_DIR} \
--use_swanlab \
--swanlab_project TASD-KTO-Nebula \
--swanlab_run_name KTO-${MODEL_SHORT}-${THINK_MODE}-beta${BETA}-lr${LR}-bs${BATCH_SIZE}-${LOSS_TYPE} \
--bf16 \
--gradient_checkpointing \
--save_steps ${SAVE_STEPS} \
--eval_steps ${SAVE_STEPS} \
--logging_steps ${LOGGING_STEPS} \
--save_total_limit ${SAVE_TOTAL_LIMIT} \
--upload_data_to_swanlab"

# Add resume_from_checkpoint if specified
if [ -n "$RESUME_FROM" ]; then
    ARGS="$ARGS --resume_from_checkpoint $RESUME_FROM"
fi

# Launch training
if [ "$N_GPUS" -gt 1 ]; then
    # Multi-GPU with Accelerate FSDP
    accelerate launch \
        --config_file ${REPO_ROOT}/configs/accelerate/multi_gpu_fsdp.yaml \
        ${REPO_ROOT}/training/kto_train.py $ARGS
else
    # Single GPU
    accelerate launch \
        --config_file ${REPO_ROOT}/configs/accelerate/single_gpu.yaml \
        ${REPO_ROOT}/training/kto_train.py $ARGS
fi

echo "KTO training completed successfully"
