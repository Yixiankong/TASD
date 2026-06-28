#!/bin/bash
# =============================================================================
# Local DPO Training — Qwen3-4B on single H20 GPU
# Usage: ./run_local_dpo.sh [experiment_suffix]
# =============================================================================
set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
LOG_ROOT="/data/oss_bucket_0/ad/kongyixian.kyx/TASD"

# Dataset — 根据第二个参数选择 think/nothink 数据
# 用法: bash run_local_dpo.sh [experiment_suffix] [think|nothink]
THINK_MODE=${2:-"nothink"}  # 默认 nothink
DATA_DIR="/data/oss_bucket_0/ad/kongyixian.kyx/dpo/dataset_d769a815e7a5"
TRAIN_DATA="${DATA_DIR}/dpo_train_${THINK_MODE}.parquet"
EVAL_DATA="${DATA_DIR}/dpo_train_${THINK_MODE}_test.parquet"

# Model
MODEL_NAME="Qwen3-4B"
MODEL_PATH="${OSS_ROOT}/base_models/${MODEL_NAME}"

# DPO Hyperparameters
BETA=0.1
LOSS_TYPE="sigmoid"
LABEL_SMOOTHING=0.0

# Training
LR=5e-7
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
MAX_LENGTH=4096
TRUNCATION_MODE="keep_start"
WARMUP_RATIO=0.1

# LoRA (set to true to enable - reduces memory by training only adapters)
USE_LORA=false
LORA_R=16

SUFFIX=${1:-"local_dpo"}

# =============================================================================
# SETUP
# =============================================================================
source activate sdpo_env 2>/dev/null || conda activate sdpo_env 2>/dev/null || true

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export WANDB_MODE=offline
export WANDB_DIR=${LOG_ROOT}/logs/wandb
export SWANLAB_LOG_DIR=${LOG_ROOT}/logs/swanlab
export TENSORBOARD_DIR=${LOG_ROOT}/logs/tensorboard

# =============================================================================
# EXECUTION
# =============================================================================
EXP_NAME="DPO-${MODEL_NAME}-beta${BETA}-lr${LR}-bs${BATCH_SIZE}x${GRAD_ACCUM}-len${MAX_LENGTH}-${THINK_MODE}-${SUFFIX}"
OUTPUT_DIR="${LOG_ROOT}/checkpoints/${EXP_NAME}"
LOG_DIR="${LOG_ROOT}/logs/training_logs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

ACCELERATE_CONFIG="${PROJECT_ROOT}/configs/accelerate/single_gpu.yaml"

ARGS="--model_name_or_path $MODEL_PATH \
--train_data_path $TRAIN_DATA \
--eval_data_path $EVAL_DATA \
--beta $BETA \
--loss_type $LOSS_TYPE \
--label_smoothing $LABEL_SMOOTHING \
--learning_rate $LR \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--gradient_accumulation_steps $GRAD_ACCUM \
--max_length $MAX_LENGTH \
--truncation_mode $TRUNCATION_MODE \
--warmup_ratio $WARMUP_RATIO \
--output_dir $OUTPUT_DIR \
--report_to swanlab \
--project_name TASD-DPO \
--run_name $EXP_NAME \
--bf16 \
--gradient_checkpointing \
--save_steps 100 \
--eval_steps 100 \
--logging_steps 5 \
--save_total_limit 3 \
--upload_data_to_swanlab"

if [ "$USE_LORA" = true ]; then
    ARGS="$ARGS --use_lora --lora_r $LORA_R"
fi

echo "================================================================"
echo "Starting Local DPO Training"
echo "================================================================"
echo "Experiment:      $EXP_NAME"
echo "Model:           $MODEL_PATH"
echo "Train data:      $TRAIN_DATA"
echo "Eval data:       $EVAL_DATA"
echo "Output:          $OUTPUT_DIR"
echo "Log file:        $LOG_FILE"
echo ""
echo "Key parameters:"
echo "  Max length:    $MAX_LENGTH tokens"
echo "  Batch size:    $BATCH_SIZE x $GRAD_ACCUM = $(($BATCH_SIZE * $GRAD_ACCUM)) effective"
echo "  LoRA:          $USE_LORA"
echo "================================================================"

accelerate launch --config_file "$ACCELERATE_CONFIG" \
    "$PROJECT_ROOT/training/dpo_train.py" $ARGS 2>&1 | tee "$LOG_FILE"
