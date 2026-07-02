#!/bin/bash
# =============================================================================
# KTO 超参扫描 - Nebula 批量提交
#
# 使用方式：bash nebula_scripts/submit_kto_sweep.sh [--dry-run]
# =============================================================================

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1  # 1 节点 × 4 GPU（FSDP num_processes=4 在 accelerate config 中配置）
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster_gpu_4.json"
SCRIPT_PATH="nebula_scripts/dpo_kto/kto_parametric.sh"
# 自定义镜像（留空则使用 --algo_name=pytorch260 默认镜像）
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:kongyixian.kyx_kyx_h20_1_20260630114431}"
PROJECT_NAME="DPO_KTO"

DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置（Qwen3-8B, 4-GPU FSDP）
# =============================================================================
DATASETS=(
    "/data/oss_bucket_0/ad/kongyixian.kyx/dpo/dataset_d769a815e7a5"
)

MODEL_NAMES=(
    "/data/oss_bucket_0/ad/loujieming.ljm/base_models/Qwen3-8B"
)

# 固定 LR=5e-7（本地 DPO/KTO 单卡验证通过的保守值，8B 模型适配）
LRS=("5e-7")
BETAS=("0.2")    # β sweep: KL 惩罚系数扫描
LOSS_TYPES=("kto")
THINK_MODES=("nothink")  # 支持 think/nothink 数据 sweep

# KTO 权重（可调，当前使用默认值）
DESIRABLE_WEIGHT="1.0"
UNDESIRABLE_WEIGHT="1.0"

# 固定参数（4 卡 FSDP 优化）
BATCH_SIZE="2"           # per_device, KTO 要求 >1（同 batch 内需含 desirable+undesirable）
NUM_EPOCHS="3"
GRAD_ACCUM="4"           # 2×4×4=32 effective，与 DPO 保持一致
MAX_LENGTH="8192"        # prompt 平均 ~3000 tokens，completion ~100 tokens，留足余量
LOGGING_STEPS="3"
SAVE_STEPS="100"
SAVE_TOTAL_LIMIT="5"

# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
for LR in "${LRS[@]}"; do
for BETA in "${BETAS[@]}"; do
for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
for THINK_MODE in "${THINK_MODES[@]}"; do

    TOTAL=$((TOTAL + 1))

    DATASET_SHORT=$(basename "$DATASET")
    MODEL_SHORT=$(basename "$MODEL_NAME")
    LR_TAG=$(echo "$LR" | tr '-' '_' | sed 's/\./p/')
    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="KTO-${MODEL_SHORT}-${THINK_MODE}-beta${BETA}-lr${LR_TAG}-${CURRENT_TIME}"

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  DATASET=$DATASET MODEL=$MODEL_NAME LR=$LR BETA=$BETA THINK_MODE=$THINK_MODE"
        echo "  BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM MAX_LENGTH=$MAX_LENGTH"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
                    --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/dpo_kto_entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL_NAME} --env=LR=${LR} --env=BETA=${BETA} --env=LOSS_TYPE=${LOSS_TYPE} --env=THINK_MODE=${THINK_MODE} --env=BATCH_SIZE=${BATCH_SIZE} --env=NUM_EPOCHS=${NUM_EPOCHS} --env=GRAD_ACCUM=${GRAD_ACCUM} --env=MAX_LENGTH=${MAX_LENGTH} --env=LOGGING_STEPS=${LOGGING_STEPS} --env=SAVE_STEPS=${SAVE_STEPS} --env=SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT} --env=DESIRABLE_WEIGHT=${DESIRABLE_WEIGHT} --env=UNDESIRABLE_WEIGHT=${UNDESIRABLE_WEIGHT}" \
            --worker_count=${WORLD_SIZE} \
            --file.cluster_file=${CLUSTER_FILE} \
            --job_name=${JOB_NAME} \
            --access_id=${access_id} \
            --access_key=${access_key} \
            --env=OPENLM_TOKEN=${OPENLM_TOKEN} \
            --env=SWANLAB_API_KEY=${SWANLAB_API_KEY} \
            $([ -n "$CUSTOM_DOCKER_IMAGE" ] && echo "--custom_docker_image=${CUSTOM_DOCKER_IMAGE}" || echo "--algo_name=pytorch260") \
            --requirements_file_name=requirements_nebula.txt \
            --oss_access_id=${OSS_ACCESS_ID} \
            --oss_access_key=${OSS_ACCESS_KEY} \
            --oss_bucket=${OSS_BUCKET} \
            --oss_endpoint=${OSS_ENDPOINT} 2>&1)
        SUBMIT_EXIT=$?
        echo "$SUBMIT_OUTPUT"
        if [ $SUBMIT_EXIT -ne 0 ]; then
            echo "❌ 提交失败 (exit code: $SUBMIT_EXIT)"
        else
            SUBMITTED=$((SUBMITTED + 1))
            echo "✅ 已提交 (${SUBMITTED}/${TOTAL})"
        fi
        sleep 2
    fi

done  # THINK_MODE
done  # LOSS_TYPE
done  # BETA
done  # LR
done  # MODEL_NAME
done  # DATASET

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
