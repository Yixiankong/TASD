#!/bin/bash
# =============================================================================
# SDPO + EPD (Entropy-Preservation Distillation) 超参扫描 - Nebula 批量提交
#
# 扫描维度：EPD Lambda × EPD Tau
# 使用方式：bash nebula_scripts/submit_sdpo_epd_sweep.sh [--dry-run]
# =============================================================================

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster_gpu_4.json"
SCRIPT_PATH="nebula_scripts/sdpo/sdpo_epd_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="TASD_EPD"

DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置
# =============================================================================
DATASETS=(
    # "sciknoweval/biology"
    # "sciknoweval/chemistry"
    # "sciknoweval/material"
    "sciknoweval/physics"
)

MODEL_NAMES=("Qwen3-8B")

LRS=("1e-5")
ALPHAS=("0.5")
DONT_REPROMPT_LIST=("True")

# ── EPD 核心扫描 ────────────────────────────────────────────────────────
# EPD Lambda: 最大保护强度
# 0.5 = 轻度保护, 0.8 = 平衡（默认）, 1.0 = 全保护
EPD_LAMBDA_LIST=(
    # "0.5"
    "0.8"
    # "1.0"
)

# EPD Tau: Sigmoid 温度
# 0.3 = 尖锐保护, 0.5 = 平滑（默认）, 1.0 = 近似线性
EPD_TAU_LIST=(
    # "0.3"
    "0.5"
    # "1.0"
)

# 固定参数
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"

# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
for LR in "${LRS[@]}"; do
for ALPHA in "${ALPHAS[@]}"; do
for DONT_REPROMPT in "${DONT_REPROMPT_LIST[@]}"; do
for EPD_LAMBDA in "${EPD_LAMBDA_LIST[@]}"; do
for EPD_TAU in "${EPD_TAU_LIST[@]}"; do

    TOTAL=$((TOTAL + 1))

    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    LR_TAG=$(echo "$LR" | tr '-' '_')
    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="SDPO-EPD-${DATASET_SHORT}-lambda${EPD_LAMBDA}-tau${EPD_TAU}-alpha${ALPHA}-lr${LR_TAG}-dross${DONT_REPROMPT}-${MODEL_NAME}-${CURRENT_TIME}"

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  DATASET=$DATASET MODEL=$MODEL_NAME LR=$LR ALPHA=$ALPHA"
        echo "  EPD_LAMBDA=$EPD_LAMBDA EPD_TAU=$EPD_TAU"
        echo "  DONT_REPROMPT=$DONT_REPROMPT"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=ALPHA=${ALPHA} --env=DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT} --env=EPD_LAMBDA=${EPD_LAMBDA} --env=EPD_TAU=${EPD_TAU} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N}" \
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

done  # EPD_TAU
done  # EPD_LAMBDA
done  # DONT_REPROMPT
done  # ALPHA
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
echo ""
echo "EPD 超参扫描说明："
echo "  - EPD Lambda: 最大保护强度 (0.5=轻度, 0.8=平衡, 1.0=全保护)"
echo "  - EPD Tau: Sigmoid 温度 (0.3=尖锐, 0.5=平滑, 1.0=线性)"
echo "  - 总组合数: ${#DATASETS[@]} datasets × ${#EPD_LAMBDA_LIST[@]} lambdas × ${#EPD_TAU_LIST[@]} taus = ${TOTAL} jobs"
echo ""
