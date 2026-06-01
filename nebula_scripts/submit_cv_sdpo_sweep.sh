#!/bin/bash
# =============================================================================
# CV-SDPO 超参扫描 - Nebula 批量提交
#
# 参考 nebula_scripts/submit_sdpo_baseline_sweep.sh
# 使用方式：bash nebula_scripts/submit_cv_sdpo_sweep.sh [--dry-run]
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
SCRIPT_PATH="nebula_scripts/sdpo/sdpo_sciknoweval_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="CV_SDPO_v1"

DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run mode: only print commands, do not submit"
fi

# =============================================================================
# 超参配置
# =============================================================================
DATASETS=(
    "sciknoweval/biology"
    "sciknoweval/chemistry"
    "sciknoweval/material"
    "sciknoweval/physics"
    "tooluse"
)

MODEL_NAMES=("Qwen3-8B")
LRS=("1e-5")
ALPHAS=("0.5")
DONT_REPROMPT_LIST=("True")
CV_GAMMAS=("0.5")
BETA_MODES=("adaptive")

# Fixed parameters
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
CONFIG_NAME="cv_sdpo"

# OOM headroom for CV-SDPO (3× teacher forward pre-computation)
PPO_MINI_BATCH_SIZE="16"
GPU_MEMORY_UTIL="0.75"

# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
for LR in "${LRS[@]}"; do
for ALPHA in "${ALPHAS[@]}"; do
for DONT_REPROMPT in "${DONT_REPROMPT_LIST[@]}"; do
for CV_GAMMA in "${CV_GAMMAS[@]}"; do
for BETA_MODE in "${BETA_MODES[@]}"; do

    TOTAL=$((TOTAL + 1))

    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    LR_TAG=$(echo "$LR" | tr '-' '_')
    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="CV_SDPO-${DATASET_SHORT}-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-gamma${CV_GAMMA}-beta${BETA_MODE}-lr${LR_TAG}-dross${DONT_REPROMPT}-${MODEL_NAME}-${CURRENT_TIME}"

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  DATASET=$DATASET MODEL=$MODEL_NAME LR=$LR ALPHA=$ALPHA GAMMA=$CV_GAMMA BETA=$BETA_MODE"
    else
        echo "Submitting Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
                    --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=ALPHA=${ALPHA} --env=DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=CONFIG_NAME=${CONFIG_NAME} --env=CV_GAMMA=${CV_GAMMA} --env=BETA_MODE=${BETA_MODE} --env=PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} --env=GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL}" \
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
            echo "Submission failed (exit code: $SUBMIT_EXIT)"
        else
            SUBMITTED=$((SUBMITTED + 1))
            echo "Submitted (${SUBMITTED}/${TOTAL})"
        fi
        sleep 2
    fi

done  # BETA_MODE
done  # CV_GAMMA
done  # DONT_REPROMPT
done  # ALPHA
done  # LR
done  # MODEL_NAME
done  # DATASET

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run complete, ${TOTAL} jobs total"
else
    echo "Submission complete: ${SUBMITTED} / ${TOTAL} jobs"
fi
echo "============================================================"
