#!/bin/bash
set -e

# =============================================================================
# Nebula 数据集准备脚本
#
# 功能：在本地/服务器生成 parquet，然后上传到 OSS
# 使用前提：已安装 ossutil 并配置好鉴权
#
# 使用方式：
#   cd /path/to/SDPO
#   bash nebula_scripts/prepare_data.sh [--sciknoweval-only | --lcb-only]
# =============================================================================

# ── OSS 配置 ─────────────────────────────────────────────────────────────
OSS_ROOT="oss://lazada-ai-model/ad/loujieming.ljm"

# ── HuggingFace 镜像 ──────────────────────────────────────────────────
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN not set}"
export PYTHONPATH=.:${PYTHONPATH:-}

# ── 参数解析 ──────────────────────────────────────────────────────────
DO_SCIKNOW=true
DO_LCB=true
for arg in "$@"; do
    case "$arg" in
        --sciknoweval-only) DO_LCB=false ;;
        --lcb-only)         DO_SCIKNOW=false ;;
    esac
done

echo "============================================================"
echo "SDPO Nebula 数据集准备"
echo "  sciknoweval : $DO_SCIKNOW"
echo "  lcb_v6      : $DO_LCB"
echo "  OSS_ROOT    : $OSS_ROOT"
echo "============================================================"

# =============================================================================
# sciknoweval（biology / chemistry / material / physics）
# =============================================================================
if [ "$DO_SCIKNOW" = true ]; then
    echo ""
    echo "=== sciknoweval 数据集 ==="

    for DOMAIN in Biology Chemistry Material Physics; do
        LOWER=$(echo $DOMAIN | tr '[:upper:]' '[:lower:]')
        DIR="datasets/sciknoweval/${LOWER}"
        PARQUET="${DIR}/train.parquet"

        if [ -f "$PARQUET" ]; then
            echo "[$DOMAIN] parquet 已存在，跳过生成，直接上传"
        else
            echo "[$DOMAIN] 生成 parquet..."
            mkdir -p "$DIR"

            python data/load_dataset.py \
                --dataset_name "$DOMAIN" \
                --output_path "${DIR}/${LOWER}.json"

            python data/split_tasks.py \
                --json_path "${DIR}/${LOWER}.json" \
                --output_dir "$DIR" \
                --test_ratio 0.1 \
                --seed 42

            python data/preprocess.py \
                --data_source "$DIR"

            echo "[$DOMAIN] ✅ parquet 生成完成"
        fi

        echo "[$DOMAIN] 上传到 OSS..."
        ossutil cp -r "${DIR}/" "${OSS_ROOT}/datasets/sciknoweval/${LOWER}/" --update
        echo "[$DOMAIN] ✅ 上传完成"
    done
fi

# =============================================================================
# livecodebench v6（rich feedback 实验用）
# =============================================================================
if [ "$DO_LCB" = true ]; then
    echo ""
    echo "=== livecodebench v6 数据集 ==="

    if [ -f "datasets/lcb_v6/train.parquet" ]; then
        echo "[lcb_v6] parquet 已存在，跳过生成，直接上传"
    else
        echo "[lcb_v6] 生成 parquet..."

        python data/load_dataset.py \
            --dataset_name livecodebench/code_generation_lite-v6 \
            --output_path datasets/lcb_v6.json

        python data/split_tests.py \
            --json_path datasets/lcb_v6.json \
            --output_dir datasets/lcb_v6

        python data/preprocess.py \
            --data_source datasets/lcb_v6

        echo "[lcb_v6] ✅ parquet 生成完成"
    fi

    echo "[lcb_v6] 上传到 OSS..."
    ossutil cp -r datasets/lcb_v6/ "${OSS_ROOT}/datasets/lcb_v6/" --update
    echo "[lcb_v6] ✅ 上传完成"
fi

echo ""
echo "============================================================"
echo "✅ 数据集准备完成！"
echo ""
echo "OSS 路径（Nebula 训练脚本中使用）："
echo "  sciknoweval biology : /data/oss_bucket_0/ad/loujieming.ljm/datasets/sciknoweval/biology"
echo "  sciknoweval chemistry: /data/oss_bucket_0/ad/loujieming.ljm/datasets/sciknoweval/chemistry"
echo "  lcb_v6              : /data/oss_bucket_0/ad/loujieming.ljm/datasets/lcb_v6"
echo "============================================================"
