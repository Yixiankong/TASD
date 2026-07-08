#!/bin/bash
# ============================================================
#  拷贝 checkpoint 配置
#  从 OSS 拷贝到本地存储，解决 OSS I/O 慢的问题
# ============================================================

# ----- 在这里修改配置 -----
SOURCE="/data/oss_bucket_0/ad/kongyixian.kyx/TASD/checkpoints/DPO-Qwen3-8B-nothink-beta0.2-lr5e-7-bs1-sigmoid/checkpoint-500"
TARGET=""                    # 留空则自动生成: /tmp/ckpt_<实验名>_<时间戳>
SUBDIR="pytorch_model_fsdp_0"  # 要拷贝的子目录，"." 表示整个目录
# --------------------------

set -e

if [[ ! -d "$SOURCE" ]]; then
    echo "❌ 源路径不存在: $SOURCE"
    exit 1
fi

# 提取实验名
extract_experiment_name() {
    local path="$1"
    local parent=$(dirname "$path")
    local exp_name=$(basename "$parent")
    if [[ "$exp_name" =~ ^checkpoint- ]] || [[ "$exp_name" =~ ^global_step_ ]] || [[ "$exp_name" =~ ^actor$ ]]; then
        parent=$(dirname "$parent")
        exp_name=$(basename "$parent")
    fi
    echo "$exp_name"
}

EXPERIMENT_NAME=$(extract_experiment_name "$SOURCE")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ -z "$TARGET" ]]; then
    TARGET="/tmp/ckpt_${EXPERIMENT_NAME}_${TIMESTAMP}"
fi

if [[ "$SUBDIR" == "." ]]; then
    SRC_PATH="$SOURCE/"
else
    if [[ ! -d "$SOURCE/$SUBDIR" ]]; then
        echo "❌ 子目录不存在: $SOURCE/$SUBDIR"
        echo "💡 提示: 将 SUBDIR 改为 \".\" 拷贝整个目录"
        exit 1
    fi
    SRC_PATH="$SOURCE/$SUBDIR/"
fi

echo "📦 拷贝 checkpoint"
echo "   实验名: $EXPERIMENT_NAME"
echo "   源路径: $SRC_PATH"
echo "   目标路径: $TARGET"
echo ""

mkdir -p "$TARGET"
cp -rv "$SRC_PATH"* "$TARGET/"

# 验证
echo ""
echo "✅ 拷贝完成"
ls -lh "$TARGET" | head -20

if [[ "$SUBDIR" == "pytorch_model_fsdp_0" ]] || [[ "$SUBDIR" == "." ]]; then
    DISTCP_COUNT=$(find "$TARGET" -name "*.distcp" 2>/dev/null | wc -l)
    if [[ $DISTCP_COUNT -gt 0 ]]; then
        METADATA_EXISTS=$([[ -f "$TARGET/.metadata" ]] && echo "✓" || echo "✗")
        echo ""
        echo "   .distcp 文件: $DISTCP_COUNT"
        echo "   .metadata: $METADATA_EXISTS"
    fi
fi

echo ""
echo "📍 本地路径: $TARGET"
echo "$TARGET"
