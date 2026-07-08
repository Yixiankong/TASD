#!/bin/bash
# ============================================================
#  一站式部署配置
#  自动串联: 拷贝 → 转换 → 启动 vLLM serve
# ============================================================

# ----- 在这里修改配置 -----

# 源 checkpoint 路径
CHECKPOINT_DIR="/data/oss_bucket_0/ad/kongyixian.kyx/TASD/checkpoints/DPO-Qwen3-8B-nothink-beta0.2-lr5e-7-bs1-sigmoid/checkpoint-500"

# 基础模型路径
BASE_MODEL_DIR="/data/oss_bucket_0/ad/loujieming.ljm/base_models/Qwen3-8B"

# vLLM 配置
PORT=8000
HOST="0.0.0.0"
MAX_MODEL_LEN=4096
GPU_MEMORY_UTIL=0.90
SERVED_MODEL_NAME=""         # 留空则自动提取
CONDA_ENV="dpo_env"
BACKGROUND=false

# 跳过步骤 (已在本地/已转换时设为 true)
SKIP_COPY=false
SKIP_CONVERT=false

# --------------------------

set -e

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

EXPERIMENT_NAME=$(extract_experiment_name "$CHECKPOINT_DIR")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🚀 一站式部署: $EXPERIMENT_NAME"
echo "   Checkpoint: $CHECKPOINT_DIR"
echo "   基础模型: $BASE_MODEL_DIR"
echo ""

# ============================================================
# 步骤 1: 拷贝
# ============================================================
if [[ "$SKIP_COPY" == true ]]; then
    echo "⏭️  跳过拷贝"
    LOCAL_DIR="$CHECKPOINT_DIR"
else
    echo "📦 步骤 1/3: 拷贝 checkpoint"

    TARGET="/tmp/ckpt_${EXPERIMENT_NAME}_${TIMESTAMP}"
    SRC_PATH="$CHECKPOINT_DIR/pytorch_model_fsdp_0/"

    # 如果没有 pytorch_model_fsdp_0，拷贝整个目录
    if [[ ! -d "$CHECKPOINT_DIR/pytorch_model_fsdp_0" ]]; then
        SRC_PATH="$CHECKPOINT_DIR/"
    fi

    mkdir -p "$TARGET"
    cp -rv "$SRC_PATH"* "$TARGET/"
    LOCAL_DIR="$TARGET"
    echo "   本地路径: $LOCAL_DIR"
fi
echo ""

# ============================================================
# 步骤 2: 转换
# ============================================================
if [[ "$SKIP_CONVERT" == true ]]; then
    echo "⏭️  跳过转换"
    HF_DIR="$LOCAL_DIR"
else
    echo "🔄 步骤 2/3: 转换为 HuggingFace 格式"

    # 自动检测是否需要 dcp_subdir
    DCP_SUBDIR=""
    if [[ -f "$LOCAL_DIR/.metadata" ]]; then
        DCP_SUBDIR=""
    elif [[ -d "$LOCAL_DIR/pytorch_model_fsdp_0" ]]; then
        # 拷贝的是整个 checkpoint 目录，DCP 在子目录中
        DCP_SUBDIR=""
    fi

    HF_DIR="$LOCAL_DIR-hf"

    source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV" 2>/dev/null

    python "$(dirname "$0")/convert_checkpoint.py" \
        --checkpoint_dir "$LOCAL_DIR" \
        --base_model_dir "$BASE_MODEL_DIR" \
        --output_dir "$HF_DIR" \
        --dcp_subdir "$DCP_SUBDIR" 2>/dev/null || {
        # 如果脚本不支持命令行参数，直接用 Python 执行
        python -c "
import sys, os
sys.path.insert(0, '$(dirname "$0")')
from convert_checkpoint import main
import convert_checkpoint as cc
cc.CONFIG['checkpoint_dir'] = '$LOCAL_DIR'
cc.CONFIG['base_model_dir'] = '$BASE_MODEL_DIR'
cc.CONFIG['output_dir'] = '$HF_DIR'
cc.CONFIG['dcp_subdir'] = '$DCP_SUBDIR'
cc.CONFIG['format'] = 'auto'
main()
"
    }

    echo "   HF 路径: $HF_DIR"
fi
echo ""

# ============================================================
# 步骤 3: 启动 vLLM
# ============================================================
echo "🚀 步骤 3/3: 启动 vLLM serve"

# 检查端口
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  端口 $PORT 已被占用"
    PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t 2>/dev/null)
    echo "   PID: $PID"
    exit 1
fi

source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV" 2>/dev/null

# 提取模型名称
if [[ -z "$SERVED_MODEL_NAME" ]]; then
    SERVED_MODEL_NAME=$(basename "$HF_DIR")
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME%-hf}"
    SERVED_MODEL_NAME=$(echo "$SERVED_MODEL_NAME" | sed -E 's/_[0-9]{8}_[0-9]{6}$//')
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME#ckpt_}"
    SERVED_MODEL_NAME=$(echo "$SERVED_MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
fi

VLLM_ARGS=(
    serve "$HF_DIR"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size 1
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
    --dtype bfloat16
    --max-model-len "$MAX_MODEL_LEN"
    --served-model-name "$SERVED_MODEL_NAME"
    --enforce-eager
)

if [[ "$BACKGROUND" == true ]]; then
    TIMESTAMP2=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="/tmp/vllm_${SERVED_MODEL_NAME}_${TIMESTAMP2}.log"

    echo "📝 后台运行，日志: $LOG_FILE"
    vllm "${VLLM_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    VLLM_PID=$!

    echo "✅ vLLM 已启动 (PID: $VLLM_PID)"
    echo "⏳ 等待服务启动..."
    for i in $(seq 1 60); do
        if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
            echo "✅ 服务已就绪!"
            break
        fi
        sleep 2
        if ! ps -p $VLLM_PID > /dev/null 2>&1; then
            echo "❌ vLLM 已退出，日志:"
            tail -20 "$LOG_FILE"
            exit 1
        fi
    done
else
    echo "▶️  前台启动..."
    vllm "${VLLM_ARGS[@]}"
fi

echo ""
echo "✅ 部署完成!"
echo ""
echo "📍 服务地址: http://$HOST:$PORT"
echo "🤖 模型名称: $SERVED_MODEL_NAME"
echo ""
echo "💡 测试:"
echo "   curl http://localhost:$PORT/v1/chat/completions \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}'"
