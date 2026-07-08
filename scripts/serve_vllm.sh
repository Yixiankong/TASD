#!/bin/bash
# ============================================================
#  vLLM Serve 配置
#  启动 vLLM OpenAI-compatible API 服务
# ============================================================

# ----- 在这里修改配置（支持环境变量覆盖）-----
MODEL_PATH="${MODEL_PATH:-/tmp/DPO-Qwen3-8B-nothink-beta0.2-lr5e-7-bs1-ipo-hf}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dpo-qwen3-8b-ipo}"  # 留空则从 MODEL_PATH 自动提取
CONDA_ENV="${CONDA_ENV:-dpo_env}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"    # true=禁用CUDA graphs(启动快), false=启用(推理快)
BACKGROUND="${BACKGROUND:-false}"           # true=后台运行, false=前台运行
# --------------------------

set -e

# 检查模型路径
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "❌ 模型路径缺少 config.json: $MODEL_PATH"
    exit 1
fi

# 提取模型名称
if [[ -z "$SERVED_MODEL_NAME" ]]; then
    SERVED_MODEL_NAME=$(basename "$MODEL_PATH")
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME%-hf}"
    SERVED_MODEL_NAME=$(echo "$SERVED_MODEL_NAME" | sed -E 's/_[0-9]{8}_[0-9]{6}$//')
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME#ckpt_}"
    SERVED_MODEL_NAME=$(echo "$SERVED_MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
fi

echo "🚀 启动 vLLM serve"
echo "   模型路径: $MODEL_PATH"
echo "   模型名称: $SERVED_MODEL_NAME"
echo "   端口: $PORT"
echo "   最大序列长度: $MAX_MODEL_LEN"
echo "   GPU 显存利用率: $GPU_MEMORY_UTIL"
echo "   Enforce eager: $ENFORCE_EAGER"
echo "   后台模式: $BACKGROUND"
echo ""

# 检查端口
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  端口 $PORT 已被占用"
    PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t 2>/dev/null)
    echo "   PID: $PID"
    echo "💡 修改脚本中的 PORT 配置，或 kill $PID"
    exit 1
fi

# 激活 conda 环境
echo "🔧 激活 conda 环境: $CONDA_ENV"
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV" 2>/dev/null || {
    echo "❌ 无法激活 conda 环境: $CONDA_ENV"
    exit 1
}

VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "   vLLM 版本: $VLLM_VERSION"
echo ""

# 将 vLLM 缓存目录挪到 /tmp，避免 home 目录磁盘配额不足
export VLLM_CACHE_ROOT="/tmp/vllm_cache"
mkdir -p "$VLLM_CACHE_ROOT"

# 构建命令
VLLM_ARGS=(
    serve "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size 1
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
    --dtype bfloat16
    --max-model-len "$MAX_MODEL_LEN"
    --served-model-name "$SERVED_MODEL_NAME"
)

if [[ "$ENFORCE_EAGER" == true ]]; then
    VLLM_ARGS+=(--enforce-eager)
fi

if [[ "$BACKGROUND" == true ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="/tmp/vllm_${SERVED_MODEL_NAME}_${TIMESTAMP}.log"

    echo "📝 后台运行"
    echo "   日志: $LOG_FILE"
    echo ""

    vllm "${VLLM_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    VLLM_PID=$!

    echo "✅ vLLM 已启动 (PID: $VLLM_PID)"
    echo ""

    echo "⏳ 等待服务启动..."
    for i in $(seq 1 60); do
        if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
            echo "✅ 服务已就绪!"
            break
        fi
        sleep 2
        if ! ps -p $VLLM_PID > /dev/null 2>&1; then
            echo "❌ vLLM 进程已退出，请查看日志: $LOG_FILE"
            tail -20 "$LOG_FILE"
            exit 1
        fi
    done

    if ! curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
        echo "⚠️  启动超时，请检查日志: $LOG_FILE"
    fi

    echo ""
    echo "📍 服务地址: http://$HOST:$PORT"
    echo ""
    echo "💡 测试:"
    echo "   curl http://localhost:$PORT/v1/chat/completions \\"
    echo "     -H \"Content-Type: application/json\" \\"
    echo "     -d '{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}'"
    echo ""
    echo "📊 查看日志: tail -f $LOG_FILE"
    echo "🛑 停止服务: kill $VLLM_PID"
else
    echo "▶️  前台启动..."
    echo ""
    vllm "${VLLM_ARGS[@]}"
fi
