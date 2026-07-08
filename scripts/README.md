# 模型部署脚本工具集

配置化的部署脚本，直接修改脚本顶部的配置参数即可运行，无需传递命令行参数。

## 脚本概览

### 1. `copy_checkpoint.sh` - 拷贝 checkpoint
**配置项**:
```bash
SOURCE="/data/oss_bucket_0/.../checkpoint-500"  # 源路径
TARGET=""                                         # 目标路径（留空自动生成）
SUBDIR="pytorch_model_fsdp_0"                     # 子目录名
```

**用法**:
```bash
bash scripts/copy_checkpoint.sh
```

---

### 2. `convert_checkpoint.py` - 转换 checkpoint
**配置项** (CONFIG 字典):
```python
CONFIG = {
    "checkpoint_dir": "/tmp/dpo_ckpt",           # checkpoint 目录
    "base_model_dir": "/data/.../Qwen3-8B",      # 基础模型路径
    "output_dir": "",                            # 输出目录（留空自动生成）
    "format": "auto",                            # 格式：auto/dcp/fsdp/megatron/hf
    "dcp_subdir": "",                            # DCP 子目录名
}
```

**用法**:
```bash
python scripts/convert_checkpoint.py
```

**支持的格式**: DCP、FSDP (verl)、Megatron、HuggingFace（自动检测）

---

### 3. `serve_vllm.sh` - 启动 vLLM serve
**配置项**:
```bash
MODEL_PATH="/tmp/dpo_ckpt_hf"    # HF 模型路径
PORT=8000                         # 端口
HOST="0.0.0.0"                    # 地址
MAX_MODEL_LEN=4096                # 最大序列长度
GPU_MEMORY_UTIL=0.90              # GPU 显存利用率
SERVED_MODEL_NAME=""              # 模型名（留空自动提取）
CONDA_ENV="dpo_env"               # conda 环境
ENFORCE_EAGER=true                # 禁用 CUDA graphs
BACKGROUND=false                  # 后台运行
```

**用法**:
```bash
bash scripts/serve_vllm.sh
```

---

### 4. `deploy_model.sh` - 一站式部署
**配置项**:
```bash
CHECKPOINT_DIR="/data/.../checkpoint-500"  # checkpoint 路径
BASE_MODEL_DIR="/data/.../Qwen3-8B"        # 基础模型路径
PORT=8000                                   # 端口
MAX_MODEL_LEN=4096                          # 最大序列长度
GPU_MEMORY_UTIL=0.90                        # GPU 显存利用率
SERVED_MODEL_NAME=""                        # 模型名（留空自动提取）
CONDA_ENV="dpo_env"                         # conda 环境
BACKGROUND=false                            # 后台运行
SKIP_COPY=false                             # 跳过拷贝
SKIP_CONVERT=false                          # 跳过转换
```

**用法**:
```bash
bash scripts/deploy_model.sh
```

---

## 快速开始

### 场景 1: 完整部署（一键）
```bash
# 1. 编辑 scripts/deploy_model.sh，修改顶部的 CHECKPOINT_DIR 和 BASE_MODEL_DIR
vim scripts/deploy_model.sh

# 2. 直接运行
bash scripts/deploy_model.sh
```

### 场景 2: 分步执行（更灵活）
```bash
# 步骤 1: 拷贝
vim scripts/copy_checkpoint.sh  # 修改 SOURCE
bash scripts/copy_checkpoint.sh
# 输出: /tmp/ckpt_<实验名>_<时间戳>

# 步骤 2: 转换
vim scripts/convert_checkpoint.py  # 修改 CONFIG["checkpoint_dir"]
python scripts/convert_checkpoint.py
# 输出: <checkpoint_dir>-hf

# 步骤 3: 启动服务
vim scripts/serve_vllm.sh  # 修改 MODEL_PATH
bash scripts/serve_vllm.sh
```

### 场景 3: 直接启动已有 HF 模型
```bash
# 编辑 scripts/serve_vllm.sh，修改 MODEL_PATH 为已转换的模型路径
vim scripts/serve_vllm.sh
bash scripts/serve_vllm.sh
```

---

## 测试 API

服务启动后，使用 curl 测试：

```bash
# 基础请求（禁用 thinking）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dpo-qwen3-8b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": false}
  }'

# 启用 thinking 模式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dpo-qwen3-8b",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "max_tokens": 512
  }'
```

---

## 实验命名规范

脚本自动从 checkpoint 路径提取实验名：

```
/data/oss_bucket_0/.../DPO-Qwen3-8B-nothink-beta0.2-lr5e-7-bs1-sigmoid/checkpoint-500
                              ↓
              DPO-Qwen3-8B-nothink-beta0.2-lr5e-7-bs1-sigmoid
```

用于：
- 本地临时目录: `/tmp/ckpt_<实验名>_<时间戳>`
- 默认 served_model_name: `<实验名>` (小写，短横线分隔)
- 日志文件: `/tmp/vllm_<实验名>_<时间戳>.log`

---

## 故障排除

### 1. OSS 拷贝很慢
**解决方案**: 使用 `copy_checkpoint.sh` 拷贝到本地 `/tmp`，然后再转换。

### 2. 转换脚本报错 "无法识别的 checkpoint 格式"
**解决方案**: 
- 检查 checkpoint 目录结构
- 在 `convert_checkpoint.py` 中设置 `CONFIG["format"]` 为具体格式（dcp/fsdp/megatron/hf）

### 3. vLLM 启动失败 "端口已被占用"
**解决方案**:
```bash
# 查看占用进程
lsof -i :8000

# 修改 serve_vllm.sh 中的 PORT 配置
vim scripts/serve_vllm.sh
```

### 4. vLLM 启动失败 "CUDA out of memory"
**解决方案**:
```bash
# 编辑 serve_vllm.sh，降低显存利用率
GPU_MEMORY_UTIL=0.80

# 或降低最大序列长度
MAX_MODEL_LEN=2048
```

---

## 脚本文件

```
scripts/
├── copy_checkpoint.sh      # 拷贝 checkpoint（配置化）
├── convert_checkpoint.py   # 转换 checkpoint 格式（配置化）
├── serve_vllm.sh           # 启动 vLLM serve（配置化）
├── deploy_model.sh         # 一站式部署（配置化）
└── convert_dpo_dcp_to_hf.py # 旧的 DCP 专用脚本（保留）
```

---

## 依赖和环境

### Conda 环境
- 默认使用 `dpo_env` (vLLM 0.17.1, transformers 5.12.1)
- 在各脚本的 `CONDA_ENV` 配置项中修改

### Python 依赖
- `torch` >= 2.10 (DCP API)
- `transformers` >= 5.0 (Qwen3 支持)
- `vllm` >= 0.17 (serving)

### 系统依赖
- `cp` (GNU coreutils，用于 checkpoint 拷贝)
- `lsof` (用于端口检查)

---

## 更多信息

- 参考 `CLAUDE.md` 了解项目架构
- 查看 vLLM 文档: https://docs.vllm.ai/
- 各脚本顶部有详细的配置说明
