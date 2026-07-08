#!/usr/bin/env python
"""
转换 checkpoint 为 HuggingFace 格式
支持: DCP / FSDP(verl) / Megatron / HuggingFace (自动检测)

修改下面的 CONFIG 配置后直接运行:
    python scripts/convert_checkpoint.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


# ============================================================
#  转换配置 — 在这里修改
# ============================================================
CONFIG = {
    # checkpoint 目录 (拷贝后的本地路径 或 原始路径)
    "checkpoint_dir": "/mnt/nebula/ap-southeast-1/juicefs/kongyixian.kyx/ckpt_KTO-Qwen3-8B-nothink-beta0.2-lr5e-7-bs2-kto",

    # 基础模型路径 (用于获取 config.json 和 tokenizer)
    "base_model_dir": "/data/oss_bucket_0/ad/loujieming.ljm/base_models/Qwen3-8B",

    # 输出目录 (留空则自动: <checkpoint_dir>-hf)
    "output_dir": "/mnt/nebula/ap-southeast-1/juicefs/kongyixian.kyx/KTO-Qwen3-8B-nothink-beta0.2-lr5e-7-bs2-kto-hf",

    # 源格式: "auto" / "dcp" / "fsdp" / "megatron" / "hf"
    "format": "auto",

    # DCP 子目录名 (留空表示 DCP 文件直接在 checkpoint_dir 中)
    "dcp_subdir": "",
}
# ============================================================


def detect_format(checkpoint_dir: Path) -> str:
    """自动检测 checkpoint 格式"""
    d = checkpoint_dir

    # DCP
    if (d / "pytorch_model_fsdp_0" / ".metadata").exists():
        return "dcp"
    if (d / ".metadata").exists() and list(d.glob("*.distcp")):
        return "dcp"

    # FSDP (verl)
    if list(d.glob("model_world_size_*_rank_*.pt")) or (d / "fsdp_config.json").exists():
        return "fsdp"

    # Megatron
    if list(d.glob("mp_rank_*_model_states.pt")):
        return "megatron"

    # HuggingFace
    if (d / "config.json").exists():
        if (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists() or list(d.glob("model-*.safetensors")):
            return "hf"

    if (d / "pytorch_model_fsdp_0").is_dir():
        return "dcp"

    raise ValueError(f"无法识别的 checkpoint 格式: {d}")


def convert_dcp(checkpoint_dir, base_model_dir, output_dir, dcp_subdir):
    """转换 DCP (Distributed Checkpoint) 格式"""
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from transformers import AutoConfig, AutoModelForCausalLM

    dcp_dir = checkpoint_dir / dcp_subdir if dcp_subdir else checkpoint_dir
    if not dcp_dir.exists():
        raise FileNotFoundError(f"DCP 目录不存在: {dcp_dir}")

    # 初始化分布式
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", init_method="env://")

    print("  加载基础模型配置...")
    config = AutoConfig.from_pretrained(base_model_dir)
    print(f"    模型类型: {config.model_type}, hidden: {config.hidden_size}, layers: {config.num_hidden_layers}")

    print("\n  创建模型骨架...")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    print(f"    参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n  包装模型以匹配 DCP key prefix...")

    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model

    wrapper = ModelWrapper(model)
    wrapper_state_dict = dict(wrapper.state_dict())

    print("\n  加载 DCP 分片...")
    reader = FileSystemReader(str(dcp_dir))
    dcp.load_state_dict(wrapper_state_dict, reader)
    print("    加载成功")

    print("\n  提取模型权重并转换为 bf16...")
    model_state_dict = {}
    prefix = "model."
    for k, v in wrapper_state_dict.items():
        if k.startswith(prefix):
            model_state_dict[k[len(prefix):]] = v.to(torch.bfloat16)
    print(f"    提取了 {len(model_state_dict)} 个 key")

    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    if not missing and not unexpected:
        print("    所有 key 完美匹配 ✓")
    else:
        if missing:
            print(f"    缺失 keys: {len(missing)}")
        if unexpected:
            print(f"    多余 keys: {len(unexpected)}")

    print("\n  保存模型...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    if dist.is_initialized():
        dist.destroy_process_group()


def convert_verl(checkpoint_dir, base_model_dir, output_dir, backend):
    """调用 verl.model_merger 转换 FSDP/Megatron 格式"""
    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", backend,
        "--hf_model_path", str(base_model_dir),
        "--local_dir", str(checkpoint_dir),
        "--target_dir", str(output_dir),
    ]
    print(f"  调用 verl.model_merger ({backend})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ 转换失败:")
        print(result.stderr)
        raise RuntimeError(f"verl.model_merger 失败: {result.stderr}")
    print(f"  ✓ {backend} 转换完成")


def convert_hf(checkpoint_dir, output_dir):
    """HuggingFace 格式 - 直接复制"""
    print("  HuggingFace 格式，直接复制...")
    if output_dir == checkpoint_dir:
        print("  输入输出相同，跳过")
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(checkpoint_dir, output_dir)
    print(f"  ✓ 已复制")


def copy_tokenizer(base_model_dir, output_dir):
    """复制 tokenizer 文件"""
    tokenizer_files = [
        "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "chat_template.json", "added_tokens.json",
    ]
    copied = 0
    for fname in tokenizer_files:
        src = base_model_dir / fname
        dst = output_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    if copied > 0:
        print(f"  复制了 {copied} 个 tokenizer 文件")


def main():
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    base_model_dir = Path(CONFIG["base_model_dir"])
    output_dir = Path(CONFIG["output_dir"]) if CONFIG["output_dir"] else Path(f"{checkpoint_dir}-hf")
    fmt = CONFIG["format"]
    dcp_subdir = CONFIG["dcp_subdir"]

    if not checkpoint_dir.exists():
        print(f"❌ checkpoint 目录不存在: {checkpoint_dir}")
        sys.exit(1)
    if not base_model_dir.exists():
        print(f"❌ 基础模型路径不存在: {base_model_dir}")
        sys.exit(1)

    # 检测格式
    if fmt == "auto":
        fmt = detect_format(checkpoint_dir)

    print(f"🔄 转换 checkpoint")
    print(f"   格式: {fmt}")
    print(f"   输入: {checkpoint_dir}")
    print(f"   输出: {output_dir}")
    print("")

    # 执行转换
    if fmt == "dcp":
        convert_dcp(checkpoint_dir, base_model_dir, output_dir, dcp_subdir)
    elif fmt in ("fsdp", "megatron"):
        convert_verl(checkpoint_dir, base_model_dir, output_dir, fmt)
    elif fmt == "hf":
        convert_hf(checkpoint_dir, output_dir)
    else:
        print(f"❌ 不支持的格式: {fmt}")
        sys.exit(1)

    # 复制 tokenizer
    if fmt != "hf":
        print("\n复制 tokenizer 文件...")
        copy_tokenizer(base_model_dir, output_dir)

    # 验证输出
    print("\n✅ 转换完成!")
    print(f"\n输出文件:")
    if output_dir.exists():
        for f in sorted(output_dir.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  {f.name:50s} {size_mb:8.1f} MB")
            else:
                print(f"  {f.name:50s} [目录]")

    print(f"\n📍 输出路径: {output_dir}")
    print(output_dir)


if __name__ == "__main__":
    main()
