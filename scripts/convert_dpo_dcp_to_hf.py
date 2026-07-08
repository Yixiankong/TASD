#!/usr/bin/env python
"""
Convert DPO/RLHF FSDP DCP (Distributed Checkpoint) shards to standard HuggingFace format.

The DPO trainer (TRL) saves checkpoints as PyTorch DCP shards when using FSDP with
SHARDED_STATE_DICT. This script reconstructs the full model and saves it in HuggingFace
format (config.json + safetensors + tokenizer) for use with vLLM or other inference engines.

Usage:
    python scripts/convert_dpo_dcp_to_hf.py \
        --checkpoint_dir /path/to/checkpoint-500 \
        --base_model_dir /path/to/Qwen3-8B \
        --output_dir /path/to/output-hf

Requirements:
    - torch (2.10+ for DCP API)
    - transformers (5.x for Qwen3 support)
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DPO DCP checkpoint to HuggingFace format")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the DCP checkpoint directory (e.g., checkpoint-500)",
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        required=True,
        help="Path to the base HuggingFace model (for config and tokenizer)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the converted HuggingFace model",
    )
    parser.add_argument(
        "--dcp_subdir",
        type=str,
        default="pytorch_model_fsdp_0",
        help="Subdirectory name containing the DCP shards (default: pytorch_model_fsdp_0)",
    )
    return parser.parse_args()


def init_distributed():
    """Initialize torch.distributed for DCP (required even for single-process)."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", init_method="env://")


def load_dcp_state_dict(dcp_dir: Path, wrapper_state_dict: dict):
    """Load DCP shards into the provided state dict (in-place)."""
    reader = FileSystemReader(str(dcp_dir))
    dcp.load_state_dict(wrapper_state_dict, reader)


def main():
    args = parse_args()

    # Paths
    checkpoint_dir = Path(args.checkpoint_dir)
    dcp_dir = checkpoint_dir / args.dcp_subdir
    base_model_dir = Path(args.base_model_dir)
    output_dir = Path(args.output_dir)

    if not dcp_dir.exists():
        raise FileNotFoundError(f"DCP directory not found: {dcp_dir}")
    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")

    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"DCP dir: {dcp_dir}")
    print(f"Base model: {base_model_dir}")
    print(f"Output: {output_dir}")

    # Initialize distributed (required for DCP)
    init_distributed()

    # Load base model config and create skeleton
    print("\n[1/6] Loading base model config...")
    config = AutoConfig.from_pretrained(base_model_dir)
    print(f"  Model type: {config.model_type}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")

    # Create empty model skeleton (no weights loaded yet)
    print("\n[2/6] Creating model skeleton...")
    # Use float32 to match DCP storage dtype (DCP stores in fp32 even if training was bf16)
    with torch.device("cpu"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Wrap model to match DCP key prefix
    # DPO trainer wraps the model with DDP, adding "model." prefix to all keys
    print("\n[3/6] Wrapping model to match DCP key prefix...")

    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model

    wrapper = ModelWrapper(model)
    wrapper_state_dict = dict(wrapper.state_dict())
    print(f"  Wrapper state_dict keys: {len(wrapper_state_dict)}")
    print(f"  Sample keys: {list(wrapper_state_dict.keys())[:3]}")

    # Load DCP shards
    print("\n[4/6] Loading DCP shards...")
    load_dcp_state_dict(dcp_dir, wrapper_state_dict)
    print("  Loaded successfully")

    # Extract base model state_dict by stripping "model." prefix
    print("\n[5/6] Extracting model weights and converting to bf16...")
    model_state_dict = {}
    prefix = "model."
    for k, v in wrapper_state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix) :]
            # Cast to bf16 (matches training dtype)
            model_state_dict[new_key] = v.to(torch.bfloat16)
        else:
            print(f"  Warning: unexpected key without prefix: {k}")

    print(f"  Extracted {len(model_state_dict)} keys")

    # Load into model
    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if not missing and not unexpected:
        print("  All keys matched perfectly")

    # Save model in HuggingFace format
    print("\n[6/6] Saving model...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config + weights (safetensors)
    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"  Saved model to {output_dir}")

    # Copy tokenizer files
    print("\nCopying tokenizer files...")
    tokenizer_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template.json",
        "added_tokens.json",
    ]
    copied = 0
    for fname in tokenizer_files:
        src = base_model_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            copied += 1
    print(f"  Copied {copied} tokenizer files")

    # Verify
    print("\n✓ Conversion complete!")
    print(f"\nOutput files:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:50s} {size_mb:8.1f} MB")

    # Cleanup distributed
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
