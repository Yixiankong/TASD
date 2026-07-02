#!/usr/bin/env python3
"""KTO (Kahneman-Tversky Optimization) training script.

This script implements KTO training using the TRL library.
KTO learns from binary feedback (good/bad) rather than preference pairs.

Data format:
    - prompt: str (chat template applied string)
    - completion: str (response text)
    - label: bool (True = desirable, False = undesirable)
"""
import os
import json
import logging
import argparse
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="KTO Training")

    # Model and data
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, default=None)

    # KTO specific
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--desirable_weight", type=float, default=1.0, help="Weight for desirable examples")
    parser.add_argument("--undesirable_weight", type=float, default=1.0, help="Weight for undesirable examples")
    parser.add_argument("--loss_type", type=str, default="kto", choices=["kto", "kto_pair"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--precompute_ref_log_probs", action="store_true",
                        help="Pre-compute ref model log probs before training (~50%% speedup)")

    # Training
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=10240, help="Max sequence length (prompt + completion)")
    parser.add_argument("--truncation_mode", type=str, default="keep_start", choices=["keep_start", "keep_end"])
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)

    # SwanLab
    parser.add_argument("--use_swanlab", action="store_true", default=True)
    parser.add_argument("--swanlab_project", type=str, default="TASD-KTO")
    parser.add_argument("--swanlab_run_name", type=str, default=None)
    parser.add_argument("--upload_data_to_swanlab", action="store_true", default=False)

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="all-linear")

    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume from")

    return parser.parse_args()


def upload_data_to_swanlab(train_dataset, eval_dataset=None):
    """Upload training data samples to SwanLab as artifacts."""
    try:
        import swanlab

        # Upload sample data as artifact
        samples = []
        for i in range(min(10, len(train_dataset))):
            sample = train_dataset[i]
            samples.append({
                "prompt": sample["prompt"][:500],
                "completion": sample["completion"][:500],
                "label": sample["label"]
            })

        with open("kto_sample_data.json", "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        swanlab.log({
            "sample_data": swanlab.Artifact("kto_sample_data.json", name="kto_samples"),
            "dataset/train_size": len(train_dataset),
            "dataset/train_desirable": sum(1 for x in train_dataset if x["label"]),
            "dataset/train_undesirable": sum(1 for x in train_dataset if not x["label"]),
        })

        if eval_dataset:
            swanlab.log({
                "dataset/eval_size": len(eval_dataset),
                "dataset/eval_desirable": sum(1 for x in eval_dataset if x["label"]),
                "dataset/eval_undesirable": sum(1 for x in eval_dataset if not x["label"]),
            })

        logger.info("✓ Uploaded sample data to SwanLab")
    except Exception as e:
        logger.warning(f"Failed to upload data to SwanLab: {e}")


def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("KTO Training")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Train data: {args.train_data_path}")
    logger.info(f"Eval data: {args.eval_data_path or 'None'}")
    logger.info(f"Output: {args.output_dir}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer: {type(tokenizer).__name__}")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_dataset("parquet", data_files=args.train_data_path, split="train")
    eval_dataset = None
    if args.eval_data_path:
        eval_dataset = load_dataset("parquet", data_files=args.eval_data_path, split="train")

    logger.info(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval samples: {len(eval_dataset)}")

    # Verify data format
    logger.info("Verifying data format...")
    sample = train_dataset[0]
    assert "prompt" in sample, "Missing 'prompt' field"
    assert "completion" in sample, "Missing 'completion' field"
    assert "label" in sample, "Missing 'label' field"
    assert isinstance(sample["prompt"], str), f"prompt must be str, got {type(sample['prompt'])}"
    assert isinstance(sample["completion"], str), f"completion must be str, got {type(sample['completion'])}"
    assert isinstance(sample["label"], bool), f"label must be bool, got {type(sample['label'])}"
    logger.info("✓ Data format verified")

    # Determine model for trainer
    # Non-LoRA: pass model path string, let trainer handle loading with model_init_kwargs
    # LoRA: pre-load base model and apply PEFT, trainer creates ref_model from base
    if args.use_lora:
        logger.info("Loading model for LoRA...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            trust_remote_code=True,
        )
        from peft import get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(",") if args.lora_target_modules != "all-linear" else "all-linear",
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        logger.info(f"LoRA applied: r={args.lora_r}, alpha={args.lora_alpha}")
        model_for_trainer = model
    else:
        logger.info(f"Model will be loaded by trainer from: {args.model_name_or_path}")
        model_for_trainer = args.model_name_or_path
        peft_config = None

    # KTO config
    # model_init_kwargs 控制 main model 和 ref model 的加载
    # device_map={"": "cpu"} 让模型先加载到 CPU，FSDP 接管后分片到各 GPU
    # 避免 device_map="auto" 导致的 FSDP 设备冲突
    kto_config = KTOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        precompute_ref_log_probs=args.precompute_ref_log_probs,
        logging_steps=args.logging_steps,
        logging_first_step=True,  # 记录第一步，避免长时间无指标
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        beta=args.beta,
        desirable_weight=args.desirable_weight,
        undesirable_weight=args.undesirable_weight,
        max_length=args.max_length,
        remove_unused_columns=False,
        report_to=["swanlab"] if args.use_swanlab else [],
        run_name=args.swanlab_run_name or f"kto_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        model_init_kwargs={
            "torch_dtype": torch.bfloat16 if args.bf16 else torch.float32,
            "trust_remote_code": True,
            "device_map": {"": "cpu"},  # FSDP: 加载到 CPU，让 FSDP 分片
        },
    )

    # 手动初始化 SwanLab，确保 run 在 trainer 创建前已就绪
    # 只在 rank 0 初始化，避免 FSDP 多进程重复创建 run
    if args.use_swanlab:
        try:
            from accelerate.state import PartialState
            is_main = PartialState().is_main_process
        except Exception:
            is_main = True
        if is_main:
            import swanlab
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_run_name or f"kto_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            logger.info(f"SwanLab initialized: project={args.swanlab_project}")

    # Create trainer
    logger.info("Creating KTO trainer...")
    trainer = KTOTrainer(
        model=model_for_trainer,
        ref_model=None,  # Will be created internally
        args=kto_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    logger.info("KTOTrainer created successfully")

    # Train
    logger.info("Starting KTO training...")
    resume = args.resume_from_checkpoint
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
    trainer.train(resume_from_checkpoint=resume)

    # Upload data to SwanLab after training (SwanLab is initialized by then)
    if args.use_swanlab and args.upload_data_to_swanlab:
        logger.info("Uploading training data to SwanLab...")
        upload_data_to_swanlab(train_dataset, eval_dataset)

    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("✓ KTO training completed!")


if __name__ == "__main__":
    main()
