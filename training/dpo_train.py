#!/usr/bin/env python3
"""DPO (Direct Preference Optimization) training script using TRL.

Standalone script — no verl/Ray dependency.
Uses HuggingFace Trainer + TRL DPOTrainer with Accelerate for distributed training.
"""
import argparse
import json
import logging
import os
import sys

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Training")

    # Required
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model path or HF hub ID")
    parser.add_argument("--train_data_path", type=str, required=True, help="Training data path")

    # Data
    parser.add_argument("--eval_data_path", type=str, default=None, help="Eval data path")
    parser.add_argument("--max_length", type=int, default=10240, help="Max total length (prompt + completion)")
    parser.add_argument("--truncation_mode", type=str, default="keep_start", choices=["keep_start", "keep_end"])
    parser.add_argument("--dataset_format", type=str, default="auto",
                        choices=["auto", "parquet", "json", "jsonl", "hf_dataset"])

    # DPO specific
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--loss_type", type=str, default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo", "kto_pair"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--reference_free", action="store_true",
                        help="Use reference-free DPO (no ref model, saves ~50%% memory)")
    parser.add_argument("--precompute_ref_log_probs", action="store_true",
                        help="Pre-compute ref model log probs before training (~50%% speedup)")

    # Training
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")

    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="all-linear")

    # Logging
    parser.add_argument("--report_to", type=str, default="swanlab")
    parser.add_argument("--project_name", type=str, default="TASD-DPO")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)

    # Data upload
    parser.add_argument("--upload_data_to_swanlab", action="store_true",
                        help="Upload training data to SwanLab as table")

    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


def load_dataset_auto(path: str, fmt: str) -> datasets.Dataset:
    """Load dataset with auto format detection."""
    if fmt == "auto":
        if path.endswith(".parquet"):
            fmt = "parquet"
        elif path.endswith(".jsonl"):
            fmt = "jsonl"
        elif path.endswith(".json"):
            fmt = "json"
        else:
            fmt = "parquet"

    if fmt == "parquet":
        return datasets.load_dataset("parquet", data_files=path, split="train")
    elif fmt == "jsonl":
        return datasets.load_dataset("json", data_files=path, split="train")
    elif fmt == "json":
        return datasets.load_dataset("json", data_files=path, split="train")
    elif fmt == "hf_dataset":
        return datasets.load_dataset(path, split="train")
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def upload_data_to_swanlab(train_ds, eval_ds, args):
    """Upload training data samples to SwanLab as a table."""
    try:
        import swanlab

        n_samples = min(50, len(train_ds))
        rows = []
        for i in range(n_samples):
            sample = train_ds[i]
            prompt = sample["prompt"]
            if isinstance(prompt, list):
                prompt_str = json.dumps(prompt, ensure_ascii=False)
            else:
                prompt_str = str(prompt)
            rows.append({
                "index": i,
                "prompt": prompt_str[:500],
                "chosen": str(sample["chosen"])[:300],
                "rejected": str(sample["rejected"])[:300],
            })
        swanlab.log({"train_data_sample": swanlab.Table(data=rows, columns=["index", "prompt", "chosen", "rejected"])})

        # Log dataset stats
        swanlab.log({
            "dataset/train_size": len(train_ds),
            "dataset/eval_size": len(eval_ds) if eval_ds else 0,
            "dataset/max_length": args.max_length,
            "dataset/truncation_mode": args.truncation_mode,
        })
        logger.info(f"Uploaded {n_samples} data samples to SwanLab")
    except Exception as e:
        logger.warning(f"Failed to upload data to SwanLab: {e}")


def main():
    args = parse_args()

    # Handle boolean flags
    use_bf16 = args.bf16 and not args.no_bf16
    use_gc = args.gradient_checkpointing and not args.no_gradient_checkpointing

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Train data: {args.train_data_path}")
    logger.info(f"Eval data: {args.eval_data_path}")
    logger.info(f"Output: {args.output_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    logger.info("Loading datasets...")
    train_ds = load_dataset_auto(args.train_data_path, args.dataset_format)
    eval_ds = load_dataset_auto(args.eval_data_path, args.dataset_format) if args.eval_data_path else None
    logger.info(f"Train: {len(train_ds)} samples, Eval: {len(eval_ds) if eval_ds else 'N/A'}")

    # Validate schema
    required_cols = {"prompt", "chosen", "rejected"}
    missing = required_cols - set(train_ds.column_names)
    if missing:
        raise ValueError(f"Missing columns in train data: {missing}. Available: {train_ds.column_names}")

    # Determine model for trainer
    # Non-LoRA: pass model path string, let trainer handle loading with model_init_kwargs
    # LoRA: pre-load base model and apply PEFT, trainer creates ref_model from base
    if args.use_lora:
        logger.info("Loading model for LoRA...")
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if use_bf16 else torch.float32,
            "trust_remote_code": True,
        }
        if use_gc:
            model_kwargs["use_cache"] = False
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
        if use_gc:
            model.gradient_checkpointing_enable()
        from peft import get_peft_model, LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        logger.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, target={args.lora_target_modules}")
        model_for_trainer = model
    else:
        logger.info(f"Model will be loaded by trainer from: {args.model_name_or_path}")
        model_for_trainer = args.model_name_or_path
        peft_config = None

    # Build DPOConfig
    report_to = args.report_to.split(",") if args.report_to else ["none"]
    logger.info(f"Report to: {report_to}")

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size or args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=use_bf16,
        gradient_checkpointing=use_gc,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_ds else None,
        save_total_limit=args.save_total_limit,
        report_to=report_to,
        run_name=args.run_name,
        project=args.project_name,
        beta=args.beta,
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,
        precompute_ref_log_probs=args.precompute_ref_log_probs,
        max_length=args.max_length,
        truncation_mode=args.truncation_mode,
        save_strategy="steps",
        eval_strategy="steps" if eval_ds else "no",
        load_best_model_at_end=True if eval_ds else False,
        metric_for_best_model="eval_loss" if eval_ds else None,
        greater_is_better=False if eval_ds else None,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        logging_first_step=True,  # 记录第一步，避免长时间无指标
        # model_init_kwargs 控制 main model 和 ref model 的加载
        # device_map={"": "cpu"} 让模型先加载到 CPU，FSDP 接管后分片到各 GPU
        # 避免 device_map="auto" 导致的 FSDP 设备冲突
        model_init_kwargs={
            "torch_dtype": torch.bfloat16 if use_bf16 else torch.float32,
            "trust_remote_code": True,
            "device_map": {"": "cpu"},  # FSDP: 加载到 CPU，让 FSDP 分片
        },
    )

    # 手动初始化 SwanLab，确保 run 在 trainer 创建前已就绪
    # 只在 rank 0 初始化，避免 FSDP 多进程重复创建 run
    if "swanlab" in report_to:
        try:
            from accelerate.state import PartialState
            is_main = PartialState().is_main_process
        except Exception:
            is_main = True
        if is_main:
            import swanlab
            swanlab.init(
                project=args.project_name,
                experiment_name=args.run_name or "dpo_run",
            )
            logger.info(f"SwanLab initialized: project={args.project_name}")

    # Create trainer
    logger.info("Creating DPOTrainer...")
    trainer = DPOTrainer(
        model=model_for_trainer,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    logger.info("DPOTrainer created successfully")

    # Train
    logger.info("Starting DPO training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Upload data to SwanLab after training (SwanLab is initialized by then)
    if args.upload_data_to_swanlab and "swanlab" in report_to:
        logger.info("Uploading training data to SwanLab...")
        upload_data_to_swanlab(train_ds, eval_ds, args)

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
