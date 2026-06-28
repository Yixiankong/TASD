#!/usr/bin/env python3
"""Convert within-sample pairs (JSONL) to DPO-standard parquet format.

Input format (within_sample_pairs.jsonl):
    {"sample_id": "...", "prompt": [{messages}], "chosen": "str", "rejected": "str", ...}

Output schema:
    {prompt, chosen, rejected} + optional metadata columns
"""
import os
import argparse
import json

import datasets
import pyarrow as pa
import pyarrow.parquet as pq


def _to_large(field: pa.Field) -> pa.Field:
    t = field.type
    if pa.types.is_string(t):
        return pa.field(field.name, pa.large_string(), field.nullable, field.metadata)
    if pa.types.is_binary(t):
        return pa.field(field.name, pa.large_binary(), field.nullable, field.metadata)
    if pa.types.is_list(t):
        return pa.field(
            field.name,
            pa.large_list(_to_large(pa.field("item", t.value_type)).type),
            field.nullable,
            field.metadata,
        )
    if pa.types.is_struct(t):
        return pa.field(
            field.name,
            pa.struct(
                [
                    _to_large(pa.field(f.name, f.type, f.nullable, f.metadata))
                    for f in t
                ]
            ),
            field.nullable,
            field.metadata,
        )
    return field


def _large_schema(schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in schema]
    )


def write_rowgrouped_large(ds, path: str, rows_per_group: int = 32):
    """Cast to LargeString/LargeList and write many small row groups."""
    # Use ds[:] to get only the selected indices (important after train_test_split)
    tbl: pa.Table = pa.Table.from_pydict(ds[:], schema=ds.features.arrow_schema)
    tbl = tbl.cast(_large_schema(tbl.schema))
    n = len(tbl)
    writer = None
    try:
        for start in range(0, n, rows_per_group):
            chunk = tbl.slice(start, min(rows_per_group, n - start))
            if writer is None:
                writer = pq.ParquetWriter(path, chunk.schema, compression="zstd")
            writer.write_table(chunk)
    finally:
        if writer is not None:
            writer.close()


def load_jsonl(input_path: str) -> list:
    """Load JSONL file directly (bypasses HF datasets mixed-struct issues)."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def serialize_response(val):
    """Convert chosen/rejected to string. Lists (card rankings) become JSON strings."""
    if isinstance(val, list):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def load_input(input_path: str, input_format: str) -> datasets.Dataset:
    """Load dataset from various formats."""
    if input_format == "auto":
        if input_path.endswith(".parquet"):
            input_format = "parquet"
        elif input_path.endswith(".jsonl"):
            input_format = "jsonl"
        elif input_path.endswith(".json"):
            input_format = "json"
        elif os.path.isdir(input_path):
            input_format = "hf_dataset"
        else:
            input_format = "hf_dataset"

    if input_format == "jsonl":
        # Direct JSONL load to avoid HF mixed-struct detection issues
        records = load_jsonl(input_path)
        print(f"Loaded {len(records)} records via direct JSONL read")
        return records  # return as list, will be converted later
    elif input_format == "parquet":
        ds = datasets.load_dataset("parquet", data_files=input_path, split="train")
        return ds
    elif input_format in ("json",):
        return datasets.load_dataset("json", data_files=input_path, split="train")
    elif input_format == "hf_dataset":
        return datasets.load_dataset(input_path, split="train")
    else:
        raise ValueError(f"Unsupported input format: {input_format}")


def prepare_dpo_data(args):
    """Main preparation logic."""
    print(f"Loading data from {args.input_path} ...")
    raw = load_input(args.input_path, args.input_format)

    # Handle list (from direct JSONL load) vs HF Dataset
    if isinstance(raw, list):
        records = raw
    else:
        records = [dict(row) for row in raw]
    print(f"Loaded {len(records)} rows")

    # Load tokenizer for chat template (if model provided)
    tokenizer = None
    if args.model_name_or_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        print(f"Loaded tokenizer from {args.model_name_or_path}")

    # Determine which thinking modes to generate
    # When tokenizer is available and args.both_thinking_modes is True, generate both variants
    modes = [args.enable_thinking]
    if tokenizer and args.both_thinking_modes:
        modes = [True, False]  # think first, then no_think

    for enable_thinking in modes:
        mode_tag = "think" if enable_thinking else "nothink"
        print(f"\n{'='*50}")
        print(f"Generating {mode_tag} variant (enable_thinking={enable_thinking})")
        print(f"{'='*50}")

        # Build DPO records with column renaming and serialization
        dpo_records = []
        skipped = 0
        for row in records:
            prompt = row.get(args.prompt_column, row.get("prompt"))
            chosen = row.get(args.chosen_column, row.get("chosen"))
            rejected = row.get(args.rejected_column, row.get("rejected"))

            if prompt is None or chosen is None or rejected is None:
                skipped += 1
                continue

            # Apply chat template to convert prompt messages to string
            if tokenizer and isinstance(prompt, list):
                prompt = tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )

            record = {
                "prompt": prompt,
                "chosen": serialize_response(chosen),
                "rejected": serialize_response(rejected),
            }
            dpo_records.append(record)

        if skipped:
            print(f"Skipped {skipped} rows with missing fields")

        ds = datasets.Dataset.from_list(dpo_records)
        print(f"Built DPO dataset with {len(ds)} rows")

        # Train/test split
        if args.test_ratio > 0:
            split = ds.train_test_split(test_size=args.test_ratio, seed=args.seed)
            train_ds = split["train"]
            test_ds = split["test"]
            print(f"Split: {len(train_ds)} train, {len(test_ds)} test")
        else:
            train_ds = ds
            test_ds = None
            print(f"Train: {len(train_ds)} rows (no test split)")

        # Statistics
        print(f"\n=== Data Statistics ===")
        print(f"Columns: {train_ds.column_names}")
        sample = train_ds[0]
        if isinstance(sample["prompt"], list):
            prompt_text = "\n".join(
                m.get("content", "") for m in sample["prompt"] if isinstance(m, dict)
            )
        else:
            prompt_text = str(sample["prompt"])
        print(f"Sample prompt (last 100 chars): ...{prompt_text[-100:]}")
        print(f"Sample chosen (first 200 chars): {str(sample['chosen'])[:200]}...")
        print(f"Sample rejected (first 200 chars): {str(sample['rejected'])[:200]}...")

        # Determine output paths
        if args.both_thinking_modes:
            # Add mode tag to filename: base.parquet -> base_think.parquet / base_nothink.parquet
            base, ext = os.path.splitext(args.output_path)
            train_path = f"{base}_{mode_tag}{ext}"
            test_path = f"{base}_{mode_tag}_test{ext}"
        else:
            train_path = args.output_path
            test_path = None

        # Write output
        os.makedirs(os.path.dirname(train_path) or ".", exist_ok=True)
        write_rowgrouped_large(train_ds, train_path)
        print(f"Wrote train data to {train_path} ({len(train_ds)} rows)")

        if test_ds is not None:
            write_rowgrouped_large(test_ds, test_path)
            print(f"Wrote test data to {test_path} ({len(test_ds)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert within-sample pairs to DPO-standard parquet format."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input file path (parquet/json/jsonl) or HF dataset ID",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output parquet path (test split will be saved with _test suffix)",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        choices=["auto", "parquet", "json", "jsonl", "hf_dataset"],
        default="auto",
        help="Input format (default: auto-detect from extension)",
    )
    parser.add_argument(
        "--prompt_column", type=str, default="prompt", help="Column name for prompt"
    )
    parser.add_argument(
        "--chosen_column", type=str, default="chosen", help="Column name for chosen response"
    )
    parser.add_argument(
        "--rejected_column", type=str, default="rejected", help="Column name for rejected response"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="Test split ratio (0 = no split)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Tokenizer path for applying chat template to prompt messages")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Enable thinking mode in chat template (default: False)")
    parser.add_argument("--both_thinking_modes", action="store_true", default=False,
                        help="Generate both 'think' and 'nothink' variants in one run (requires tokenizer)")

    args = parser.parse_args()
    prepare_dpo_data(args)


if __name__ == "__main__":
    main()
