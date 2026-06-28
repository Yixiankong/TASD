#!/usr/bin/env python3
"""Convert cross-sample pairs (JSONL) to KTO-standard parquet format.

Input format (cross_sample_pairs.jsonl):
    {"pair_type": "cross", "chosen_prompt": [{messages}], "rejected_prompt": [{messages}],
     "chosen": [card_ids], "rejected": [card_ids], ...}

Conversion: Each cross-sample pair generates 2 KTO samples:
    - chosen_prompt + chosen (serialized to JSON string) -> label=True
    - rejected_prompt + rejected (serialized to JSON string) -> label=False

Output schema:
    {prompt, completion, label}
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
    """Convert response to string. Lists (card rankings) become JSON strings."""
    if isinstance(val, list):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def load_input(input_path: str, input_format: str):
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
        return load_jsonl(input_path)
    elif input_format == "parquet":
        return datasets.load_dataset("parquet", data_files=input_path, split="train")
    elif input_format in ("json",):
        return datasets.load_dataset("json", data_files=input_path, split="train")
    elif input_format == "hf_dataset":
        return datasets.load_dataset(input_path, split="train")
    else:
        raise ValueError(f"Unsupported input format: {input_format}")


def convert_cross_pairs_to_kto(records: list, args, tokenizer=None, enable_thinking: bool = False) -> list:
    """Convert cross-sample pairs to KTO format (2 samples per pair)."""
    kto_records = []
    for i, row in enumerate(records):
        chosen_prompt = row.get(args.chosen_prompt_column)
        rejected_prompt = row.get(args.rejected_prompt_column)
        chosen_resp = row.get(args.chosen_response_column)
        rejected_resp = row.get(args.rejected_response_column)

        if chosen_prompt is None or rejected_prompt is None:
            print(f"Warning: row {i} missing prompt fields, skipping")
            continue
        if chosen_resp is None or rejected_resp is None:
            print(f"Warning: row {i} missing response fields, skipping")
            continue

        # Apply chat template to convert prompt messages to string
        if tokenizer:
            if isinstance(chosen_prompt, list):
                chosen_prompt = tokenizer.apply_chat_template(
                    chosen_prompt, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
            if isinstance(rejected_prompt, list):
                rejected_prompt = tokenizer.apply_chat_template(
                    rejected_prompt, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )

        # Chosen sample -> label=True
        kto_records.append({
            "prompt": chosen_prompt,
            "completion": serialize_response(chosen_resp),
            "label": True,
        })

        # Rejected sample -> label=False
        kto_records.append({
            "prompt": rejected_prompt,
            "completion": serialize_response(rejected_resp),
            "label": False,
        })

    return kto_records


def prepare_kto_data(args):
    """Main preparation logic."""
    print(f"Loading data from {args.input_path} ...")
    raw = load_input(args.input_path, args.input_format)

    # Handle list (from direct JSONL load) vs HF Dataset
    if isinstance(raw, list):
        records = raw
    else:
        records = [dict(row) for row in raw]
    print(f"Loaded {len(records)} cross-sample pairs")

    # Validate required columns
    required = {
        args.chosen_prompt_column,
        args.rejected_prompt_column,
        args.chosen_response_column,
        args.rejected_response_column,
    }
    first_keys = set(records[0].keys()) if records else set()
    missing = required - first_keys
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available: {first_keys}"
        )

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

        # Convert cross pairs to KTO format
        kto_records = convert_cross_pairs_to_kto(
            records, args, tokenizer=tokenizer, enable_thinking=enable_thinking
        )
        print(f"Generated {len(kto_records)} KTO samples from {len(records)} pairs")

        kto_ds = datasets.Dataset.from_list(kto_records)

        # Train/test split
        if args.test_ratio > 0:
            split = kto_ds.train_test_split(test_size=args.test_ratio, seed=args.seed)
            train_ds = split["train"]
            test_ds = split["test"]
            print(f"Split: {len(train_ds)} train, {len(test_ds)} test")
        else:
            train_ds = kto_ds
            test_ds = None
            print(f"Train: {len(train_ds)} rows (no test split)")

        # Statistics
        print(f"\n=== Data Statistics ===")
        print(f"Columns: {train_ds.column_names}")
        labels = train_ds["label"]
        n_positive = sum(1 for l in labels if l)
        n_negative = len(labels) - n_positive
        print(f"Positive samples (label=True): {n_positive}")
        print(f"Negative samples (label=False): {n_negative}")
        print(f"Ratio: {n_positive / len(labels):.2%} positive")

        sample = train_ds[0]
        if isinstance(sample["prompt"], list):
            prompt_text = "\n".join(
                m.get("content", "") for m in sample["prompt"] if isinstance(m, dict)
            )
        else:
            prompt_text = str(sample["prompt"])
        print(f"Sample prompt (last 100 chars): ...{prompt_text[-100:]}")
        print(f"Sample completion (first 200 chars): {str(sample['completion'])[:200]}...")
        print(f"Sample label: {sample['label']}")

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
        description="Convert cross-sample pairs to KTO-standard parquet format."
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
        "--chosen_prompt_column",
        type=str,
        default="chosen_prompt",
        help="Column name for chosen prompt",
    )
    parser.add_argument(
        "--rejected_prompt_column",
        type=str,
        default="rejected_prompt",
        help="Column name for rejected prompt",
    )
    parser.add_argument(
        "--chosen_response_column",
        type=str,
        default="chosen",
        help="Column name for chosen response",
    )
    parser.add_argument(
        "--rejected_response_column",
        type=str,
        default="rejected",
        help="Column name for rejected response",
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
    prepare_kto_data(args)


if __name__ == "__main__":
    main()
