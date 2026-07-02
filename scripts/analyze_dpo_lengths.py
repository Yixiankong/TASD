#!/usr/bin/env python3
"""Analyze DPO data token lengths — minimal deps version."""
import argparse
import json
import pyarrow.parquet as pq
from transformers import AutoTokenizer
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--sample", type=int, default=0, help="Only analyze first N samples (0=all)")
    args = parser.parse_args()

    print(f"Loading data from {args.data_path} ...")
    table = pq.read_table(args.data_path)
    prompts = table.column("prompt").to_pylist()
    chosens = table.column("chosen").to_pylist()
    rejects = table.column("rejected").to_pylist()
    n_total = len(prompts)
    print(f"Loaded {n_total} samples")

    if args.sample and args.sample < n_total:
        prompts = prompts[:args.sample]
        chosens = chosens[:args.sample]
        rejects = rejects[:args.sample]
        print(f"Analyzing first {args.sample} samples")

    print(f"Loading tokenizer from {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    label = args.label or args.data_path.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Analysis: {label}")
    print(f"max_length = {args.max_length}")
    print(f"{'='*60}")

    prompt_lens = []
    chosen_lens = []
    rejected_lens = []

    for i in range(len(prompts)):
        p_tokens = tokenizer.encode(prompts[i], add_special_tokens=False)
        c_tokens = tokenizer.encode(str(chosens[i]), add_special_tokens=False)
        r_tokens = tokenizer.encode(str(rejects[i]), add_special_tokens=False)
        prompt_lens.append(len(p_tokens))
        chosen_lens.append(len(c_tokens))
        rejected_lens.append(len(r_tokens))

        if (i + 1) % 1000 == 0:
            print(f"  Tokenized {i+1}/{len(prompts)} ...")

    prompt_lens = np.array(prompt_lens)
    chosen_lens = np.array(chosen_lens)
    rejected_lens = np.array(rejected_lens)
    total_chosen = prompt_lens + chosen_lens
    total_rejected = prompt_lens + rejected_lens
    max_total = np.maximum(total_chosen, total_rejected)

    def stats(name, arr, show_trunc=None):
        print(f"\n  {name}:")
        print(f"    Mean:   {arr.mean():.0f}")
        print(f"    Median: {np.median(arr):.0f}")
        print(f"    Min:    {arr.min()}")
        print(f"    Max:    {arr.max()}")
        print(f"    P25/P75: {np.percentile(arr, 25):.0f} / {np.percentile(arr, 75):.0f}")
        print(f"    P90/P95/P99: {np.percentile(arr, 90):.0f} / {np.percentile(arr, 95):.0f} / {np.percentile(arr, 99):.0f}")
        if show_trunc:
            n = (arr > show_trunc).sum()
            print(f"    > {show_trunc} tokens: {n}/{len(arr)} ({n/len(arr):.1%})")

    stats("Prompt length", prompt_lens)
    stats("Chosen completion length", chosen_lens)
    stats("Rejected completion length", rejected_lens)
    stats("Total (prompt + chosen)", total_chosen, show_trunc=args.max_length)
    stats("Total (prompt + rejected)", total_rejected, show_trunc=args.max_length)
    stats("Total (max of chosen/rejected)", max_total, show_trunc=args.max_length)

    # Remaining space for completion
    remaining = args.max_length - prompt_lens
    print(f"\n  Remaining space for completion ({args.max_length} - prompt):")
    print(f"    Mean:   {remaining.mean():.0f} tokens")
    print(f"    Median: {np.median(remaining):.0f} tokens")
    print(f"    Min:    {remaining.min()} tokens  {'⚠️ NEGATIVE - no space!' if remaining.min() <= 0 else ''}")
    print(f"    Max:    {remaining.max()} tokens")

    no_space = (remaining <= 0).sum()
    if no_space > 0:
        print(f"    ❌ {no_space}/{len(remaining)} ({no_space/len(remaining):.1%}) samples have ZERO or NEGATIVE space for completion!")

    # Truncation severity
    print(f"\n  Truncation analysis (max_length={args.max_length}):")
    truncated_mask = max_total > args.max_length
    n_truncated = truncated_mask.sum()
    print(f"    Truncated samples: {n_truncated}/{n_total} ({n_truncated/n_total:.1%})")

    if n_truncated > 0:
        t_prompt = prompt_lens[truncated_mask]
        t_chosen = chosen_lens[truncated_mask]
        t_rejected = rejected_lens[truncated_mask]
        rem_for_c = args.max_length - t_prompt
        c_surviving = np.maximum(rem_for_c, 0)
        r_surviving = np.maximum(args.max_length - t_prompt, 0)

        print(f"\n    For {n_truncated} truncated samples:")
        print(f"      Avg prompt length: {t_prompt.mean():.0f}")
        print(f"      Avg chosen completion: {t_chosen.mean():.0f} tokens → surviving: {c_surviving.mean():.0f} tokens ({c_surviving.mean()/max(t_chosen.mean(),1)*100:.1f}%)")
        print(f"      Avg rejected completion: {t_rejected.mean():.0f} tokens → surviving: {r_surviving.mean():.0f} tokens ({r_surviving.mean()/max(t_rejected.mean(),1)*100:.1f}%)")

        chosen_fully_lost = (rem_for_c <= 0).sum()
        if chosen_fully_lost > 0:
            print(f"      ❌ Chosen completion COMPLETELY lost: {chosen_fully_lost} samples")

    print()


if __name__ == "__main__":
    main()
