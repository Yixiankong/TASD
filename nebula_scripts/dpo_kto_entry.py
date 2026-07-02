#!/usr/bin/env python3
"""Simplified Nebula entry point for offline DPO/KTO training.
Unlike entry.py, this does NOT start a Ray cluster since DPO/KTO
use HuggingFace Accelerate for distributed training instead."""
import os
import sys
import argparse
import subprocess


def dedup_path(path_str):
    """Remove duplicates from PATH-like environment variables."""
    if not path_str:
        return path_str
    seen = set()
    result = []
    for p in path_str.split(":"):
        if p and p not in seen:
            seen.add(p)
            result.append(p)
    return ":".join(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_path", type=str, required=True,
                        help="Path to parametric training script")
    parser.add_argument("--job_name", type=str, default="dpo_kto_job",
                        help="Job name for logging")
    parser.add_argument("--env", type=str, action="append", default=[],
                        help="Environment variables as KEY=VALUE pairs")
    args = parser.parse_args()

    print(f"[dpo_kto_entry.py] Starting job: {args.job_name}")
    print(f"[dpo_kto_entry.py] Script: {args.script_path}")

    # Build environment
    env = os.environ.copy()

    # Inject custom env vars
    for kv in args.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
            print(f"[dpo_kto_entry.py] inject: {k}={v}")

    # Prevent ~/.local from shadowing conda env packages
    env["PYTHONNOUSERSITE"] = "1"

    # Activate conda env (same pattern as launch_ray_cluster.sh)
    conda_bin = "/opt/conda/envs/dpo_env/bin"
    if os.path.isdir(conda_bin):
        env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
        print(f"[dpo_kto_entry.py] Activated conda env: dpo_env")

    # Set PYTHONPATH
    cwd = os.getcwd()
    env["PYTHONPATH"] = f"{cwd}:{env.get('PYTHONPATH', '')}"

    # Clean LD_LIBRARY_PATH (same dedup logic as launch_ray_cluster.sh)
    ld_path = env.get("LD_LIBRARY_PATH", "")
    if ld_path:
        env["LD_LIBRARY_PATH"] = dedup_path(ld_path)

    # Clean PATH
    path = env.get("PATH", "")
    if path:
        env["PATH"] = dedup_path(path)

    # Detect GPUs
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        n_gpus = len(result.stdout.strip().split("\n"))
        env["N_GPUS"] = str(n_gpus)
        print(f"[dpo_kto_entry.py] Detected {n_gpus} GPUs")
    except Exception as e:
        print(f"[dpo_kto_entry.py] Warning: Could not detect GPUs: {e}")
        env["N_GPUS"] = "1"

    # Execute parametric script
    cmd = ["bash", args.script_path]
    print(f"[dpo_kto_entry.py] Executing: {' '.join(cmd)}")
    ret = subprocess.run(cmd, env=env)
    sys.exit(ret.returncode)
