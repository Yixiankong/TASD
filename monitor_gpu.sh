#!/bin/bash
# GPU memory monitor - logs every 5 seconds
LOG_FILE="/data/oss_bucket_0/ad/kongyixian.kyx/logs/training_logs/gpu_memory_$(date +%Y%m%d_%H%M%S).csv"
echo "timestamp,memory_used_MiB,memory_total_MiB,utilization_pct" > "$LOG_FILE"
echo "Logging GPU memory to $LOG_FILE"
while true; do
    ts=$(date +"%H:%M:%S")
    mem=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    echo "$ts,$mem" >> "$LOG_FILE"
    sleep 5
done
