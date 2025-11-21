#!/bin/bash
# 检测 GPU 空闲显存，大于 40GB 时运行 Python 程序

# 目标空闲显存阈值（单位：MB）
THRESHOLD=$((40 * 1024))   # 40 GB = 40960 MB

# 检测间隔（秒）
INTERVAL=600

# 你的 Python 程序命令
PYTHON_CMD="python AS2.py"

echo "🚀 Waiting for GPU with > ${THRESHOLD} MB free memory..."

while true; do
    # 从 nvidia-smi 提取每张卡的可用显存（MiB）
    AVAILABLE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -nr | head -n1)

    # 转换为整数
    FREE_MEM=$(echo $AVAILABLE | awk '{print int($1)}')

    echo "🧠 Current free GPU memory: ${FREE_MEM} MB"

    if [ "$FREE_MEM" -gt "$THRESHOLD" ]; then
        echo "✅ GPU has enough free memory (${FREE_MEM} MB > ${THRESHOLD} MB)."
        echo "▶️ Running: $PYTHON_CMD"
        eval $PYTHON_CMD
        exit 0
    else
        echo "⏳ Not enough memory, checking again in ${INTERVAL}s..."
        sleep $INTERVAL
    fi
done
