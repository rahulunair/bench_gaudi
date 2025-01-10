#!/bin/bash

# Visual Question Answering BF16 Runner Script
#
# Usage:
#   ./scripts/vqa_bf16_runner.sh [options]
#
# Options:
#   --model-size <11b|90b>  Model size to run (default: 11b)
#                          11b: Runs on 1 card
#                          90b: Runs on 8 cards
#   --help                Display this help message
#
# Example:
#   ./scripts/vqa_bf16_runner.sh --model-size 11b  # Run VQA benchmarks for 11B model
#   ./scripts/vqa_bf16_runner.sh --model-size 90b  # Run VQA benchmarks for 90B model

export HF_DATASETS_TRUST_REMOTE_CODE=true
export PT_HPU_MEMORY_LIMIT=99

MODEL_SIZE="11b"
WAIT_TIME=10

show_help() {
    grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# \?//'
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-size) 
            MODEL_SIZE="$2"
            shift ;;
        --help) show_help ;;
        *) echo "[Error] Unknown parameter: $1"; show_help ;;
    esac
    shift
done

# Define configurations based on model size
if [ "$MODEL_SIZE" = "90b" ]; then
    # Configs for 90B model
    CONFIGS=(
        "128 8"
        "128 64"
        "128 128" 
        "128 256"    # From 1750 (70B)
        "128 128"    # From 512 (70B)
        "2048 64"    # From 242 (70B)
        "2048 64"    # From 241 (70B)
    )
else
    # Configs for 11B model
    CONFIGS=(
        "128 1"
        "128 8"
        "128 32"
        "128 64"
        "128 128"
        "128 1024"   # From 1536 (7B)
        "2048 96"    # From 153 (7B)
        "2048 64"    # From 117 (7B)
    )
fi

echo "Starting VQA BF16 benchmarks for ${MODEL_SIZE} model..."
echo "=================================================="

for config in "${CONFIGS[@]}"; do
    read -r output_tokens batch_size <<< "$config"
    echo "Running configuration: OUTPUT_TOKENS=$output_tokens, BATCH_SIZE=$batch_size"
    echo "--------------------------------------------"
    
    ./scripts/vqa_bf16.sh --model-size $MODEL_SIZE $output_tokens $batch_size
    if [ $? -ne 0 ]; then
        echo "[Error] Configuration failed. Stopping further execution."
        exit 1
    fi
    
    echo "Waiting $WAIT_TIME seconds before next run..."
    sleep $WAIT_TIME
done

echo "=================================================="
echo "VQA BF16 benchmarks completed for ${MODEL_SIZE} model"
