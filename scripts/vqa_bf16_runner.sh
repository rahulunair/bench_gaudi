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

echo "Starting VQA BF16 benchmarks for ${MODEL_SIZE} model..."
echo "=================================================="

./scripts/vqa_bf16.sh --model-size $MODEL_SIZE

echo "=================================================="
echo "VQA BF16 benchmarks completed for ${MODEL_SIZE} model"
