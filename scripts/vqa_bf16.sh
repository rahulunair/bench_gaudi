#!/bin/bash

# Visual Question Answering BF16 Script
#
# Usage:
#   ./scripts/vqa_bf16.sh [options] <output_tokens> <batch_size>
#
# Options:
#   --model-size <11b|90b>  Model size to run (default: 11b)
#                          11b: Runs on 1 card
#                          90b: Runs on 8 cards
#   --help                Display this help message
#
# Example:
#   ./scripts/vqa_bf16.sh --model-size 11b 128 256  # Run 11B model with output_tokens=128, batch_size=256
#   ./scripts/vqa_bf16.sh --model-size 90b 128 64   # Run 90B model with output_tokens=128, batch_size=64

export HF_DATASETS_TRUST_REMOTE_CODE=true
export PT_HPU_MEMORY_LIMIT=99

MODEL_SIZE="11b"
WORLD_SIZE=1  # Default for 11B

show_help() {
    grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# \?//'
    exit 0
}

# Parse options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-size) 
            MODEL_SIZE="$2"
            if [ "$2" = "90b" ]; then
                WORLD_SIZE=8
            fi
            shift ;;
        --help) show_help ;;
        -*) echo "[Error] Unknown parameter: $1"; show_help ;;
        *) break ;;
    esac
    shift
done

# Check remaining arguments
if [ $# -ne 2 ]; then
    echo "[Error] Missing required arguments"
    show_help
fi

OUTPUT_TOKENS=$1
BATCH_SIZE=$2

if [ "$MODEL_SIZE" = "90b" ]; then
    MODEL="meta-llama/Llama-3.2-90B-Vision"
    MODEL_NAME="llama3.2-90b-vision"
else
    MODEL="meta-llama/Llama-3.2-11B-Vision"
    MODEL_NAME="llama3.2-11b-vision"
fi

OPTIMUM_PATH="/workspace/optimum-habana/examples"
OUTPUT_DIR="benchmark_results/${MODEL_NAME}/vqa"
RUN_DIR="${OUTPUT_DIR}/tokens_${OUTPUT_TOKENS}_bs_${BATCH_SIZE}"
mkdir -p "$RUN_DIR"

echo "Running VQA BF16 for ${MODEL_SIZE} model on ${WORLD_SIZE} cards"
echo "Configuration: OUTPUT_TOKENS=$OUTPUT_TOKENS, BATCH_SIZE=$BATCH_SIZE"
echo "Logs saved in: $RUN_DIR"

python3 $OPTIMUM_PATH/gaudi_spawn.py \
    --world_size $WORLD_SIZE \
    --use_deepspeed \
    ./scripts/pipeline_wrapper.py \
    --model_name_or_path "$MODEL" \
    --use_kv_cache \
    --warmup 3 \
    --n_iterations 5 \
    --use_hpu_graphs \
    --bf16 \
    --max_new_tokens $OUTPUT_TOKENS \
    --batch_size $BATCH_SIZE \
    --use_flash_attention \
    --flash_attention_recompute 2>&1 | tee -a "${RUN_DIR}/run.log"

if [ $? -eq 0 ]; then
    echo "Configuration OUTPUT_TOKENS=$OUTPUT_TOKENS, BATCH_SIZE=$BATCH_SIZE completed successfully."
else
    echo "[Error] Configuration failed. Check logs for details."
    exit 1
fi
