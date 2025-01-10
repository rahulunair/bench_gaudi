#!/bin/bash

# Run LLama Vision Model in BF16 mode
#
# Usage:
#   ./run_benchmarks.sh [options]
#
# Options:
#   --model-size <11b|90b>  Model size to run (default: 11b)
#                          11b: Runs on 1 card
#                          90b: Runs on 8 cards
#   --help                Display this help message
#
# Example:
#   ./run_benchmarks.sh --model-size 11b  # Run 11B model on 1 card
#   ./run_benchmarks.sh --model-size 90b  # Run 90B model on 8 cards

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
            if [ "$2" = "90b" ]; then
                WORLD_SIZE=8
            fi
            shift ;;
        --help) show_help ;;
        *) echo "[Error] Unknown parameter: $1"; show_help ;;
    esac
    shift
done

if [ "$MODEL_SIZE" = "90b" ]; then
    MODEL="meta-llama/Llama-3.2-90B-Vision"
    MODEL_NAME="llama3.2-90b-vision"
    CONFIGS=(
        "128 128 8"
        "128 128 256"   
        "128 2048 128"  
        "2048 128 64"   
        "2048 2048 64"  
    )
else
    MODEL="meta-llama/Llama-3.2-11B-Vision"
    MODEL_NAME="llama3.2-11b-vision"
    CONFIGS=(
        "128 128 1"
        "128 128 1024"   
        "128 2048 128"   
        "2048 128 96"    
        "2048 2048 64"   
    )
fi

OPTIMUM_PATH="/workspace/optimum-habana/examples"
OUTPUT_DIR="benchmark_results/${MODEL_NAME}/bf16"
mkdir -p "$OUTPUT_DIR"
echo "Running BF16 benchmarks for ${MODEL_SIZE} model on ${WORLD_SIZE} cards..."

for config in "${CONFIGS[@]}"; do
    read -r input_len output_len batch_s <<< "$config"
    echo "Running with input_tokens=$input_len output_tokens=$output_len batch_size=$batch_s"
    python3 $OPTIMUM_PATH/gaudi_spawn.py \
        --use_deepspeed \
        --world_size $WORLD_SIZE \
        $OPTIMUM_PATH/text-generation/run_generation.py \
        --model_name_or_path "$MODEL" \
        --bucket_size=128 \
        --use_hpu_graphs \
        --limit_hpu_graphs \
        --max_input_tokens $input_len \
        --max_new_tokens $output_len \
        --batch_size $batch_s \
        --bf16 \
        --warmup 2 \
        --trim_logits \
        --attn_softmax_bf16 \
        --use_flash_attention \
        --flash_attention_recompute \
        --flash_attention_causal_mask 2>&1 | tee -a "${OUTPUT_DIR}/run_b${batch_s}.log"
done

echo "All configurations completed for ${MODEL_SIZE} model. Logs saved in ${OUTPUT_DIR}/"
