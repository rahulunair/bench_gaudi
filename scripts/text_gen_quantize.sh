#!/bin/bash

# Quanize and Run LLama Vision Model
#
# Usage:
#   ./quantize_model.sh [options]
#
# Options:
#   --model-size <11b|90b>  Model size to run (default: 11b)
#                          11b: Runs on 1 card
#                          90b: Runs on 8 cards
#   --measure              Run measurement phase for quantization
#   --help                Display this help message
#
# Example:
#   ./quantize_model.sh --model-size 11b --measure  # Run 11B model with measurement
#   ./quantize_model.sh --model-size 90b           # Run 90B model directly

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
        --measure) RUN_MEASURE=true ;;          
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

EXAMPLES_PATH="/workspace/optimum-habana/examples"  
TEXT_GEN_PATH="$EXAMPLES_PATH/text-generation"      
OUTPUT_DIR="benchmark_results/${MODEL_NAME}/fp8"       
QUANT_CONFIG_DIR="$TEXT_GEN_PATH/quantization_config"  
MEASURE_CONFIG="maxabs_measure_include_outputs.json"
QUANT_CONFIG="maxabs_quant.json"
mkdir -p "$OUTPUT_DIR"

# Stage 1: Measure Tensor Quantization Statistics
if [ "$RUN_MEASURE" = true ]; then
    echo "Starting quant stats measurement for model: $MODEL on $WORLD_SIZE cards." | tee "${OUTPUT_DIR}/run_measure.log"
    QUANT_CONFIG="$QUANT_CONFIG_DIR/$MEASURE_CONFIG" python $EXAMPLES_PATH/gaudi_spawn.py \
        --use_deepspeed \
        --world_size $WORLD_SIZE \
        $TEXT_GEN_PATH/run_lm_eval.py \
        --model_name_or_path "$MODEL" \
        --use_kv_cache \
        --bucket_size=128 \
        --use_hpu_graphs \
        --trim_logits \
        --batch_size 1 \
        --bf16 \
        --use_flash_attention \
        --flash_attention_recompute \
        --flash_attention_causal_mask \
        -o "${OUTPUT_DIR}/measure_results.txt" 2>&1 | tee -a "${OUTPUT_DIR}/run_measure.log"
    echo "Measurement completed. Results saved in ${OUTPUT_DIR}/measure_results.txt"
else
    echo "Skipping quantization measurement phase."
fi

# Stage 2: Quantize and Run the Model
sleep 10
echo "Starting quantization with batch size 1 for ${MODEL_SIZE} model on ${WORLD_SIZE} cards..."
QUANT_CONFIG="$QUANT_CONFIG_DIR/$QUANT_CONFIG" python $EXAMPLES_PATH/gaudi_spawn.py \
    --use_deepspeed \
    --world_size $WORLD_SIZE \
    $TEXT_GEN_PATH/run_generation.py \
    --model_name_or_path "$MODEL" \
    --bucket_size=128 \
    --use_hpu_graphs \
    --limit_hpu_graphs \
    --max_input_tokens 128 \
    --max_new_tokens 128 \
    --batch_size 1 \
    --bf16 \
    --warmup 2 \
    --trim_logits \
    --attn_softmax_bf16 \
    --use_flash_attention \
    --flash_attention_recompute \
    --flash_attention_causal_mask 2>&1 | tee -a "${OUTPUT_DIR}/run_quant_b1.log"
    
sleep 10
for config in "${CONFIGS[@]}"; do
    read -r input_len output_len batch_s <<< "$config"
    echo "Running ${MODEL_SIZE} model with input_tokens=$input_len output_tokens=$output_len batch_size=$batch_s on ${WORLD_SIZE} cards"
    
    QUANT_CONFIG="$QUANT_CONFIG_DIR/$QUANT_CONFIG" python $EXAMPLES_PATH/gaudi_spawn.py \
        --use_deepspeed \
        --world_size $WORLD_SIZE \
        $TEXT_GEN_PATH/run_generation.py \
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
        --flash_attention_causal_mask 2>&1 | tee -a "${OUTPUT_DIR}/run_quant_b${batch_s}.log"
done
echo "All configurations completed for ${MODEL_SIZE} model. Logs saved in ${OUTPUT_DIR}/"