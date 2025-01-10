# LLaMA Vision Benchmarks

Benchmarking scripts for LLaMA Vision models on Habana Gaudi.

## Supported Models
- LLaMA Vision 3.2 11B (runs on 1 card)
- LLaMA Vision 3.2 90B (runs on 8 cards)

## Setup

1. Set your Hugging Face token:
```bash
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

2. Run setup script:
```bash
./scripts/setup_gaudi.sh
```

## Running Benchmarks

After setup completes, follow these steps:

1. Enter the container:
```bash
docker exec -it gaudi_setup bash
```

2. Change to workspace:
```bash
cd /workspace
```

3. (Optional) Monitor memory usage in a separate terminal:
```bash
./scripts/monitor_memory.sh  # Shows real-time memory stats for all cards
```

4. Run benchmarks:

### Text Generation
- BF16 mode:
```bash
# 11B model (1 card)
./scripts/text_gen_bf16.sh --model-size 11b  # Input/Output/Batch: 128/128/1024, 2048/2048/64, etc.

# 90B model (8 cards)
./scripts/text_gen_bf16.sh --model-size 90b  # Input/Output/Batch: 128/128/256, 2048/2048/64, etc.
```

- Quantized mode:
```bash
# 11B model (1 card)
./scripts/text_gen_quantize.sh --model-size 11b  # Input/Output/Batch: 128/128/1024, 2048/2048/64, etc.

# 90B model (8 cards)
./scripts/text_gen_quantize.sh --model-size 90b  # Input/Output/Batch: 128/128/256, 2048/2048/64, etc.
```

### Visual Question Answering (VQA)
```bash
# 11B model (1 card)
./scripts/vqa_bf16_runner.sh --model-size 11b  # Output Tokens=128/Batch=1024, Output=2048/Batch=96, etc.

# 90B model (8 cards)
./scripts/vqa_bf16_runner.sh --model-size 90b  # Output Tokens=128/Batch=256, Output=2048/Batch=64, etc.
