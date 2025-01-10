#!/bin/bash

# Gaudi Setup Script for LLaMA Vision Benchmarking
#
# This script sets up the Habana Gaudi environment for running LLaMA Vision benchmarks.
# After setup, you must:
#   1. Enter the container: docker exec -it gaudi_setup bash
#   2. Change to workspace: cd /workspace
#   3. Then run the benchmark scripts:
#      - For text generation:
#        ./scripts/text_gen_bf16.sh       # BF16 text generation
#        ./scripts/text_gen_quantize.sh   # Quantized text generation
#      - For visual question answering:
#        ./scripts/vqa_bf16_runner.sh     # BF16 VQA benchmarks
#
# Requirements:
#   - Docker with Habana runtime
#   - Valid Hugging Face token (HUGGING_FACE_HUB_TOKEN env var)

set -e
set -u
set -o pipefail

CONTAINER_NAME="gaudi_setup"
# Default container for 1.18.0
DEFAULT_DOCKER_IMAGE="vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest"
# Container for 1.19.0
NEW_DOCKER_IMAGE="vault.habana.ai/gaudi-docker/1.19.0/ubuntu24.04/habanalabs/pytorch-installer-2.5.1:latest"

MOUNT_PATH="/tmp:/mnt"
HOME_MOUNT="$HOME:/root"
WORKSPACE_DIR=$(pwd)

GAUDI_VERSION=$(hl-smi -v 2>/dev/null | grep -oP 'hl-\K\d+\.\d+\.\d+' || echo "1.18.0")
echo "Detected Gaudi firmware version: ${GAUDI_VERSION}"
if [[ "${GAUDI_VERSION}" == "1.19.0" ]]; then
    DOCKER_IMAGE="${NEW_DOCKER_IMAGE}"
    echo "Using PyTorch 2.5.1 container for Gaudi ${GAUDI_VERSION}"
else
    DOCKER_IMAGE="${DEFAULT_DOCKER_IMAGE}"
    echo "Using default PyTorch 2.4.0 container for Gaudi ${GAUDI_VERSION}"
fi

check_container_state() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '$CONTAINER_NAME' already exists. Removing it..."
        docker rm -f $CONTAINER_NAME
    fi
    return 0
}

echo "Running pre-flight checks..."
if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set"
    exit 1
fi

echo "Checking container state..."
check_container_state || exit 1

echo "Launching container and running setup..."
docker run -d --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} \
    --cap-add=sys_nice \
    --net=host \
    --ipc=host \
    --name=$CONTAINER_NAME \
    --volume $MOUNT_PATH \
    -v $HOME_MOUNT \
    -v $WORKSPACE_DIR:/workspace \
    $DOCKER_IMAGE \
    /bin/bash -c "cd /workspace && \
    echo 'Installing optimum-habana...' && \
    echo 'Setting up optimum-habana repository...' && \
    if [ -d 'optimum-habana' ]; then \
        (cd optimum-habana && git fetch && git checkout v1.15.0) || exit 1; \
    else \
        git clone https://github.com/huggingface/optimum-habana.git && \
        (cd optimum-habana && git checkout v1.15.0) || exit 1; \
    fi && \
    echo 'Installing additional requirements...' && \
    pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0 && \
    pip install accelerate==0.33.0 && \
    pip install huggingface-hub==0.27.1 && \
    pip install tokenizers==0.20.3 && \
    pip install transformers==4.45.2 && \
    pip install sentence-transformers==3.2.1 && \
    pip install -r optimum-habana/examples/text-generation/requirements.txt && \
    pip install -r optimum-habana/examples/text-generation/requirements_lm_eval.txt && \
    pip install --upgrade optimum[habana] && \
    echo 'Setting up HuggingFace cache...' && \
    export HF_HOME=/mnt/huggingface && \
    mkdir -p \$HF_HOME && \
    echo 'Verifying installations...' && \
    echo 'Setup completed successfully!' && \
    tail -f /dev/null"

sleep 5

echo "Waiting for setup to complete..."
echo "To view setup progress in real-time: docker logs -f $CONTAINER_NAME"
timeout=300  # 5 minutes timeout
start_time=$(date +%s)
while true; do
    if docker logs $CONTAINER_NAME 2>&1 | grep -q "Setup completed successfully!"; then
        break
    fi
    if docker logs $CONTAINER_NAME 2>&1 | grep -q "Error\|error\|Failed\|failed"; then
        echo "Error detected in setup:"
        docker logs $CONTAINER_NAME
        docker rm -f $CONTAINER_NAME 2>/dev/null || true
        exit 1
    fi
    current_time=$(date +%s)
    if [ $((current_time - start_time)) -gt $timeout ]; then
        echo "Setup timed out after ${timeout} seconds"
        docker logs $CONTAINER_NAME
        docker rm -f $CONTAINER_NAME 2>/dev/null || true
        exit 1
    fi
    sleep 5
done


if ! docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "Error: Container failed to start or setup failed"
    docker logs $CONTAINER_NAME
    docker rm -f $CONTAINER_NAME 2>/dev/null || true
    exit 1
fi

echo "Setup completed successfully!"
echo
echo "To run benchmarks:"
echo "1. Enter the container:    docker exec -it $CONTAINER_NAME bash"
echo "2. Change to workspace:    cd /workspace"
echo "3. Run benchmark scripts:"
echo "   - Text Generation:"
echo "     ./scripts/text_gen_bf16.sh       # BF16 text generation"
echo "     ./scripts/text_gen_quantize.sh   # Quantized text generation"
echo "   - Visual Question Answering:"
echo "     ./scripts/vqa_bf16_runner.sh     # BF16 VQA benchmarks"
echo
