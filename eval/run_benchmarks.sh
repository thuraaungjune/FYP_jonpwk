#!/bin/bash

set -e

# =====================================================================
# ⚙️ GLOBAL CONFIGURATION
# =====================================================================
EVAL_SCRIPT="evaluate.py"
PROMPT_FILE="vanilla_prompt.txt"
DATASET_PATH="/home/thura/data/Jawi-OCR-data-v4-aug_ready/test_part_1"
KRAKEN_LOCAL_PATH="/home/thura/models/Jawi-OCR-Kraken-v1.mlmodel"
KRAKEN_HF_REPO="culturalheritagenus/Jawi-OCR-Kraken-v1"

# Set optimal batch size for NVIDIA A40 (46GB VRAM)
BATCH_SIZE=1

run_eval() {
    local label="$1"
    shift
    echo ""
    echo "Running Evaluation for ${label} (Batch Size: ${BATCH_SIZE}) ..."
    echo "---------------------------------------------------------------------"
    python "$EVAL_SCRIPT" "$@" --batch_size "$BATCH_SIZE"
}

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: Evaluation script '$EVAL_SCRIPT' not found!"
    exit 1
fi
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: Prompt file '$PROMPT_FILE' not found!"
    exit 1
fi

echo "====================================================================="
echo "STARTING BATCH-ACCELERATED SWEEP ON NVIDIA A40 🚀"
echo "====================================================================="

# ---------------------------------------------------------------------
# Mode 1 & 2: Custom Architecture Overrides (Jawi-OCR-Qwen series)
# # ---------------------------------------------------------------------
run_eval "Custom Jawi-OCR-Qwen-v2" \
    --model "culturalheritagenus/Jawi-OCR-Qwen-v2" \
    --prompt "$PROMPT_FILE" \
    --dataset_path "$DATASET_PATH" \
    --jawi_qwen_v2 

run_eval "Custom Jawi-OCR-Qwen-v1" \
    --model "culturalheritagenus/Jawi-OCR-Qwen-v1" \
    --prompt "$PROMPT_FILE" \
    --dataset_path "$DATASET_PATH" \
    --jawi_qwen_v1 

# ---------------------------------------------------------------------
# Mode 3: Standard Vanilla Vision-Language Model Loop
# ---------------------------------------------------------------------
# Add Qari OCR model(s) to the sweep
QARI_VLMS=(
    "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct"
)

for MODEL in "${QARI_VLMS[@]}"; do
    run_eval "Qari OCR: $MODEL" \
        --model "$MODEL" \
        --prompt "$PROMPT_FILE" \
        --dataset_path "$DATASET_PATH"

    echo "Finished benchmark run for: $MODEL"
done

STANDARD_VLMS=(
    "aisingapore/Qwen-SEA-LION-v4-4B-VL"
    "Qwen/Qwen3-VL-4B-Instruct"
    "Qwen/Qwen2-VL-2B-Instruct"
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

for MODEL in "${STANDARD_VLMS[@]}"; do
    run_eval "VLM: $MODEL" \
        --model "$MODEL" \
        --prompt "$PROMPT_FILE" \
        --dataset_path "$DATASET_PATH"

    echo "Finished benchmark run for: $MODEL"
done

# ---------------------------------------------------------------------
# Mode 4: Standalone Sequence Engine (Kraken Robust Pathing)
# ---------------------------------------------------------------------
if [ -f "$KRAKEN_LOCAL_PATH" ]; then
    KRAKEN_MODEL_SOURCE="$KRAKEN_LOCAL_PATH"
    echo "Found local Kraken model at: $KRAKEN_LOCAL_PATH"
else
    KRAKEN_MODEL_SOURCE="$KRAKEN_HF_REPO"
    echo "Local Kraken model missing at: $KRAKEN_LOCAL_PATH"
    echo "Falling back to Hugging Face repo: $KRAKEN_HF_REPO"
fi

run_eval "Native Kraken" \
    --model "$KRAKEN_MODEL_SOURCE" \
    --prompt "$PROMPT_FILE" \
    --dataset_path "$DATASET_PATH" \
    --kraken 
