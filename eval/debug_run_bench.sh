#!/bin/bash

set -e

# =====================================================================
# ⚙️ GLOBAL PATH CONFIGURATION
# =====================================================================
EVAL_SCRIPT="evaluate.py"
PROMPT_FILE="vanilla_with_jawi.txt"
DATASET_PATH="/home/thura/data/Jawi-OCR-data-v4-aug_ready/test_part_1"
KRAKEN_LOCAL_PATH="/home/thura/models/Jawi-OCR-Kraken-v1.mlmodel"
KRAKEN_HF_REPO="culturalheritagenus/Jawi-OCR-Kraken-v1"
SAMPLE_SIZE=10

run_eval() {
    local label="$1"
    shift
    echo ""
    echo "Running Evaluation for ${label} ..."
    echo "-----------------------------------------------------"
    CUDA_VISIBLE_DEVICES=0,1 python "$EVAL_SCRIPT" "$@"
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
echo "STARTING AUTOMATED CLEAN BENCHMARKS (Sample Size: $SAMPLE_SIZE) 🚀"
echo "====================================================================="

# =====================================================================
# ⚙️ CONDA INITIALIZATION
# =====================================================================
CONDA_PROFILE="/home/thura/data/miniconda3/etc/profile.d/conda.sh"

if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
else
    echo "Error: Conda profile script not found at $CONDA_PROFILE"
    exit 1
fi

conda activate ocr_eval

# ---------------------------------------------------------------------
# Mode 1 & 2: Custom Architecture Overrides (Jawi-OCR-Qwen series)
# ---------------------------------------------------------------------
run_eval "Custom Jawi-OCR-Qwen-v2" \
    --model "culturalheritagenus/Jawi-OCR-Qwen-v2" \
    --prompt "$PROMPT_FILE" \
    --dataset_path "$DATASET_PATH" \
    --clean \
    --sample "$SAMPLE_SIZE" \
    --jawi_qwen_v2

run_eval "Custom Jawi-OCR-Qwen-v1" \
    --model "culturalheritagenus/Jawi-OCR-Qwen-v1" \
    --prompt "$PROMPT_FILE" \
    --dataset_path "$DATASET_PATH" \
    --clean \
    --sample "$SAMPLE_SIZE" \
    --jawi_qwen_v1

# ---------------------------------------------------------------------
# Mode 3: Standard Vanilla Vision-Language Model Loop
# ---------------------------------------------------------------------
STANDARD_VLMS=(
    "aisingapore/Gemma-SEA-LION-v4-4B-VL"
    "google/gemma-3-4b-it"
    "Qwen/Qwen2-VL-2B-Instruct"
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

for MODEL in "${STANDARD_VLMS[@]}"; do
    run_eval "VLM: $MODEL" \
        --model "$MODEL" \
        --prompt "$PROMPT_FILE" \
        --dataset_path "$DATASET_PATH" \
        --clean \
        --sample "$SAMPLE_SIZE"
    echo "Finished: $MODEL"
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
    --clean \
    --sample "$SAMPLE_SIZE" \
    --kraken

# conda deactivate

# conda activate deepseek_ocr

# # ---------------------------------------------------------------------
# # Mode 5: Specialized DeepSeek OCR
# # ---------------------------------------------------------------------
# run_eval "Specialized DeepSeek-OCR" \
#     --model "deepseek-ai/DeepSeek-OCR" \
#     --prompt "$PROMPT_FILE" \
#     --dataset_path "$DATASET_PATH" \
#     --clean \
#     --sample "$SAMPLE_SIZE" \
#     --deepseek

echo ""
echo "====================================================================="
echo "ALL BENCHMARKS EXECUTED"
echo "====================================================================="
ls -lh *.csv *.json 2>/dev/null || echo "No summary files generated."
