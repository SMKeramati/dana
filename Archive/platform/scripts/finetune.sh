#!/bin/bash
# =============================================================================
# Dana AI Platform - Fine-tuning Script
#
# Complete fine-tuning pipeline for Dana models:
#   1. setup   - Install Python dependencies (transformers, peft, bitsandbytes)
#   2. download - Download base model weights from HuggingFace
#   3. train   - Run LoRA/QLoRA fine-tuning on your dataset
#   4. evaluate - Evaluate fine-tuned model vs base model
#   5. export  - Merge adapter with base model for deployment
#   6. serve   - Start inference-worker with the fine-tuned model
#
# Usage:
#   ./scripts/finetune.sh setup
#   ./scripts/finetune.sh download [--model Qwen/Qwen3-0.6B]
#   ./scripts/finetune.sh train --dataset data.jsonl [options]
#   ./scripts/finetune.sh evaluate --adapter /path/to/adapter --dataset data.jsonl
#   ./scripts/finetune.sh export --adapter /path/to/adapter --output /models/merged
#
# Environment variables (or set in .env):
#   MODEL_NAME        - HuggingFace model ID (default: Qwen/Qwen3-235B-A22B)
#   DRAFT_MODEL       - Draft model for speculative decoding (default: Qwen/Qwen3-0.6B)
#   MODEL_DIR         - Where to store model weights (default: /models)
#   OUTPUT_DIR        - Where to store training outputs (default: /tmp/dana-finetune)
#   HF_TOKEN          - HuggingFace token (if model is gated)
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERR ]${NC} $1"; }

DANA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load .env if exists
if [ -f "$DANA_DIR/.env" ]; then
    set +u
    source "$DANA_DIR/.env"
    set -u
fi

# Defaults
DEFAULT_MODEL="${MODEL_NAME:-Qwen/Qwen3-235B-A22B}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen3-0.6B}"
MODEL_DIR="${MODEL_DIR:-/models}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dana-finetune}"
HF_TOKEN="${HF_TOKEN:-}"

# =============================================================================
# setup - Install fine-tuning dependencies
# =============================================================================
cmd_setup() {
    echo ""
    echo "======================================================"
    echo "  Dana Fine-tuning - Environment Setup"
    echo "======================================================"
    echo ""

    # Check Python
    if ! python3 --version 2>&1 | grep -qE "3\.(11|12|13)"; then
        log_error "Python 3.11+ required. Found: $(python3 --version 2>&1)"
        exit 1
    fi
    log_ok "Python: $(python3 --version)"

    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
        log_ok "GPU: $GPU_NAME ($GPU_MEM)"
    else
        log_warn "No GPU detected. Training will be very slow on CPU."
    fi

    # Install dana-common
    log_info "Installing dana-common..."
    pip install -e "$DANA_DIR/packages/dana-common" --quiet

    # Install finetuning service
    log_info "Installing finetuning-service..."
    pip install -e "$DANA_DIR/services/finetuning-service" --quiet

    # Install training dependencies
    log_info "Installing training dependencies (transformers, peft, bitsandbytes, accelerate)..."
    pip install --quiet \
        torch \
        transformers \
        peft \
        bitsandbytes \
        accelerate \
        datasets \
        sentencepiece \
        protobuf \
        scipy

    # Install evaluation dependencies
    log_info "Installing evaluation dependencies..."
    pip install --quiet \
        rouge-score \
        nltk

    # Download NLTK data
    python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

    echo ""
    log_ok "Fine-tuning environment ready!"
    echo ""
    echo "  Next: ./scripts/finetune.sh download"
    echo ""
}

# =============================================================================
# download - Download model weights
# =============================================================================
cmd_download() {
    local model="$DEFAULT_MODEL"

    # Parse args
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model) model="$2"; shift 2 ;;
            --draft) model="$DRAFT_MODEL"; shift ;;
            *) shift ;;
        esac
    done

    echo ""
    echo "======================================================"
    echo "  Dana Fine-tuning - Model Download"
    echo "======================================================"
    echo ""
    echo "  Model:     $model"
    echo "  Directory: $MODEL_DIR"
    echo ""

    mkdir -p "$MODEL_DIR"

    # Check for huggingface-cli
    if ! command -v huggingface-cli &> /dev/null; then
        log_info "Installing huggingface-hub CLI..."
        pip install --quiet huggingface-hub[cli]
    fi

    # Login if token provided
    if [ -n "$HF_TOKEN" ]; then
        log_info "Authenticating with HuggingFace..."
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
    fi

    local model_slug=$(echo "$model" | tr '/' '_')
    local target_dir="$MODEL_DIR/$model_slug"

    if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir" 2>/dev/null)" ]; then
        log_warn "Model already exists at $target_dir"
        read -p "Re-download? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_ok "Skipping download"
            return 0
        fi
    fi

    log_info "Downloading $model..."
    log_info "This may take a while for large models (235B = ~120GB quantized)"
    echo ""

    huggingface-cli download "$model" \
        --local-dir "$target_dir" \
        --local-dir-use-symlinks False \
        --resume-download

    echo ""
    log_ok "Model downloaded to: $target_dir"
    echo ""

    # Show size
    local size=$(du -sh "$target_dir" 2>/dev/null | cut -f1)
    log_info "Total size: $size"
    echo ""
    echo "  To also download the draft model for speculative decoding:"
    echo "  ./scripts/finetune.sh download --draft"
    echo ""
}

# =============================================================================
# train - Run LoRA/QLoRA fine-tuning
# =============================================================================
cmd_train() {
    local dataset=""
    local format="jsonl"
    local model="$DEFAULT_MODEL"
    local output="$OUTPUT_DIR"
    local epochs=3
    local lr="2e-4"
    local rank=16
    local alpha=32
    local batch_size=4
    local quant="int4"
    local max_seq_length=2048
    local val_ratio=0.1

    # Parse args
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset)         dataset="$2"; shift 2 ;;
            --format)          format="$2"; shift 2 ;;
            --model)           model="$2"; shift 2 ;;
            --output)          output="$2"; shift 2 ;;
            --epochs)          epochs="$2"; shift 2 ;;
            --lr)              lr="$2"; shift 2 ;;
            --rank)            rank="$2"; shift 2 ;;
            --alpha)           alpha="$2"; shift 2 ;;
            --batch-size)      batch_size="$2"; shift 2 ;;
            --quant)           quant="$2"; shift 2 ;;
            --max-seq-length)  max_seq_length="$2"; shift 2 ;;
            --val-ratio)       val_ratio="$2"; shift 2 ;;
            *) log_error "Unknown option: $1"; exit 1 ;;
        esac
    done

    if [ -z "$dataset" ]; then
        log_error "Dataset required: --dataset path/to/data.jsonl"
        echo ""
        echo "  Usage:"
        echo "    ./scripts/finetune.sh train --dataset data.jsonl [options]"
        echo ""
        echo "  Options:"
        echo "    --dataset PATH         Path to training data (required)"
        echo "    --format FORMAT        Dataset format: jsonl, alpaca, sharegpt (default: jsonl)"
        echo "    --model MODEL          Base model ID (default: $DEFAULT_MODEL)"
        echo "    --output DIR           Output directory (default: $OUTPUT_DIR)"
        echo "    --epochs N             Number of epochs (default: 3)"
        echo "    --lr RATE              Learning rate (default: 2e-4)"
        echo "    --rank N               LoRA rank (default: 16)"
        echo "    --alpha N              LoRA alpha (default: 32)"
        echo "    --batch-size N         Batch size (default: 4)"
        echo "    --quant MODE           Quantization: none, int8, int4 (default: int4)"
        echo "    --max-seq-length N     Max sequence length (default: 2048)"
        echo "    --val-ratio FLOAT      Validation split ratio (default: 0.1)"
        echo ""
        echo "  Dataset format (JSONL):"
        echo '    {"instruction": "...", "output": "...", "input": "", "system": ""}'
        echo ""
        exit 1
    fi

    if [ ! -f "$dataset" ]; then
        log_error "Dataset file not found: $dataset"
        exit 1
    fi

    echo ""
    echo "======================================================"
    echo "  Dana Fine-tuning - LoRA Training"
    echo "======================================================"
    echo ""
    echo "  Dataset:     $dataset"
    echo "  Format:      $format"
    echo "  Base model:  $model"
    echo "  Output:      $output"
    echo "  Epochs:      $epochs"
    echo "  LR:          $lr"
    echo "  LoRA:        rank=$rank, alpha=$alpha"
    echo "  Quantization: $quant"
    echo "  Batch size:  $batch_size"
    echo "  Max seq len: $max_seq_length"
    echo ""

    # Check GPU memory
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        log_info "GPU memory: ${GPU_MEM_MB}MB"

        if [ "$quant" = "none" ] && [ "${GPU_MEM_MB:-0}" -lt 80000 ]; then
            log_warn "Full precision training needs 80GB+ VRAM. Consider --quant int4"
        fi
    fi

    # Dataset line count
    LINES=$(wc -l < "$dataset")
    log_info "Dataset: $LINES lines"

    mkdir -p "$output"

    # Run training
    log_info "Starting training..."
    echo ""

    python3 "$DANA_DIR/scripts/finetune/train_lora.py" \
        --dataset "$dataset" \
        --format "$format" \
        --base-model "$model" \
        --output "$output" \
        --epochs "$epochs" \
        --lr "$lr" \
        --rank "$rank" \
        --alpha "$alpha" \
        --batch-size "$batch_size" \
        --quant "$quant" \
        --max-seq-length "$max_seq_length" \
        --val-ratio "$val_ratio"

    echo ""
    log_ok "Training complete!"
    echo ""
    echo "  Adapter saved to: $output/ft-*/adapter"
    echo ""
    echo "  Next steps:"
    echo "    1. Evaluate: ./scripts/finetune.sh evaluate --adapter $output/ft-*/adapter --dataset $dataset"
    echo "    2. Export:   ./scripts/finetune.sh export --adapter $output/ft-*/adapter --output /models/merged"
    echo ""
}

# =============================================================================
# evaluate - Evaluate fine-tuned model
# =============================================================================
cmd_evaluate() {
    local dataset=""
    local model="$DEFAULT_MODEL"
    local adapter=""
    local max_samples=50

    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset)      dataset="$2"; shift 2 ;;
            --model)        model="$2"; shift 2 ;;
            --adapter)      adapter="$2"; shift 2 ;;
            --max-samples)  max_samples="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    if [ -z "$dataset" ]; then
        log_error "Dataset required: --dataset path/to/data.jsonl"
        exit 1
    fi

    echo ""
    echo "======================================================"
    echo "  Dana Fine-tuning - Model Evaluation"
    echo "======================================================"
    echo ""
    echo "  Dataset:     $dataset"
    echo "  Base model:  $model"
    echo "  Adapter:     ${adapter:-none (base model only)}"
    echo "  Max samples: $max_samples"
    echo ""

    python3 "$DANA_DIR/scripts/finetune/evaluate.py" \
        --dataset "$dataset" \
        --base-model "$model" \
        ${adapter:+--adapter "$adapter"} \
        --max-samples "$max_samples"

    echo ""
}

# =============================================================================
# export - Merge LoRA adapter with base model
# =============================================================================
cmd_export() {
    local adapter=""
    local model="$DEFAULT_MODEL"
    local output=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --adapter) adapter="$2"; shift 2 ;;
            --model)   model="$2"; shift 2 ;;
            --output)  output="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    if [ -z "$adapter" ] || [ -z "$output" ]; then
        log_error "Usage: ./scripts/finetune.sh export --adapter PATH --output PATH"
        exit 1
    fi

    echo ""
    echo "======================================================"
    echo "  Dana Fine-tuning - Merge & Export"
    echo "======================================================"
    echo ""
    echo "  Base model: $model"
    echo "  Adapter:    $adapter"
    echo "  Output:     $output"
    echo ""

    mkdir -p "$output"

    python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

print('Loading base model...')
base = AutoModelForCausalLM.from_pretrained('$model', device_map='auto', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('$model', trust_remote_code=True)

print('Loading adapter...')
model = PeftModel.from_pretrained(base, '$adapter')

print('Merging weights...')
merged = model.merge_and_unload()

print('Saving merged model...')
merged.save_pretrained('$output')
tokenizer.save_pretrained('$output')

print('Done!')
"

    echo ""
    log_ok "Merged model saved to: $output"
    echo ""
    echo "  To use with inference-worker, update .env:"
    echo "    MODEL_PATH=$output"
    echo ""
}

# =============================================================================
# Main dispatch
# =============================================================================
show_help() {
    echo ""
    echo "Dana AI Platform - Fine-tuning Script"
    echo ""
    echo "Usage: ./scripts/finetune.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  setup       Install Python dependencies for fine-tuning"
    echo "  download    Download model weights from HuggingFace"
    echo "  train       Run LoRA/QLoRA fine-tuning on a dataset"
    echo "  evaluate    Evaluate model quality (BLEU, ROUGE-L, Persian metrics)"
    echo "  export      Merge LoRA adapter with base model for deployment"
    echo ""
    echo "Quick start:"
    echo "  1. ./scripts/finetune.sh setup"
    echo "  2. ./scripts/finetune.sh download --model Qwen/Qwen3-0.6B"
    echo "  3. ./scripts/finetune.sh train --dataset my_data.jsonl --model Qwen/Qwen3-0.6B"
    echo "  4. ./scripts/finetune.sh evaluate --adapter /tmp/dana-finetune/ft-*/adapter --dataset my_data.jsonl"
    echo "  5. ./scripts/finetune.sh export --adapter /tmp/dana-finetune/ft-*/adapter --output /models/merged"
    echo ""
    echo "For full 235B model fine-tuning (requires multi-GPU):"
    echo "  1. ./scripts/finetune.sh download"
    echo "  2. ./scripts/finetune.sh train --dataset data.jsonl --quant int4 --rank 16"
    echo ""
}

COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    setup)    cmd_setup "$@" ;;
    download) cmd_download "$@" ;;
    train)    cmd_train "$@" ;;
    evaluate) cmd_evaluate "$@" ;;
    export)   cmd_export "$@" ;;
    help|--help|-h) show_help ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
