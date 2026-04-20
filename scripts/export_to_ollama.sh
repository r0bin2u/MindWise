#!/usr/bin/env bash
# Merge LoRA -> fp16 HF -> GGUF q8_0 -> register with Ollama.
#
# Usage: bash scripts/export_to_ollama.sh [ollama-model-name]
#
# Requires: llama.cpp convert_hf_to_gguf.py + ollama binary on PATH.

set -euo pipefail

MODEL_NAME="${1:-qwen2.5-7b-psychqa}"
ADAPTER="${ADAPTER:-checkpoints/qwen25-7b-psychqa}"
MERGED="${MERGED:-checkpoints/qwen25-7b-psychqa-merged}"
GGUF="${GGUF:-checkpoints/qwen25-7b-psychqa.gguf}"
LLAMA_CPP="${LLAMA_CPP:-/home/xie/Desktop/llama.cpp}"

echo "[1/3] merging LoRA adapter -> fp16 HF model"
python3 -m scripts.merge_lora --adapter "$ADAPTER" --output "$MERGED"

echo "[2/3] converting merged HF model -> GGUF (q8_0)"
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED" \
    --outfile "$GGUF" \
    --outtype q8_0

echo "[3/3] registering with Ollama: $MODEL_NAME"
# Modelfile lives next to the GGUF; update the FROM path to the actual gguf name
cat > checkpoints/Modelfile.auto <<EOF
FROM ./$(basename "$GGUF")

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM "你是校园心理咨询助手 MindWise，识别用户情绪并以温和、共情的方式回应。"

PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
EOF

(cd checkpoints && ollama create "$MODEL_NAME" -f Modelfile.auto)
echo "done. test with:  ollama run $MODEL_NAME"
