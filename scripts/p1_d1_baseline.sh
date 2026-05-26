#!/usr/bin/env bash
# Phase 1 Day 1 — llama.cpp Metal baseline for Qwen3.5-35B-A3B Q4_K_M
# Gates: coherent output, >= 8 tok/s, peak RAM < 14 GB

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=$(ls models/qwen35-35b-a3b/*Q4_K_M*.gguf 2>/dev/null | head -1)
[ -z "$MODEL" ] && { echo "ERROR: no Q4_K_M gguf under models/qwen35-35b-a3b/"; exit 1; }

OUT=reports/llamacpp_baseline.txt
PROMPT="Explain mixture-of-experts in one paragraph:"
N=200

mkdir -p reports

{
  echo "=== Phase 1 Day 1 — llama.cpp Qwen3.5-35B-A3B Q4_K_M baseline ==="
  echo "date:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host:  $(uname -mrs)"
  echo "model: $MODEL ($(du -h "$MODEL" | cut -f1))"
  echo "prompt: $PROMPT"
  echo "n:      $N"
  echo ""
  echo "--- vm_stat (before) ---"
  vm_stat | head -8
  echo ""
  echo "================ LLAMA-CLI OUTPUT ================"
} > "$OUT"

/usr/bin/time -lp llama-cli -m "$MODEL" -p "$PROMPT" -n $N --mmap -no-cnv 2>&1 | tee -a "$OUT"

{
  echo "================ END LLAMA-CLI ================"
  echo ""
  echo "--- vm_stat (after) ---"
  vm_stat | head -8
} >> "$OUT"

echo ""
echo "==== Gate check ===="
grep -E "(tokens per second|tok/s|eval time|^Maximum resident set size)" "$OUT" | head -10 || echo "(no perf stats — check $OUT)"
echo ""
echo "Full log: $OUT"
