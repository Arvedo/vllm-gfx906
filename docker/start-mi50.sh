#!/bin/sh
set -eu

if [ -z "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]; then
  VLLM_GPU_MEMORY_UTILIZATION="$(python3 - <<'PY'
import math
import sys

try:
    import torch
except Exception as exc:
    print("0.85")
    print(f"[mi50-preflight] Unable to import torch for VRAM preflight: {exc}", file=sys.stderr)
    raise SystemExit(0)

if not torch.cuda.is_available():
    print("0.85")
    print("[mi50-preflight] ROCm device not visible during preflight; using fallback VLLM_GPU_MEMORY_UTILIZATION=0.85", file=sys.stderr)
    raise SystemExit(0)

ratios = []
for idx in range(torch.cuda.device_count()):
    free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
    free_gib = free_bytes / 1024**3
    total_gib = total_bytes / 1024**3
    ratio = free_bytes / total_bytes if total_bytes else 0.0
    ratios.append(ratio)
    print(
        f"[mi50-preflight] gpu{idx}: free={free_gib:.2f} GiB / total={total_gib:.2f} GiB",
        file=sys.stderr,
    )

min_ratio = min(ratios) if ratios else 0.85
target = min(0.85, max(0.05, math.floor((min_ratio - 0.03) * 100) / 100))

if min_ratio < 0.35:
    print(
        "[mi50-preflight] Visible GPUs are already heavily occupied. If startup still fails, use emptier GPUs, lower tensor parallelism, or switch to a smaller model.",
        file=sys.stderr,
    )

print(f"[mi50-preflight] Auto-selected VLLM_GPU_MEMORY_UTILIZATION={target:.2f}", file=sys.stderr)
print(f"{target:.2f}")
PY
)"
  export VLLM_GPU_MEMORY_UTILIZATION
fi

echo "[mi50-preflight] Starting model ${VLLM_MODEL:-QuantTrio/Qwen3.5-35B-A3B-AWQ} with TP=${VLLM_TENSOR_PARALLEL_SIZE:-2} and VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"

if [ -n "${VLLM_LIMIT_MM_PER_PROMPT:-}" ]; then
    LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT}"
else
    LIMIT_MM_PER_PROMPT='{"image":1,"video":0}'
fi

# Avoid importing the source checkout at /workspace/vllm. We want the installed
# package from site-packages, which contains the compiled extensions.
cd /opt/vllm-runtime

exec python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model "${VLLM_MODEL:-QuantTrio/Qwen3.5-35B-A3B-AWQ}" \
  --dtype "${VLLM_DTYPE:-float16}" \
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-2}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
  --reasoning-parser "${VLLM_REASONING_PARSER:-qwen3}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${VLLM_TOOL_CALL_PARSER:-qwen3_xml}" \
    --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
  ${VLLM_EXTRA_ARGS:-}
