# MI50 Remote Validation Checklist

Target host: Remote server with AMD MI50 (gfx906) and ROCm runtime.
Purpose: Validate Dockerized runtime and minimal serving sanity for Qwen 3.5 after local non-GPU checks.

## Preconditions
- Access to MI50 server with Docker and ROCm device runtime configured.
- Repository available on server at a known path.
- Model access/auth configured for selected Qwen 3.5 checkpoint.

## 1) Docker Build/Run Verification (MI50)

### Build image
```bash
docker build -f docker/Dockerfile.rocm.mi50 -t vllm-rocm-mi50:local .
```
Expected:
- Build completes without dependency resolution errors.

### Start container with ROCm devices
```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size=16g \
  -p 8000:8000 \
  vllm-rocm-mi50:local bash
```
Expected:
- Container starts and shell is available.

## 2) ROCm Device Visibility Checks

Inside container:
```bash
rocminfo | head -n 60
```
Expected:
- MI50/gfx906 agent visible.

Optional detailed check:
```bash
rocm-smi
```
Expected:
- GPU listed; health/clock/power metrics readable.

## 3) triton-gfx906 Detection Checks

Inside container (Python smoke):
```bash
python - <<'PY'
import importlib
mods = [
  "vllm.triton_utils",
  "vllm.triton_utils.importing",
]
for m in mods:
    try:
        importlib.import_module(m)
        print("OK", m)
    except Exception as e:
        print("FAIL", m, repr(e))
PY
```
Expected:
- Imports succeed.
- No hard failure caused by Triton/gfx906 compatibility path.

Optional env sanity (if used by branch logic):
```bash
env | grep -Ei 'TRITON|ROCM|HIP|HSA|VLLM'
```
Expected:
- Required ROCm/Triton env vars present for deployment policy.

## 4) Qwen 3.5 Serve Smoke Test

Inside container, launch server (example):
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
```
Expected:
- Server starts and reports ready state.
- No immediate model-registry/config errors for Qwen 3.5.

From another shell, run minimal completion request:
```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-7B-Instruct","prompt":"Say hello in five words.","max_tokens":16}'
```
Expected:
- HTTP 200 response with non-empty completion text.

## 5) Minimal Throughput Sanity

Run 3 short requests and capture latency/token stats from logs or client timing.

Example quick loop:
```bash
for i in 1 2 3; do
  time curl -s http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3.5-7B-Instruct","prompt":"Count to ten.","max_tokens":32}' >/dev/null
done
```
Expected:
- All requests succeed.
- No repeated runtime faults/OOM.
- Throughput/latency in a stable range for MI50 baseline.

## Result Recording Template

- Docker build: PASS/FAIL + logs
- Docker run with ROCm devices: PASS/FAIL + logs
- ROCm visibility (`rocminfo`/`rocm-smi`): PASS/FAIL + logs
- Triton gfx906 detection/imports: PASS/FAIL + logs
- Qwen 3.5 serve startup: PASS/FAIL + logs
- API smoke request: PASS/FAIL + sample response
- 3-request throughput sanity: PASS/FAIL + timing snapshot
- Blockers and remediation notes
