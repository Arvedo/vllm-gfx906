# Ubuntu MI50 (gfx906) Docker install path

This document provides a production-oriented container path for this fork on Ubuntu servers with AMD Instinct MI50 (gfx906).

## Scope

- Dockerfile: `docker/Dockerfile.rocm.mi50`
- GPU target: AMD gfx906 (MI50/MI60/Radeon VII family)
- Triton path: `nlzy/triton-gfx906`
- Transformers line: `>=5.2,<6`

## Build

From the repository root:

```bash
docker build -f docker/Dockerfile.rocm.mi50 -t vllm-gfx906:mi50 .
```

Prerequisite/reliability note: the MI50 Docker build path installs `pybind11` (and build helpers `ninja`/`cmake`) before local vLLM install, and uses non-editable `pip install --no-build-isolation .` with `VLLM_TARGET_DEVICE=rocm` to avoid `RuntimeError: Unknown runtime environment` in legacy `docker-compose` builds.

Triton package naming note: depending on source/build backend, Triton metadata may appear under `triton-gfx906`, `triton_gfx906`, `triton`, or `pytorch-triton-rocm`. The Dockerfile verification now treats these as acceptable metadata variants, while still hard-failing if `import triton` fails.

Troubleshooting note: missing `.git` in Docker build context is expected; the image sets `SETUPTOOLS_SCM_PRETEND_VERSION` and `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM` so local vLLM install remains deterministic.

Optional version/index overrides:

```bash
docker build \
  -f docker/Dockerfile.rocm.mi50 \
  --build-arg ROCM_PYTORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.3 \
  --build-arg TORCH_VERSION=2.9.0 \
  --build-arg TORCHVISION_VERSION=0.24.0 \
  --build-arg TORCHAUDIO_VERSION=2.9.0 \
  -t vllm-gfx906:mi50 .
```

## Run (Ubuntu server with ROCm devices)

Use `/dev/kfd`, `/dev/dri`, video/render groups, larger shared memory, and permissive seccomp/capability settings commonly required by ROCm userspace in containers:

```bash
docker run --rm -it \
  --network host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --ipc=host \
  --shm-size=16g \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HSA_OVERRIDE_GFX_VERSION=9.0.6 \
  -e HIP_VISIBLE_DEVICES=0 \
  vllm-gfx906:mi50
```

Notes:
- `HSA_OVERRIDE_GFX_VERSION=9.0.6` can help on some host ROCm stacks for gfx906 reporting/compat behavior.
- Adjust `HIP_VISIBLE_DEVICES` for multi-GPU hosts.
- Container runtime must expose ROCm character devices from host driver installation.

## Quick verification

Inside the running container:

1) ROCm visibility:

```bash
rocminfo | grep -E "Name:|gfx906" || true
```

2) Python package/runtime sanity:

```bash
python3 - <<'PY'
import importlib.metadata as m
import torch
import triton
import vllm

print('torch', torch.__version__)
print('triton.__version__', getattr(triton, '__version__', 'unknown'))
print('transformers', m.version('transformers'))
for name in ('triton-gfx906', 'triton_gfx906', 'triton', 'pytorch-triton-rocm'):
    try:
        print(name, m.version(name))
    except m.PackageNotFoundError:
        pass
print('vllm ok')
PY
```

3) Minimal vLLM import + platform check:

```bash
python3 -c "import torch; print('cuda/rocm visible:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

4) Optional server startup smoke test (replace model path):

```bash
vllm serve /models/<your-model> --dtype float16 --port 8000
```

## Docker Compose single-command start (legacy v1 friendly)

Run from repository root:

```bash
docker-compose up -d --build vllm-mi50
```

This works without exporting variables and uses MI50 defaults from `docker-compose.yml`:
- model: `QuantTrio/Qwen3.5-35B-A3B-AWQ`
- host port: `8010` (container port remains `8000`)
- `HIP_VISIBLE_DEVICES=4,5`
- tensor parallel size: `2`
- `HSA_OVERRIDE_GFX_VERSION=9.0.6`
- `VLLM_GPU_MEMORY_UTILIZATION` is auto-derived from currently free VRAM when unset, capped at `0.85`

If the first `docker-compose up -d --build vllm-mi50` appears stuck, it is usually still building (legacy v1 builder can be very slow on first build). After the first successful image build, start without rebuild:

```bash
docker-compose up -d vllm-mi50
```

One-time cleanup for older `docker-compose` v1 `ContainerConfig` recreate bug:

```bash
docker-compose down --remove-orphans && docker-compose rm -f vllm-mi50
```

Runtime note: the Compose service starts vLLM via `python3`/`vllm` (not `python`).

After this fix, restart the service with:

```bash
docker compose up -d --force-recreate vllm-mi50
```

If the selected GPUs are already partially occupied, the compose entrypoint now prints a preflight summary of free VRAM per visible GPU and automatically lowers `VLLM_GPU_MEMORY_UTILIZATION` before launching vLLM. This prevents the immediate startup failure where desired utilization is higher than currently free memory. It does **not** make an oversized model fit on busy GPUs; if the model still fails later, switch to emptier GPUs, lower `VLLM_TENSOR_PARALLEL_SIZE`, or choose a smaller checkpoint.

## Runtime defaults set in Dockerfile

`docker/Dockerfile.rocm.mi50` sets conservative defaults for gfx906 safety:

- `PYTORCH_ROCM_ARCH=gfx906`
- `VLLM_TARGET_DEVICE=rocm`
- `VLLM_TRITON_SKIP_ACTIVE_DRIVER_CHECK=1`
- `VLLM_ATTENTION_BACKEND=ROCM_ATTN`
- `VLLM_ROCM_USE_AITER=0`
- `VLLM_ROCM_CUSTOM_PAGED_ATTN=0`
- `VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1`
- `HIP_FORCE_DEV_KERNARG=1`

These can be overridden at `docker run` time if needed.

## Assumptions and limits

- This path assumes host ROCm kernel/user-space compatibility for MI50 and functional `/dev/kfd` + `/dev/dri` exposure.
- The image build itself does not validate model-specific runtime kernels or throughput.
- Server-level validation (real model load and sustained inference) must be executed on the target MI50 Ubuntu host.
