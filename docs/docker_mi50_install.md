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

Prerequisite note: the MI50 Docker build path installs `pybind11` before `pip install --no-build-isolation -e .` so `fastsafetensors` metadata generation does not fail.

Triton package naming note: depending on source/build backend, Triton metadata may appear under `triton-gfx906`, `triton_gfx906`, `triton`, or `pytorch-triton-rocm`. The Dockerfile verification now treats these as acceptable metadata variants, while still hard-failing if `import triton` fails.

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

## Docker Compose port-collision workaround

If host port `8000` is already in use, set `VLLM_HOST_PORT` to publish the container's internal `8000` on a different host port:

```bash
VLLM_HOST_PORT=8010 docker compose up -d vllm-mi50
```

Identify what is using host port `8000` (Linux) and stop it:

```bash
sudo ss -lptn 'sport = :8000'
# then stop the reported process/container, e.g.:
# sudo kill <PID>
# docker stop <container_name_or_id>
```

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
