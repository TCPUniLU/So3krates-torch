# Docker Usage for So3krates-torch

Docker images give you a reproducible environment with all dependencies pre-installed,
including GPU support via the NVIDIA CUDA base image.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (GPU builds only)

## Building the Image

```bash
# GPU build (default) — requires NVIDIA Container Toolkit at runtime
docker build -t torchkrates .

# CPU-only build — no GPU required
docker build --build-arg CUDA=false -t torchkrates-cpu .
```

## CLI Tools

All eight CLI entry points are available inside the container:

| Command | Purpose |
|---------|---------|
| `torchkrates-train` | Train a model from a YAML config |
| `torchkrates-eval` | Run inference on a dataset |
| `torchkrates-metric` | Compute error metrics |
| `torchkrates-preprocess` | Convert XYZ/extxyz → HDF5 |
| `torchkrates-merge` | Merge HDF5 files |
| `torchkrates-create-lammps-model` | Export model for LAMMPS |
| `torchkrates-jax2torch` | Convert JAX → PyTorch weights |
| `torchkrates-torch2jax` | Convert PyTorch → JAX weights |

## Common Workflows

### Inference with the bundled SO3LR model

```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  torchkrates \
  torchkrates-eval \
    --model_path so3lr \
    --data_path /workspace/data/test.xyz \
    --output_file /workspace/data/out.h5
```

### Training

```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  torchkrates \
  torchkrates-train --config /workspace/data/config.yaml
```

### Multi-GPU training (torchrun)

```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  torchkrates \
  torchrun --nproc_per_node=2 \
    -m so3krates_torch.cli.run_train --config /workspace/data/config.yaml
```

### Preprocess XYZ → HDF5

```bash
docker run \
  -v $(pwd)/data:/workspace/data \
  torchkrates \
  torchkrates-preprocess \
    --input /workspace/data/train.xyz \
    --output /workspace/data/train.h5 \
    --r-max 5.0
```

### Merge HDF5 files

```bash
docker run \
  -v $(pwd)/data:/workspace/data \
  torchkrates \
  torchkrates-merge \
    --inputs /workspace/data/a.h5 /workspace/data/b.h5 \
    --output /workspace/data/merged.h5
```

### CPU-only inference

```bash
docker build --build-arg CUDA=false -t torchkrates-cpu .

docker run \
  -v $(pwd)/data:/workspace/data \
  torchkrates-cpu \
  torchkrates-eval \
    --model_path so3lr \
    --data_path /workspace/data/test.xyz \
    --output_file /workspace/data/out.h5
```

## Using Docker Compose

`docker-compose.yml` provides pre-configured services for each workflow.
Volume mounts default to `./data`, `./checkpoints`, `./logs`, `./output`, `./models`
relative to the project root.

```bash
# Training (pass args after the service name)
docker compose run train --config /workspace/data/config.yaml

# Preprocessing
docker compose run preprocess \
  --input /workspace/data/train.xyz --output /workspace/data/train.h5

# Metrics
docker compose run metric \
  --models /workspace/models --data /workspace/data/test.h5

# Interactive development shell (with GPU)
docker compose run dev
```

## Volume Layout

| Host path | Container path | Used by |
|-----------|---------------|---------|
| `./data` | `/workspace/data` | All services |
| `./checkpoints` | `/workspace/checkpoints` | `train` |
| `./logs` | `/workspace/logs` | `train` |
| `./output` | `/workspace/output` | `eval` |
| `./models` | `/workspace/models` | `metric` |

Data and configs are **never baked into the image** — always pass them via volume mounts.

## Verifying the Image

```bash
# Check the package imports correctly
docker run torchkrates python -c "import so3krates_torch; print('OK')"

# Check CUDA is visible (GPU build)
docker run --gpus all torchkrates \
  python -c "import torch; print(torch.cuda.is_available())"

# Check all CLI tools are available
docker run torchkrates torchkrates-train --help
docker run torchkrates torchkrates-eval --help
```
