# Docker Cheatsheet for So3krates-torch

## Build

```bash
# GPU (default)
docker build -t torchkrates .

# CPU-only
docker build --build-arg CUDA=false -t torchkrates-cpu .
```

---

## Preprocess XYZ → HDF5

```bash
sudo docker run \
  -v $(pwd)/data:/workspace/data \
  torchkrates \
  torchkrates-preprocess \
    --input /workspace/data/train.xyz \
    --output /workspace/data/train.h5 \
    --mode raw
```

## Train

```bash
sudo docker run \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/model:/workspace/model \
  torchkrates \
  torchkrates-train --config /workspace/data/train_settings_example.yaml
```

Config paths must use `/workspace/...`:
```yaml
GENERAL:
  name_exp: my_model
  checkpoints_dir: /workspace/checkpoints
  model_dir: /workspace/model
  log_dir: /workspace/logs
```

## Evaluate (inference)

```bash
sudo docker run \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/output:/workspace/output \
  torchkrates \
  torchkrates-eval \
    --model_path /workspace/data/my_model.model \
    --data_path /workspace/data/test.xyz \
    --output_file /workspace/output/out.h5
```

Bundled SO3LR pretrained model:
```bash
sudo docker run \
  -v $(pwd)/data:/workspace/data \
  torchkrates \
  torchkrates-eval \
    --model_path so3lr \
    --data_path /workspace/data/test.xyz \
    --output_file /workspace/data/out.h5
```

## Metrics

```bash
sudo docker run \
  -v $(pwd)/data:/workspace/data \
  torchkrates \
  torchkrates-metric \
    --model_path /workspace/data/my_model.model \
    --data_path /workspace/data/test.h5
```

## Merge HDF5 files

```bash
sudo docker run \
  -v $(pwd)/data:/workspace/data \
  torchkrates \
  torchkrates-merge \
    --inputs /workspace/data/a.h5 /workspace/data/b.h5 \
    --output /workspace/data/merged.h5
```

## Export LAMMPS model

```bash
sudo docker run \
  -v $(pwd)/data:/workspace/data \
  torchkrates \
  torchkrates-create-lammps-model \
    --model_path /workspace/data/my_model.model \
    --output_path /workspace/data/lammps_model.pt
```

## Multi-GPU training

```bash
sudo docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/model:/workspace/model \
  torchkrates \
  torchrun --nproc_per_node=2 \
    -m so3krates_torch.cli.run_train \
    --config /workspace/data/train_settings_example.yaml
```

## Interactive shell

```bash
sudo docker run -it \
  -v $(pwd)/data:/workspace/data \
  torchkrates \
  /bin/bash
```

---

## Notes

- All paths inside the container must start with `/workspace/...`
- Host directory `$(pwd)/data` maps to `/workspace/data` inside the container
- Checkpoints are saved as `{name_exp}_epoch-N.pt` in `checkpoints_dir`
- Final model is saved as `{name_exp}.pth` and `{name_exp}.model` in `model_dir`
- Replace `torchkrates` with `torchkrates-cpu` for CPU-only runs
