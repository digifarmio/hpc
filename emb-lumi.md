# EMB Project: LUMI Guide

LUMI-specific guide for the CDL crop segmentation from foundation model embeddings project. Covers AMD GPU differences, ROCm container gotchas, DDP on 8 GCDs, and the critical differences from Betzy.

---

## System Overview

| Property | Value |
|----------|-------|
| Hostname | `lumi.csc.fi` |
| User | `konstanv` |
| Account | `project_465002500` |
| GPU | AMD MI250X, 4 per node = 8 GCDs (logical GPUs) |
| VRAM | 64 GB per GCD |
| GPU partition | `small-g` (single node, up to 3 days) |
| CPU partition | `small` (single node, up to 7 days) |
| Container runtime | Singularity |
| PyTorch container | `/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif` |
| Dependencies venv | `/projappl/project_465002500/venvs/emb-deps/` |
| Internet from compute | Yes |

---

## Storage Layout

### Scratch (main workspace)

```
/scratch/project_465002500/emb/
```

- Code, data, outputs all live here
- **42-day purge policy** -- files not accessed for 42 days are deleted
- Touch files periodically or redownload as needed

```
/scratch/project_465002500/emb/
  src/                              # Python source
  configs/                          # YAML configs (including LUMI-specific)
  scripts/                          # Standalone scripts
  slurm/                            # SLURM batch scripts
  outputs/
    shards/tessera/                 # WebDataset shards
    runs/                           # Training checkpoints and logs
  configs/
    data/                           # tessera_channel_mean.npy, tessera_channel_std.npy
  logs/                             # SLURM output logs
```

### Projappl (persistent)

```
/projappl/project_465002500/venvs/emb-deps/
```

Persistent storage for virtual environment with Python dependencies that are not in the PyTorch container (webdataset, pyyaml, etc.). Not purged.

### No project space equivalent

Unlike Betzy, LUMI does not have a split between project/work storage. Everything goes in scratch (with purge risk) or projappl (persistent but small).

---

## Key Differences from Betzy

| Aspect | Betzy | LUMI |
|--------|-------|------|
| GPU | NVIDIA A100 (CUDA) | AMD MI250X (ROCm) |
| GPUs per node | 4 | 8 GCDs (4 physical MI250X) |
| Container flag | `apptainer exec --nv` | `singularity exec` (NO `--rocm`!) |
| Internet on compute | No | Yes |
| Module system on compute | Works | Broken (missing lua5.3) |
| Billing | 4 GPU-hours/wall-hour | 8 GPU-hours/wall-hour |
| Container | Custom `emb.sif` | LUMI-provided PyTorch container |
| Extra deps | Baked into container | External venv on projappl |
| PROJ fix needed | Yes | No (different container) |
| Max GPU wall-time | 7 days | 3 days |

---

## Container and Environment Setup

### DO NOT use the `--rocm` flag

The `--rocm` flag with `singularity exec` injects the host's `libdrm.so`, which requires GLIBC 2.33. The LUMI PyTorch container has GLIBC 2.31. This causes an immediate crash:

```
/lib/x86_64-linux-gnu/libdrm.so.2: version `GLIBC_2.33' not found
```

**Fix: use SINGULARITY_BIND instead:**

```bash
export SINGULARITY_BIND="/var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/pfs,/scratch,/projappl,/project,/flash,/appl"
```

This replicates what the `singularity-AI-bindings/24.03` module does, but without the `--rocm` flag.

### Module system is broken on compute nodes

LUMI compute nodes are missing `lua5.3`, so `module load` fails. You cannot load `singularity-AI-bindings` or any other module on compute nodes. Set all paths manually.

### PYTHONPATH must include both deps and project

The PyTorch container does not have webdataset, pyyaml, or the project's `src` package. Both must be on PYTHONPATH:

```bash
export PYTHONPATH=/projappl/project_465002500/venvs/emb-deps:/scratch/project_465002500/emb:${PYTHONPATH:-}
```

### MIOpen cache

AMD's MIOpen (equivalent of cuDNN) writes kernel caches. Point it to a job-local temp directory to avoid contention:

```bash
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}
```

### RCCL settings (AMD's NCCL equivalent)

For optimal single-node communication between 8 GCDs:

```bash
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
```

---

## SLURM Training Script

The full training SBATCH for LUMI:

```bash
#!/bin/bash
#SBATCH --job-name=emb-train
#SBATCH --account=project_465002500
#SBATCH --partition=small-g
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --output=/scratch/project_465002500/emb/logs/train_%j.log
#SBATCH --error=/scratch/project_465002500/emb/logs/train_%j.err

set -euo pipefail

PROJECT_DIR=/scratch/project_465002500/emb
CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif
DEPS_DIR=/projappl/project_465002500/venvs/emb-deps
CONFIG=${CONFIG:-configs/train/tessera_segformer_lumi.yaml}

echo "===== Training on LUMI ====="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPUs:       8 GCDs (4x MI250X)"
echo "Config:     ${CONFIG}"
echo "Start time: $(date -Iseconds)"

mkdir -p ${PROJECT_DIR}/logs ${PROJECT_DIR}/outputs/runs
cd "${PROJECT_DIR}"

# Do NOT use --rocm: injects host libdrm.so requiring GLIBC 2.33
export SINGULARITY_BIND="/var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/pfs,/scratch,/projappl,/project,/flash,/appl"

# MIOpen cache
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# RCCL settings
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB

singularity exec "${CONTAINER}" bash -c "
  export PYTHONPATH=${DEPS_DIR}:${PROJECT_DIR}:\${PYTHONPATH:-}
  export PYTHONUNBUFFERED=1
  torchrun --standalone --nproc_per_node=8 -m src.train --config ${CONFIG}
"

echo "End time: $(date -Iseconds)"
```

Submit with:

```bash
sbatch --export=CONFIG=configs/train/tessera_segformer_lumi.yaml \
  slurm/phase4_training/train_lumi.sbatch
```

### Billing note

1 wall-hour on `small-g` with 8 GCDs = 8 GPU-hours billed. A 3-day job = 576 GPU-hours. Budget carefully.

---

## LUMI-Specific Training Configs

LUMI configs differ from Betzy configs in several ways:

```yaml
# configs/train/tessera_segformer_lumi.yaml vs tessera_segformer.yaml
training:
  batch_size: 16          # 16 vs 32 (per-GPU, smaller to fit MI250X memory patterns)
  gradient_accumulation_steps: 2  # 2 vs 1 (compensate for smaller batch)
  num_workers: 2          # 2 vs 4 (fewer workers per GCD)
  ema:
    enabled: false        # EMA disabled on LUMI (reduces memory, simplifies debugging)

data:
  shuffle_buffer: 200     # 200 vs 2000 (reduce memory pressure)
  channel_mean_path: "/scratch/project_465002500/emb/configs/data/tessera_channel_mean.npy"
  channel_std_path: "/scratch/project_465002500/emb/configs/data/tessera_channel_std.npy"

wandb:
  enabled: false          # No wandb on LUMI initially
```

Effective global batch size: `16 * 2 * 8 = 256` (batch_size * accum_steps * GPUs), vs `32 * 1 * 4 = 128` on Betzy.

---

## Dependencies Virtual Environment

The PyTorch container has PyTorch, numpy, etc., but is missing several packages we need. These are installed in a persistent venv:

```
/projappl/project_465002500/venvs/emb-deps/
```

Packages installed:
- `webdataset` -- streaming dataset loading from .tar shards
- `pyyaml` -- YAML config parsing
- Any other pure-Python dependencies

To add a package:

```bash
# On a login node (has internet)
singularity exec $CONTAINER bash -c "
  python -m pip install --target=/projappl/project_465002500/venvs/emb-deps webdataset pyyaml
"
```

---

## Code Transfer

### Upload from EC2

```bash
# rsync works fine to scratch (unlike Betzy project space)
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='data/' --exclude='outputs/' \
  /home/ubuntu/emb/ lumi:/scratch/project_465002500/emb/

# Or scp for individual files
scp /home/ubuntu/emb/src/train.py lumi:/scratch/project_465002500/emb/src/train.py
```

### Upload data

TESSERA shards and stats files must be uploaded separately:

```bash
# Upload channel stats
scp /path/to/tessera_channel_mean.npy \
  lumi:/scratch/project_465002500/emb/configs/data/

# Upload shards (large transfer)
rsync -avz /path/to/shards/tessera/ \
  lumi:/scratch/project_465002500/emb/outputs/shards/tessera/
```

Since LUMI compute nodes have internet, an alternative is to download data directly from S3 within a SLURM job.

---

## DDP on 8 GCDs

Each MI250X has 2 GCDs (Graphics Compute Dies), each acting as an independent GPU. 4 MI250X per node = 8 GCDs.

```bash
torchrun --standalone --nproc_per_node=8 -m src.train --config ...
```

- `--standalone`: single-node DDP (no rendezvous server needed)
- `--nproc_per_node=8`: one process per GCD
- WebDataset `split_by_node` handles shard distribution across 8 ranks
- Ensure at least 8 shards for training (we have 32)
- Validation uses `num_workers=0` because shard count (8) equals rank count (8)

---

## Monitoring Jobs

```bash
# Check your queue
squeue -u konstanv

# Detailed job info
squeue -u konstanv --format="%.10i %.9P %.20j %.8T %.10M %.6D %R"

# Check finished job stats
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,MaxVMSize

# Follow live output
tail -f /scratch/project_465002500/emb/logs/train_JOBID.log
```

---

## Key File Locations

```
# Container (system-provided, persistent)
/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif

# Dependencies (persistent)
/projappl/project_465002500/venvs/emb-deps/

# Everything else (scratch, 42-day purge)
/scratch/project_465002500/emb/
  src/                              # Python source
  configs/train/                    # Training configs (including *_lumi.yaml)
  configs/data/                     # Channel stats
  outputs/shards/tessera/           # WebDataset shards
  outputs/runs/                     # Checkpoints
  logs/                             # SLURM logs
```
