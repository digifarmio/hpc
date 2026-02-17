# EMB Project: Betzy Guide

Betzy-specific guide for the CDL crop segmentation from foundation model embeddings project. Covers storage layout, container issues, SLURM patterns, and hard-won lessons from running Phases 3-5 on Betzy.

---

## System Overview

| Property | Value |
|----------|-------|
| Hostname | `betzy.sigma2.no` |
| User | `digifarm` |
| Account | `nn12037k` |
| GPU | NVIDIA A100 40GB |
| GPU partition | `accel` (4 nodes, 4 GPUs each, 64 CPUs, 494 GB RAM per node) |
| CPU partitions | `normal` (4-day max), `preproc` (1-day max) |
| Container runtime | Apptainer/Singularity 1.2.5 |
| Container path | `/cluster/projects/nn12037k/emb/containers/emb.sif` |
| Container specs | PyTorch 2.5.1+cu124, rasterio 1.4.4, GDAL 3.9.3, pyarrow 23.0.0 |
| AWS CLI | `~/.local/bin/aws` (account 633376437264) |
| CUDA modules | Up to 12.9.1 (load via `bash -l` or `source /etc/profile.d/modules.sh`) |
| Internet from compute | No |

---

## Critical Storage Layout

Betzy has two storage tiers with very different quotas. Getting this wrong causes `Disk quota exceeded` errors that silently kill jobs.

### Project space (read-only for data)

```
/cluster/projects/nn12037k/emb/
```

- Shared quota across ALL projects under `nn12037k` -- user `digifarm` has ~1.74 TB total, most consumed by other projects
- Use ONLY for: code, configs, container `.sif` file
- **NEVER write data, logs, shards, or checkpoints here**
- `rsync` fails here because it creates temp files that exceed quota. Use `scp` for individual files instead.

### Work space (all data goes here)

```
/cluster/work/users/digifarm/emb/
```

- Large quota, suitable for multi-TB datasets
- ALL large data lives here:

```
/cluster/work/users/digifarm/emb/
  data/
    tessera_cache/    # Downloaded TESSERA .npy tiles (~1 TB cached, 23 TB full)
    cdl/              # CDL COGs: cdl_{year}_conus.tif
  outputs/
    shards/           # WebDataset .tar shards (AEF + TESSERA)
    runs/             # Training checkpoints and logs
    eval/             # Evaluation results
  logs/               # SLURM stdout/stderr logs
  configs/
    data/             # tessera_channel_mean.npy, tessera_channel_std.npy
  tmp/                # Temp files (set TMPDIR here; system /tmp is tiny)
```

### Symlinks connecting the two

The project directory has symlinks pointing into work space so that scripts using relative paths from `PROJECT_DIR` still find data:

```bash
# These already exist -- do NOT recreate
/cluster/projects/nn12037k/emb/data/tessera_cache -> /cluster/work/users/digifarm/emb/data/tessera_cache
/cluster/projects/nn12037k/emb/data/cdl           -> /cluster/work/users/digifarm/emb/data/cdl
/cluster/projects/nn12037k/emb/outputs/shards      -> /cluster/work/users/digifarm/emb/outputs/shards
/cluster/projects/nn12037k/emb/tmp                 -> /cluster/work/users/digifarm/emb/tmp
```

---

## Container Configuration

The container (`emb.sif`) was built with `--fakeroot` and uses conda-based GDAL/rasterio. Two critical environment issues must be set for every `apptainer exec` call.

### PROJ database mismatch

rasterio bundles PROJ 9.7.1 (proj.db minor=6), but pyproj bundles PROJ 9.5.1 (proj.db minor=4). When pyproj loads its own proj.db, any EPSG lookup fails with "EPSG code unknown" errors.

**Fix: force PROJ_DATA to rasterio's copy:**

```bash
apptainer exec --bind /cluster \
  --env "PROJ_DATA=/opt/conda/lib/python3.11/site-packages/rasterio/proj_data" \
  ...
```

### PYTHONPATH for src module imports

The container does not have the project directory on its Python path. Any `from src.xxx import yyy` will fail with `ModuleNotFoundError: No module named 'src'`.

**Fix: set PYTHONPATH:**

```bash
apptainer exec --bind /cluster \
  --env "PYTHONPATH=/cluster/projects/nn12037k/emb" \
  ...
```

### Standard apptainer exec template

Every SLURM script should use this pattern:

```bash
PROJECT_DIR=/cluster/projects/nn12037k/emb
CONTAINER=${PROJECT_DIR}/containers/emb.sif

apptainer exec --bind /cluster \
  --env "PYTHONPATH=${PROJECT_DIR}" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PROJ_DATA=/opt/conda/lib/python3.11/site-packages/rasterio/proj_data" \
  --env "GDAL_HTTP_TIMEOUT=60" \
  --env "GDAL_HTTP_MAX_RETRY=3" \
  --env "GDAL_HTTP_RETRY_DELAY=2" \
  "${CONTAINER}" \
  python -m src.some_module --arg value
```

For GPU jobs, add `--nv` after `exec`:

```bash
apptainer exec --nv --bind /cluster "${CONTAINER}" \
  torchrun --nproc_per_node=4 -m src.train --config ...
```

---

## SLURM Patterns for EMB Phases

### Phase 3: Data Preparation (preproc partition)

#### 3a. Download TESSERA tiles for stats

```bash
sbatch slurm/phase3_sampling/download_tessera.sbatch
```

Uses `preproc` partition, 4 CPUs, 8 GB RAM, 4 hours. Downloads ~500 tiles (~60 GB) for computing channel normalization statistics.

#### 3c. Compute TESSERA channel stats

```bash
sbatch slurm/phase3_sampling/compute_tessera_stats.sbatch
```

Reads 500 tiles, outputs `tessera_channel_mean.npy` and `tessera_channel_std.npy` to `/cluster/work/users/digifarm/emb/configs/data/`.

#### 3d. Build WebDataset shards (array jobs)

AEF shards (12 total workers, 3 SLURM tasks with 4 workers each):

```bash
sbatch --array=0-2 --cpus-per-task=4 --mem=8G \
  --export=EMB_SOURCE=aef,WORKERS_PER_JOB=4 \
  slurm/phase3_sampling/build_shards.sbatch
```

TESSERA shards (stream mode -- downloads, processes, deletes tiles):

```bash
sbatch --array=0-3 --cpus-per-task=16 --mem=64G \
  --export=EMB_SOURCE=tessera,WORKERS_PER_JOB=1 \
  slurm/phase3_sampling/build_shards.sbatch
```

Key design decisions:
- `--stream-tessera` flag avoids caching all 23 TB of TESSERA tiles; only ~123 MB in flight at a time
- AEF Source Cooperative limits to ~12-20 concurrent HTTP connections; more causes 502 errors
- Checkpoint/resume: resubmit the same command after wall-time expiry; workers pick up where they left off
- `--signal=B:TERM@120`: SLURM sends SIGTERM 120 seconds before wall-time, script forwards to workers for graceful checkpoint
- SAS tokens for CDL Planetary Computer expire ~1 hour; refresh interval set to 200 samples (not 2000)

### Phase 4: Training (accel partition, DDP)

```bash
# SegFormer with AEF embeddings
sbatch --export=CONFIG=configs/train/aef_segformer.yaml \
  slurm/phase4_training/train.sbatch

# DeepLab with TESSERA embeddings
sbatch --export=CONFIG=configs/train/tessera_deeplab.yaml \
  slurm/phase4_training/train.sbatch
```

The training SBATCH requests:
- `--partition=accel`, `--gpus=4`, `--cpus-per-gpu=16`, `--mem=0` (all memory)
- `--time=7-00:00:00` (7 days)
- `torchrun --nproc_per_node=4` for DDP across 4 A100s

**Queue wait times**: The `accel` partition has only 4 nodes (16 GPUs total). Jobs routinely wait days in queue. Plan accordingly.

### Phase 5: Evaluation (accel partition, single GPU)

```bash
# Evaluate one model
sbatch --export=CONFIG=configs/train/aef_segformer.yaml \
  slurm/phase5_eval/evaluate.sbatch

# Evaluate all 4 models
for cfg in aef_segformer aef_deeplab tessera_segformer tessera_deeplab; do
  sbatch --export=CONFIG=configs/train/${cfg}.yaml \
    slurm/phase5_eval/evaluate.sbatch
done
```

Uses 1 GPU, 32 GB RAM, 4 hours. Auto-detects checkpoint from `outputs/runs/{run_name}/best.pt`.

---

## Partition Rules

| Partition | `--mem` flag | `--gpus` | Max wall-time | Notes |
|-----------|-------------|----------|---------------|-------|
| `accel`   | OK (`--mem=0` for all) | Required | 7 days | A100 40GB, 4 per node |
| `preproc` | OK | No | 1 day | CPU-only, good for I/O-bound data prep |
| `normal`  | **NO** (do NOT use `--mem`) | No | 4 days | CPU-only, large node count |

Using `--mem` on the `normal` partition causes job rejection.

---

## Code Sync from EC2

### Preferred: scp for individual files

```bash
scp /home/ubuntu/emb/src/train.py betzy:/cluster/projects/nn12037k/emb/src/train.py
```

### rsync for bulk sync (to work space only)

```bash
# This works: syncing to work space
rsync -avz --exclude='.git' --exclude='__pycache__' \
  /home/ubuntu/emb/ betzy:/cluster/work/users/digifarm/emb/

# This FAILS: syncing to project space (temp files exceed quota)
rsync -avz ... /home/ubuntu/emb/ betzy:/cluster/projects/nn12037k/emb/  # DON'T DO THIS
```

### Safe rsync to project space (code only, small files)

If you must sync code to project space, sync only small files and use `--inplace` to avoid temp files:

```bash
rsync -avz --inplace \
  --exclude='.git' --exclude='__pycache__' --exclude='data/' --exclude='outputs/' \
  --exclude='*.parquet' --exclude='*.geojson' --exclude='*.sif' \
  /home/ubuntu/emb/ betzy:/cluster/projects/nn12037k/emb/
```

---

## Module Loading

Modules are not available by default in non-login shells (SLURM scripts). Load the module system first:

```bash
# Option 1: source the module init
source /etc/profile.d/modules.sh

# Option 2: use bash login shell
bash -l -c "module load CUDA/12.9.1 && ..."
```

Available CUDA modules go up to 12.9.1 (but the container has its own CUDA 12.4, so module loading is rarely needed).

---

## Interactive GPU Session

```bash
ssh betzy
srun --partition=accel --gpus=1 --mem=32G --time=4:00:00 --account=nn12037k --pty bash

# Inside the interactive session:
cd /cluster/projects/nn12037k/emb
apptainer exec --nv --bind /cluster containers/emb.sif python -c "import torch; print(torch.cuda.get_device_name())"
```

---

## Key File Locations

```
# Code and configs (project space, tiny quota)
/cluster/projects/nn12037k/emb/
  src/                          # Python source
  configs/                      # YAML configs
  scripts/                      # Standalone scripts
  slurm/                        # SLURM batch scripts
  containers/emb.sif            # Apptainer container

# Data and outputs (work space, large quota)
/cluster/work/users/digifarm/emb/
  data/tessera_cache/            # TESSERA .npy tiles
  data/cdl/                      # CDL CONUS GeoTIFFs
  outputs/shards/aef/            # AEF WebDataset shards
  outputs/shards/tessera/        # TESSERA WebDataset shards
  outputs/runs/                  # Training checkpoints
  outputs/eval/                  # Evaluation results
  configs/data/                  # TESSERA channel stats (.npy)
  logs/                          # All SLURM logs
```
