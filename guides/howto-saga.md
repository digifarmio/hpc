# How-To Guide: Saga

Step-by-step recipes for common tasks on Saga (NRIS/Sigma2, Norway). CPU + P100 GPUs, Apptainer containers, no internet from compute nodes.

---

## Connect to Saga

```bash
# First time: add to ~/.ssh/config
Host saga
    HostName saga.sigma2.no
    User digifarm
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 7d

# Create socket directory
mkdir -p ~/.ssh/sockets

# Connect (OTP-based auth via authenticator app)
ssh saga
```

---

## Transfer Code to Saga

```bash
# From dev machine
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '/data' --exclude 'outputs/' \
    /home/ubuntu/myproject/ saga:/cluster/work/users/digifarm/myproject/
```

**Important:** Use `--exclude '/data'` with leading slash (excludes only top-level `data/`). Without the slash, rsync also removes `src/*/data/` directories containing Python source.

No internet on compute nodes, so all code and data must be transferred before job submission.

---

## Set Up a Python Environment

Saga's module-provided PyTorch is broken (stub CUDA libraries only). Use a pip-installed venv:

```bash
# One-time setup on login node
cd /cluster/work/users/digifarm/myproject
module purge
module load Python/3.10.4-GCCcore-11.3.0
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

**Always `module purge` first.** Leftover module state causes cryptic library conflicts.

---

## Set Up an Apptainer Container

For geospatial workflows (rasterio, GDAL, pyproj), containers are more reliable than venvs:

```bash
# Build container from definition file
apptainer build mycontainer.sif mycontainer.def

# Or from Docker image
apptainer build mycontainer.sif docker-archive://myimage.tar.gz
```

### Install extra packages into a target directory

```bash
cd /cluster/work/users/digifarm/myproject
apptainer exec containers/mycontainer.sif \
    pip install --target pylibs boto3 cdsapi xarray netCDF4
```

**Important:** Never modify a `pylibs/` directory while running jobs reference it. Lustre caching causes `Stale file handle` errors. If you need to update packages, create a new directory (e.g., `pylibs2/`).

---

## Run a CPU Job

```bash
#!/bin/bash
#SBATCH --job-name=my_pipeline
#SBATCH --account=nn12037k
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

PROJECT_DIR=/cluster/work/users/digifarm/myproject
CONTAINER="${PROJECT_DIR}/containers/mycontainer.sif"

apptainer exec \
  --bind "$PROJECT_DIR" \
  --env "PYTHONPATH=${PROJECT_DIR}/pylibs" \
  --env "PROJ_DATA=/usr/local/lib/python3.12/dist-packages/rasterio/proj_data" \
  --env "PYTHONUNBUFFERED=1" \
  "$CONTAINER" \
  python3 "${PROJECT_DIR}/scripts/run_pipeline.py"
```

Submit:

```bash
sbatch myjob.sbatch
```

---

## Run a GPU Job

```bash
#!/bin/bash
#SBATCH --job-name=my_training
#SBATCH --account=nn12037k
#SBATCH --partition=accel
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

module purge
module load Python/3.10.4-GCCcore-11.3.0
source /cluster/work/users/digifarm/myproject/venv/bin/activate
export PYTHONPATH="/cluster/work/users/digifarm/myproject/src:${PYTHONPATH:-}"
export WANDB_MODE=disabled

python3 scripts/train.py --config config/model_small.yaml
```

Key notes:
- Partition is `accel` (not `gpu`, not `small-g`)
- Check GPU availability first: `sinfo -p accel --format="%n %G %t %C"`
- P100 GPUs have 16GB VRAM -- only suitable for small models

---

## Run a Containerized Job with Correct PROJ_DATA

For any rasterio/pyproj workflow, PROJ_DATA must point to the container's own proj.db:

```bash
apptainer exec \
  --bind "$PROJECT_DIR" \
  --env "PROJ_DATA=/usr/local/lib/python3.12/dist-packages/rasterio/proj_data" \
  --env "AWS_NO_SIGN_REQUEST=YES" \
  --env "GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR" \
  "$CONTAINER" \
  python3 script.py
```

**Why:** The system proj.db has an old schema version. Without this fix, coordinate transforms produce **silently wrong results** -- not errors, just incorrect coordinates.

**GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR** gives a 3-5x speedup for S3 COG reads by preventing GDAL from listing the parent S3 directory before each file open.

---

## Run a Fully Isolated Container Job

For maximum reproducibility, use `--containall --cleanenv`:

```bash
apptainer run --containall --cleanenv \
  --bind "$output_dir:/data/results" \
  --env "AWS_NO_SIGN_REQUEST=YES" \
  --env "GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR" \
  --env "TMPDIR=/tmp" \
  "$CONTAINER" \
  --input-file /embedded/data.geojson --output-dir /data/results
```

This strips ALL host environment and filesystem. Only explicitly bound paths and `--env` variables are visible. More work to set up, but eliminates host environment leak bugs.

**Always test isolated containers with a single item first** before submitting batch jobs.

---

## Start an Interactive Session

### CPU

```bash
srun --account=nn12037k --partition=normal --mem=16G --time=01:00:00 --pty bash
```

### GPU

```bash
srun --account=nn12037k --partition=accel --gpus=1 --mem=16G --time=01:00:00 --pty bash
```

Inside the session:

```bash
module purge
module load Python/3.10.4-GCCcore-11.3.0
source /cluster/work/users/digifarm/myproject/venv/bin/activate
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Pre-Load Climate Data for Batch Processing

For pipelines that process many fields against the same climate dataset, pre-load everything into RAM before the per-field loop:

```python
import xarray as xr
import rasterio

# Phase 1: Bulk load (one-time, ~700MB)
era5_data = {}
for f in era5_files:
    ds = xr.open_dataset(f).load()  # .load() forces into numpy arrays
    era5_data[month] = ds

chirps_data = {}
for f in chirps_files:
    with rasterio.open(f) as src:
        chirps_data[date] = src.read(1)

# Phase 2: Per-field extraction (pure numpy, ~0.1ms each)
for field in fields:
    value = extract_from_preloaded(era5_data, field.centroid)
```

This avoids Lustre's random I/O latency (~100ms per read). Throughput goes from ~1 field/second to 6-7 fields/second.

---

## Build a Resumable Batch Pipeline

```python
import glob, os

# Build work list, skipping completed items
work_list = []
for item_id in all_items:
    output_path = f"outputs/{item_id}/result.json"
    if os.path.exists(output_path):
        continue  # already done
    work_list.append(item_id)

print(f"Skipping {len(all_items) - len(work_list)} completed, processing {len(work_list)}")
```

In the worker function, double-check (race condition guard):

```bash
if ls "$output_dir"/result.json &>/dev/null 2>&1; then
    return 0  # already completed by another worker
fi
```

---

## Monitor Jobs

```bash
# Check queue
squeue -u digifarm

# Detailed format
squeue -u digifarm --format="%.10i %.9P %.20j %.8T %.10M %.6D %R"

# Post-mortem stats
sacct -j JOBID --format=JobID,Elapsed,MaxRSS,MaxVMSize,State

# Follow live output
tail -f logs/my_pipeline_JOBID.out

# Check GPU node availability
sinfo -p accel --format="%n %G %t %C"

# Check billing quota
cost -p nn12037k

# Cancel jobs
scancel JOBID
scancel -u digifarm          # cancel all
```

---

## Retrieve Results

```bash
# From dev machine: download outputs
rsync -avz saga:/cluster/work/users/digifarm/myproject/outputs/ local_outputs/

# Just specific file types
rsync -avz --include='*/' --include='*.csv' --exclude='*' \
    saga:/cluster/work/users/digifarm/myproject/outputs/ local_outputs/
```

---

## Filesystem Quick Reference

| Path | Quota | Purge | Use for |
|------|-------|-------|---------|
| `/cluster/work/users/digifarm/` | Large | ~42-90 days inactive | All project data, outputs, venvs |
| `/cluster/projects/nn12037k/` | Shared, limited | Never | Code only (not data) |

**Warning:** `/cluster/work/` has no backup. Always retrieve important results to your dev machine.

---

## Common Errors and Fixes

| Error | Fix |
|-------|-----|
| `torch.cuda` failures | Don't use module PyTorch; pip install in venv |
| Cryptic library conflicts | `module purge` before loading anything |
| Silently wrong coordinates | Set `PROJ_DATA` to container's rasterio proj_data path |
| `Stale file handle` on imports | Create fresh `pylibs2/` directory; don't modify in-place |
| Segfault with ThreadPoolExecutor + rasterio | Use sequential processing; GDAL is not thread-safe |
| `python3 -u` or buffered logs | Set `PYTHONUNBUFFERED=1` |
| GPU jobs stuck in queue | Check `sinfo -p accel`; nodes may be down |
| `xr.concat` HDF errors | Use `.load()` before concat; don't lazy-concat NetCDF |
