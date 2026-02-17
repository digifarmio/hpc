# SAGA HPC Guide

SAGA is a Norwegian NRIS/Sigma2 cluster. In the vcloud project it served as the initial development cluster for data preparation and small-model validation before scaling up to LUMI.

## Cluster Details

| Item | Value |
|------|-------|
| Host | `saga.sigma2.no` |
| SSH alias | `saga` |
| User | `digifarm` |
| SLURM account | `nn12037k` |
| GPU partition | `accel` |
| GPUs | NVIDIA Tesla P100, 16GB VRAM |
| Max job time | 7 days |
| Project directory | `/cluster/work/users/digifarm/vcloud/v0` |

## SSH Access

NRIS uses one-time passwords (OTP) via an authenticator app. Once you authenticate, the SSH connection persists for approximately 7 days.

```bash
# SSH config entry (~/.ssh/config)
Host saga
    HostName saga.sigma2.no
    User digifarm

# Connect
ssh saga
```

## Environment Setup

SAGA uses environment modules and a Python virtual environment. The module-provided PyTorch was broken (see Issues below), so we installed PyTorch via pip in a venv.

```bash
# Initial setup (run once on login node)
cd /cluster/work/users/digifarm/vcloud/v0
module purge
module load Python/3.10.4-GCCcore-11.3.0
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

```bash
# Before every session / in every job script
module purge
module load Python/3.10.4-GCCcore-11.3.0
source /cluster/work/users/digifarm/vcloud/v0/venv/bin/activate
export PYTHONPATH="/cluster/work/users/digifarm/vcloud/v0/src:${PYTHONPATH:-}"
```

**Always use `module purge` first.** Without it, leftover module state can cause library conflicts that produce cryptic errors.

## Directory Structure

```
/cluster/work/users/digifarm/vcloud/v0/
    config/                     # model_small.yaml, model_full.yaml
    src/vcloud/                 # Source code
    scripts/                    # CLI entry points
    slurm/                      # SLURM job scripts
    data/
        sen12mscrts_real/
            prepared/           # 33GB prepared dataset
                train/          # 23,888 patches from 800 locations
                val/            # 2,974 patches from 100 locations
                test/           # 2,900 patches from 100 locations
    outputs_temporal/           # Small-model checkpoints (stages 0-1)
    venv/                       # Python virtual environment
    logs/                       # SLURM job logs
```

## Data Preparation

SAGA was used for the initial SEN12MS-CR-TS dataset download, conversion, and preparation:

- **Source**: SEN12MS-CR-TS benchmark dataset (MediaTUM)
- **Coverage**: 1,000 locations in Africa
- **Temporal depth**: ~30 time steps per location, Jan-Dec 2018
- **Patch size**: 256x256 pixels
- **Total size**: 33GB prepared

```bash
# Download (SLURM array job)
sbatch slurm/phase1_download/download_sen12mscrts.sbatch

# Convert from raw format to prepared NPZ files
sbatch slurm/phase1_download/extract_and_convert_ts.sbatch

# Prepare temporal data (create index, compute statistics, split train/val/test)
sbatch slurm/phase2_prep/prepare_temporal.sbatch
```

## Training (Small Model Validation)

SAGA was used to run stages 0 and 1 with `model_small.yaml` to validate the architecture before committing LUMI GPU-hours.

```bash
cd /cluster/work/users/digifarm/vcloud/v0
sbatch slurm/phase4_training/train_all_temporal.sbatch
```

The small model config uses 64x64 patches, 4 temporal dates, and much smaller network dimensions, so it fits easily on P100 16GB GPUs.

Old checkpoints at:
- `outputs_temporal/stage0/` (autoencoder)
- `outputs_temporal/stage1/` (draft model, 44 epochs, avg PSNR 31.87 dB)

## SLURM Job Script Example

```bash
#!/bin/bash
#SBATCH --job-name=vcloud_temporal
#SBATCH --account=nn12037k
#SBATCH --partition=accel
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load Python/3.10.4-GCCcore-11.3.0
source /cluster/work/users/digifarm/vcloud/v0/venv/bin/activate
export PYTHONPATH="/cluster/work/users/digifarm/vcloud/v0/src:${PYTHONPATH:-}"
export WANDB_MODE=disabled

python3 scripts/train.py \
    --stage 0 \
    --config config/model_small.yaml \
    --data-dir data/sen12mscrts_real/prepared \
    --output-dir outputs_temporal/stage0
```

Key SLURM directives for SAGA:
- `--partition=accel` (NOT `gpu`, NOT `small-g` — those are for other clusters)
- `--account=nn12037k`
- `--gpus=1` (request GPU from the accel partition)
- `--requeue` (add for long jobs to handle preemption)

## Useful Commands

```bash
# Check queue
squeue -u digifarm

# Check job resources after completion
sacct -j JOBID --format=JobID,Elapsed,MaxRSS,MaxVMSize,State

# Interactive GPU session (for debugging)
srun --account=nn12037k --partition=accel --gpus=1 --mem=16G --time=01:00:00 --pty bash

# Check GPU node availability
sinfo -p accel --format="%n %G %t %C"

# Check billing quota
cost -p nn12037k
```

## Viewer Data Export

SAGA was also used to export the initial viewer data (small-model results):

```bash
sbatch slurm/phase5_viewer/export_viewer.sbatch
```

This exports PNGs + manifest.json from checkpoints for the S3-hosted static viewer.

## Issues and Caveats

### GPU Node Availability (MAJOR)

GPU nodes on SAGA were frequently DOWN or unavailable. Jobs would sit in the queue for days waiting for a working GPU node. This was the single biggest frustration with SAGA.

**Workaround**: Check node status with `sinfo -p accel` before submitting. If all nodes show `down` or `drain`, switch to LUMI or Betzy.

### Limited Billing Quota

The `nn12037k` account is shared across projects (digifarm uses it for multiple workloads). By late 2025, only ~10% of the billing quota remained. Check with `cost -p nn12037k`.

### P100 VRAM Too Small for Full Model

The P100 GPUs have only 16GB VRAM — sufficient for the small model (model_small.yaml: 64x64 patches, batch_size=8) but far too small for the full model (model_full.yaml: 154M params, 256x256 patches, requires ~55GB even with all memory optimizations).

**Do not attempt full-model training on SAGA.** Use LUMI.

### Broken Module PyTorch

The SAGA-provided PyTorch module (`PyTorch/1.12.1-foss-2022a-CUDA-11.7.0`) was broken. It contained only CUDA stub libraries, not the actual runtime libraries (`libcurand.so.10`, `libcufft.so.10`, etc.). Importing `torch.cuda` would fail or produce wrong results.

**Solution**: Use pip-installed PyTorch in a venv:
```bash
module purge
module load Python/3.10.4-GCCcore-11.3.0
source venv/bin/activate  # Has pip-installed torch with bundled CUDA runtime
```

We created `slurm/debug_cuda.sbatch` specifically to diagnose this. If you encounter CUDA problems on SAGA, start there.

### No Container Approach

SAGA uses a bare venv, not containers. This is more brittle:
- Dependency conflicts between pip packages and module-provided packages
- CUDA library version mismatches
- Harder to reproduce the exact environment

If starting fresh, consider using Apptainer/Singularity on SAGA instead.

### Filesystem Warnings

- `/cluster/work/` has **no backup**. Always push important checkpoints elsewhere.
- `/cluster/work/` has a generous quota but is shared across all project users.
- There is a purge policy (files unused for ~42-90 days may be deleted).
