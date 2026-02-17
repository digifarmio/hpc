# How-To Guide: LUMI

Step-by-step recipes for common tasks on LUMI (CSC, Finland). AMD MI250X GPUs, Singularity containers, ROCm stack.

---

## Connect to LUMI

```bash
# First time: add to ~/.ssh/config
Host lumi2
    HostName lumi.csc.fi
    User digifarm
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 7d

# Create socket directory
mkdir -p ~/.ssh/sockets

# Connect (OTP-based auth)
ssh lumi2
```

After the first connection, `ControlPersist 7d` keeps the SSH socket open for 7 days. All subsequent SSH/SCP/rsync connections are instant.

---

## Transfer Code to LUMI

```bash
# From your dev machine (e.g., EC2)
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '/data' --exclude 'outputs/' \
    /home/ubuntu/myproject/ lumi2:/scratch/project_465002500/myproject/
```

**Critical:** Use `--exclude '/data'` (leading slash), not `--exclude 'data'`. Without the leading slash, rsync also excludes `src/mypackage/data/` (Python source code), causing `ModuleNotFoundError` on LUMI that is extremely hard to debug.

For individual files:

```bash
scp myfile.py lumi2:/scratch/project_465002500/myproject/
```

---

## Run a GPU Training Job

### Step 1: Write the outer SBATCH script

```bash
#!/bin/bash
#SBATCH --job-name=my_training
#SBATCH --account=project_465002500
#SBATCH --partition=small-g
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SIF="/appl/local/csc/soft/ai/images/pytorch_2.5.1_lumi_rocm6.2.4_flash-attn-3.0.0.post1.sif"
export SINGULARITY_CONTAINLIBS="/usr/lib64/libcxi.so.1,/opt/rocm/lib/librocm_smi64.so.7"

mkdir -p logs

srun singularity exec \
    -B /opt/cray/libfabric \
    -B /scratch/project_465002500 \
    -B /projappl/project_465002500 \
    "$SIF" \
    bash /scratch/project_465002500/myproject/train_inner.sh
```

### Step 2: Write the inner script (runs inside container)

```bash
#!/bin/bash
set -euo pipefail

export PYTHONPATH="/scratch/project_465002500/myproject/src:/projappl/project_465002500/pylibs:${PYTHONPATH:-}"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

cd /scratch/project_465002500/myproject

python3 scripts/train.py --config config/model.yaml
```

### Step 3: Submit

```bash
ssh lumi2
cd /scratch/project_465002500/myproject
mkdir -p logs
sbatch train.sbatch
```

---

## Run a Multi-GPU DDP Job (8 GCDs)

Each LUMI node has 4 MI250X GPUs = 8 GCDs (logical GPUs).

### Outer SBATCH

```bash
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
```

### Inner script

```bash
# RCCL settings for single-node multi-GCD
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB

# MIOpen cache (avoid contention between ranks)
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

torchrun --standalone --nproc_per_node=8 -m src.train --config configs/model.yaml
```

**Billing:** 1 wall-hour with 8 GCDs = 8 GPU-hours. A 3-day job = 576 GPU-hours.

---

## Run a Multi-GPU DDP Job with SINGULARITY_BIND

For the EMB project containers (or when `singularity-AI-bindings` module is not available):

### Outer SBATCH

```bash
# Do NOT use --rocm flag (causes GLIBC 2.33 mismatch)
export SINGULARITY_BIND="/var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/pfs,/scratch,/projappl,/project,/flash,/appl"

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif

singularity exec "${CONTAINER}" bash -c "
  export PYTHONPATH=/projappl/project_465002500/venvs/emb-deps:/scratch/project_465002500/emb:\${PYTHONPATH:-}
  export PYTHONUNBUFFERED=1
  torchrun --standalone --nproc_per_node=8 -m src.train --config \${CONFIG}
"
```

---

## Install Extra Python Packages

The CSC container doesn't include everything. Install to `/projappl/` (persistent):

```bash
# Start interactive session
srun --account=project_465002500 --partition=small-g --time=00:30:00 --ntasks=1 --pty bash

# Install inside container to projappl
SIF="/appl/local/csc/soft/ai/images/pytorch_2.5.1_lumi_rocm6.2.4_flash-attn-3.0.0.post1.sif"
singularity exec "$SIF" pip install --target=/projappl/project_465002500/pylibs tifffile webdataset pyyaml
```

Then add to PYTHONPATH in your inner script:

```bash
export PYTHONPATH="/projappl/project_465002500/pylibs:${PYTHONPATH:-}"
```

---

## Start an Interactive GPU Session

```bash
srun --account=project_465002500 --partition=small-g --gpus=1 --mem=64G --time=04:00:00 --pty bash
```

Once on the node:

```bash
SIF="/appl/local/csc/soft/ai/images/pytorch_2.5.1_lumi_rocm6.2.4_flash-attn-3.0.0.post1.sif"
export SINGULARITY_CONTAINLIBS="/usr/lib64/libcxi.so.1,/opt/rocm/lib/librocm_smi64.so.7"
singularity exec -B /opt/cray/libfabric -B /scratch/project_465002500 "$SIF" bash
```

Inside the container, verify GPU access:

```python
import torch
print(f'PyTorch: {torch.__version__}')
print(f'ROCm: {torch.version.hip}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
```

Expected: `AMD Instinct MI250X`, `~68.7 GB`.

---

## Monitor Jobs

```bash
# Check your queue
squeue -u $USER

# Detailed format
squeue -u $USER --format="%.10i %.9P %.20j %.8T %.10M %.6D %R"

# Follow live output
tail -f logs/my_training_JOBID.out

# Post-mortem stats (memory, runtime)
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,MaxVMSize

# Cancel jobs
scancel JOBID                  # one job
scancel -u $USER               # all your jobs
```

---

## Check GPU Memory from Checkpoint

When WandB is disabled, extract metrics from saved checkpoints:

```python
import torch
c = torch.load('outputs/stage1/latest.pt', map_location='cpu')
print(f'Epoch {c["epoch"]}, Loss {c.get("best_val_loss", "?")}, PSNR {c.get("best_val_psnr", "?")}')
```

---

## Resume Training After Interruption

Training scripts should auto-resume from `latest.pt`:

```bash
RESUME_ARG=""
if [ -f "$OUTPUT_DIR/latest.pt" ]; then
    echo "Resuming from $OUTPUT_DIR/latest.pt"
    RESUME_ARG="--resume $OUTPUT_DIR/latest.pt"
fi
python3 scripts/train.py --config config/model.yaml $RESUME_ARG
```

With `#SBATCH --requeue`, SLURM automatically resubmits preempted jobs.

---

## Skip Completed Stages

```bash
START_STAGE="${START_STAGE:-0}"
for STAGE in 0 1 2 3; do
    [ "$STAGE" -lt "$START_STAGE" ] && continue
    echo "=== Running stage $STAGE ==="
    python3 scripts/train.py --stage $STAGE ...
done
```

Submit skipping to stage 2:

```bash
sbatch --export=ALL,START_STAGE=2 train.sbatch
```

---

## Use bfloat16 Mixed Precision (Correctly)

```python
# CORRECT: bfloat16 autocast, NO GradScaler
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(batch)
    loss = criterion(outputs, target)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

**Never use `GradScaler` with bfloat16.** It skips 75-87% of optimizer steps due to false-positive inf detections. GradScaler is only for float16.

---

## Filesystem Quick Reference

| Path | Speed | Purge | Use for |
|------|-------|-------|---------|
| `/scratch/project_*/` | Fast parallel | 90 days inactive | Code, data, outputs |
| `/projappl/project_*/` | Moderate | Never | Extra pip packages |
| `/flash/project_*/` | Ultra-fast NVMe | Short-term | Temp training cache |

---

## Common Errors and Fixes

| Error | Fix |
|-------|-----|
| `No HIP GPUs available` | Add `SINGULARITY_CONTAINLIBS` and `-B /opt/cray/libfabric` |
| `GLIBC_2.33 not found` | Remove `--rocm` flag, use `SINGULARITY_BIND` instead |
| `OutOfMemoryError` at ~40GB | Set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` |
| `ModuleNotFoundError: vcloud.data` | Fix rsync exclude: use `--exclude '/data'` (leading slash) |
| WandB hangs at startup | Set `WANDB_MODE=disabled` |
| `module: command not found` | Don't use `module load` on compute nodes; set paths manually |
| Job sits in queue forever | Use `small-g` partition, reduce resource requests |
