# LUMI HPC Guide (PRIMARY)

LUMI is a EuroHPC pre-exascale supercomputer hosted by CSC in Finland. It is the primary training cluster for vcloud, providing AMD MI250X GPUs with 64GB VRAM and a 100,000 GPU-hour budget.

This is the most detailed guide because LUMI is where the actual production training happens.

## Cluster Details

| Item | Value |
|------|-------|
| Host | `lumi.csc.fi` |
| SSH alias | `lumi2` |
| User | `digifarm` |
| SLURM account | `project_465002500` |
| GPU-hour budget | 100,000 |
| GPUs | AMD MI250X, ~68.7GB VRAM (2x 32GB GCDs + overhead) |
| GPU compute stack | ROCm 6.2 (**NOT CUDA**) |
| GPU partitions | `small-g` (1-4 GPUs), `standard-g` (full nodes), `dev-g` (debug) |
| Max job time | 3 days (`small-g`), 2 days (`standard-g`) |

## SSH Access

```bash
Host lumi2
    HostName lumi.csc.fi
    User digifarm
```

CSC uses similar OTP-based authentication to NRIS.

## Directory Structure

```
/scratch/project_465002500/vcloud/
    code/                           # Source code (rsync'd from local)
        config/                     # model_small.yaml, model_full.yaml
        src/vcloud/                 # Python package
        scripts/                    # CLI entry points
        slurm/                      # SLURM job scripts
    data/
        sen12mscrts_real/
            prepared/               # 33GB prepared dataset
                train/              # 23,888 patches, 800 locations
                val/                # 2,974 patches, 100 locations
                test/               # 2,900 patches, 100 locations
    outputs_full/                   # Production checkpoints
        stage0/                     # Autoencoder (COMPLETE: 39.6 dB, 50 epochs)
            best.pt (342MB)
            latest.pt
        stage1/                     # Draft model (IN PROGRESS: epoch 46/100, 34.0 dB)
        stage2/                     # Diffusion refiner (pending)
        stage3/                     # Self-training (pending)
    viewer_data_full/               # Exported viewer images (from export job)
    logs/

/projappl/project_465002500/
    pylibs/                         # Extra Python packages not in container (tifffile)

/flash/project_465002500/
    vcloud_tmp/                     # Ultra-fast NVMe temp space
```

### Filesystem Characteristics

| Path | Speed | Quota | Purge Policy | Use For |
|------|-------|-------|-------------|---------|
| `/scratch/` | Fast parallel FS | Large | 90-day inactive | Active data, outputs, code |
| `/projappl/` | Moderate | Small | Persistent | Extra Python packages |
| `/flash/` | Ultra-fast NVMe | Very small | Short-term | Temporary training cache |
| `/tmp` (in container) | Ephemeral | Tiny | Gone on job end | **NEVER rely on this** |

## Container Setup

LUMI uses Singularity/Apptainer containers with a CSC-provided PyTorch image. Far more reliable than the bare venv approach on SAGA.

### Container Image

```bash
SIF="/appl/local/csc/soft/ai/images/pytorch_2.5.1_lumi_rocm6.2.4_flash-attn-3.0.0.post1.sif"
```

Includes PyTorch 2.5.1 with ROCm 6.2.4 and Flash Attention 3.0.

### GPU Bindings (CRITICAL)

Without these, PyTorch will not detect any GPUs inside the container:

```bash
export SINGULARITY_CONTAINLIBS="/usr/lib64/libcxi.so.1,/opt/rocm/lib/librocm_smi64.so.7"

srun singularity exec \
    -B /opt/cray/libfabric \
    -B /scratch/project_465002500 \
    -B /projappl/project_465002500 \
    -B /flash/project_465002500 \
    "$SIF" \
    bash your_inner_script.sh
```

What each binding does:
- `libcxi.so.1` — Cray interconnect, required for GPU communication
- `librocm_smi64.so.7` — ROCm System Management Interface, required for GPU detection
- `-B /opt/cray/libfabric` — Inter-node communication (needed even for single-node)
- `-B /scratch/...`, `-B /projappl/...` — Project filesystems visible inside container

### Two-Script Pattern

Use outer + inner scripts. This keeps SLURM concerns separate from Python and makes debugging much easier.

**Outer script** (`.sbatch`): SLURM directives, container setup, `srun singularity exec`

```bash
#!/bin/bash
#SBATCH --job-name=vcloud_full
#SBATCH --account=project_465002500
#SBATCH --partition=small-g
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

SIF="/appl/local/csc/soft/ai/images/pytorch_2.5.1_lumi_rocm6.2.4_flash-attn-3.0.0.post1.sif"
export SINGULARITY_CONTAINLIBS="/usr/lib64/libcxi.so.1,/opt/rocm/lib/librocm_smi64.so.7"

srun singularity exec \
    -B /opt/cray/libfabric \
    -B /scratch/project_465002500 \
    -B /projappl/project_465002500 \
    "$SIF" \
    bash "$CODE_DIR/slurm/phase4_training/lumi_train_inner.sh"
```

**Inner script** (`.sh`): Runs INSIDE the container. Sets PYTHONPATH, env vars, runs Python.

```bash
#!/bin/bash
set -euo pipefail

export PYTHONPATH="/scratch/project_465002500/vcloud/code/src:/projappl/project_465002500/pylibs:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

python3 scripts/train.py --stage 0 --config config/model_full.yaml ...
```

## Environment Variables (Set Inside Container)

```bash
# CRITICAL: Prevents PyTorch memory allocator fragmentation on AMD GPUs.
# Without this, you get OOM errors at ~40GB instead of ~60GB.
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# CRITICAL: WandB hangs inside Singularity containers on LUMI (network restrictions).
# Disable it and rely on log files + checkpoint metrics.
export WANDB_MODE=disabled

# CRITICAL: Must include both source code AND extra packages.
export PYTHONPATH=".../code/src:/projappl/project_465002500/pylibs:${PYTHONPATH:-}"
```

## AMD MI250X and ROCm vs NVIDIA CUDA

### What works the same

- `torch.cuda.is_available()` → True (PyTorch maps "cuda" to HIP internally)
- `torch.device("cuda")` → selects the MI250X
- `torch.autocast("cuda", dtype=torch.bfloat16)` → works (use `"cuda"`, NOT `"hip"`)
- `model.cuda()`, standard ops, optimizers, schedulers → all work

### What is different

- **No `nvidia-smi`**: Use `rocm-smi` (may not be in container)
- **Memory architecture**: 2x 32GB GCDs appear as one ~64GB device with overhead showing ~68.7GB
- **bf16 is native and fast**: MI250X has native bf16 support
- **Memory allocator**: Without `expandable_segments:True`, fragments badly → early OOM
- **Flash Attention**: CSC container has flash-attn 3.0 compiled for MI250X

### Verifying GPU access

```python
import torch
print(f'PyTorch: {torch.__version__}')
print(f'ROCm: {torch.version.hip}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
```

Expected:
```
PyTorch: 2.5.1+rocm6.2.4
GPU: AMD Instinct MI250X
GPU memory: 68.7 GB
```

## Memory Optimizations (HARD-WON LESSONS)

The full CFDT model (154M params, T=16 dates, 256x256 patches) required ~90GB in fp32 — far exceeding MI250X's 64GB. After extensive debugging and profiling, we applied **5 simultaneous optimizations** to bring it to ~55GB.

### 1. bf16 Mixed Precision

Biggest single win. Applied in all stage 1-3 training loops:

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(batch)
    loss = criterion(output, target)
loss.backward()  # backward is outside autocast
```

Use `"cuda"` as device string, NOT `"hip"`.

### 2. Gradient Checkpointing

Applied to per-date encoder calls and change gate (most memory-intensive passes):

```python
from torch.utils.checkpoint import checkpoint
features = checkpoint(self.encoder, x_t, use_reentrant=False)
```

Trades ~30% more compute for ~40% less activation memory. Adds ~8 min/epoch in stage 1.

### 3. Reduced-Resolution Change Gate

The change gate processes O(T^2) date pairs (T=16 → 240 pairs). At full 256x256 resolution, intermediate features exceeded 30GB alone.

**Solution**: Downsample to 64x64 for change gate computation, bilinearly upsample output gate values back to 256x256. The gate is a smooth spatial mask, so resolution reduction has minimal quality impact.

```python
# In cfdt.py forward():
x_small = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
gate_small = self.change_gate(x_small)
gate = F.interpolate(gate_small, size=(256, 256), mode='bilinear', align_corners=False)
```

### 4. Target-Only Multi-Scale Features

Encoder produces 4-level feature pyramids per date. With T=16, keeping all = O(T * 4 levels).

**Solution**: Only keep full 4-level pyramid for the target date (needed for decoder skip connections). Non-target dates only keep the coarsest level (needed for TMT temporal attention).

Memory: O(T * 4) → O(1 * 4 + T * 1)

### 5. Chunked Change Gate Convolution

Process date pairs in chunks of 64 through the change gate CNN instead of all 240 at once:

```python
chunk_size = 64
outputs = []
for i in range(0, num_pairs, chunk_size):
    outputs.append(self.conv_layers(pairs[i:i+chunk_size]))
gate = torch.cat(outputs, dim=0)
```

### Memory Budget Summary

| Component | fp32 | After Optimization | Technique |
|-----------|------|-------------------|-----------|
| Parameters (154M) | ~0.6 GB | ~0.3 GB | bf16 |
| Encoder activations (16 dates) | ~24 GB | ~8 GB | Grad checkpoint + target-only |
| Change gate (240 pairs) | ~32 GB | ~6 GB | Reduced resolution + chunking |
| TMT activations | ~12 GB | ~6 GB | bf16 |
| Draft decoder | ~8 GB | ~4 GB | bf16 |
| Optimizer states | ~2.4 GB | ~2.4 GB | Stays fp32 |
| Gradients | ~0.6 GB | ~0.6 GB | Stays fp32 |
| **Total** | **~90 GB** | **~55 GB** | **Fits in 64 GB** |

## Rsync: Code Deployment (CRITICAL PITFALL)

```bash
# CORRECT: Exclude only the top-level /data directory
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '/data' \
    ./ lumi2:/scratch/project_465002500/vcloud/code/

# WRONG: Excludes ANY directory named "data" at ANY level
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude 'data' \
    ./ lumi2:/scratch/project_465002500/vcloud/code/
```

The wrong version silently excludes `src/vcloud/data/` (the Python dataset module), causing `ModuleNotFoundError: No module named 'vcloud.data'` on LUMI. This is extremely hard to debug because:
- Error only appears on LUMI, never locally
- `ls src/vcloud/` on LUMI shows no `data/` directory, but you might not think to check
- rsync output scrolls by quickly and the exclusion is easy to miss

**Always use the leading slash** in rsync exclude patterns for top-level directories.

## Submitting Jobs

```bash
ssh lumi2
cd /scratch/project_465002500/vcloud

# Full training (all 4 stages)
sbatch code/slurm/phase4_training/train_full_lumi.sbatch

# Skip completed stages
sbatch --export=ALL,START_STAGE=2 code/slurm/phase4_training/train_full_lumi.sbatch

# Export viewer data
sbatch code/slurm/phase5_viewer/export_viewer_lumi.sbatch
```

### Auto-Resume

Training scripts auto-resume from `latest.pt` if it exists. Handles:
- SLURM preemption (with `--requeue`)
- Walltime expiry
- Node failures

Saves two checkpoints per stage:
- `latest.pt` — every epoch (for resume)
- `best.pt` — on validation metric improvement (for inference)

## Installing Extra Python Packages

Container doesn't include everything (e.g., `tifffile`). Install to `/projappl/`:

```bash
srun --account=project_465002500 --partition=small-g --time=00:30:00 --ntasks=1 --pty bash
SIF="/appl/local/csc/soft/ai/images/pytorch_2.5.1_lumi_rocm6.2.4_flash-attn-3.0.0.post1.sif"
singularity exec "$SIF" pip install --target=/projappl/project_465002500/pylibs tifffile
```

Then add to PYTHONPATH in inner script.

## Training Results (as of Feb 16, 2026)

### Stage 0: Autoencoder — COMPLETE

| Metric | Value |
|--------|-------|
| Epochs | 50 |
| PSNR | **39.6 dB** |
| Wall time | ~12h |
| Parameters | 29.8M |
| Batch size | 2 |
| Time/epoch | ~25 min |
| Checkpoint | `outputs_full/stage0/best.pt` (342MB) |

### Stage 1: Draft Model (CFDT) — IN PROGRESS

| Metric | Value |
|--------|-------|
| Epoch | 46/100 |
| PSNR | **34.0 dB** (climbing) |
| Parameters | 154M |
| Batch size | 1 |
| Time/epoch | ~27 min |
| Job ID | 16082076 |

Curriculum learning:
- Epochs 0-29: `thin_only` (thin clouds only)
- Epochs 30-89: `mixed` (thin + thick)
- Epochs 90-99: `all` (all cloud types including opaque)

### Stages 2-3 — Pending

Submit after stage 1 completes:
```bash
sbatch --export=ALL,START_STAGE=2 code/slurm/phase4_training/train_full_lumi.sbatch
```

## Common Errors and Solutions

### 1. "RuntimeError: No HIP GPUs available"

**Cause**: Missing GPU library bindings in Singularity container.

**Fix**: Ensure BOTH are set:
```bash
export SINGULARITY_CONTAINLIBS="/usr/lib64/libcxi.so.1,/opt/rocm/lib/librocm_smi64.so.7"
srun singularity exec -B /opt/cray/libfabric ...
```

### 2. "torch.cuda.OutOfMemoryError"

**Cause**: Model too large without optimizations, or allocator fragmenting.

**Fix**:
1. Set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`
2. Apply all 5 memory optimizations
3. Reduce batch size to 1
4. Verify gradient checkpointing is enabled

### 3. "ModuleNotFoundError: No module named 'vcloud.data'"

**Cause**: rsync excluded `src/vcloud/data/` due to non-anchored `--exclude 'data'`.

**Fix**: Use `--exclude '/data'` (leading slash). Re-sync code.

### 4. "libfabric" / "CXI" errors

**Cause**: Missing `-B /opt/cray/libfabric` bind mount.

**Fix**: Add `-B /opt/cray/libfabric` to `singularity exec`.

### 5. WandB hanging at initialization

**Cause**: Network restrictions in Singularity containers.

**Fix**: `WANDB_MODE=disabled`. Optionally sync offline runs later:
```bash
wandb sync outputs_full/stage1/wandb/offline-run-* --entity digifarm --project vcloud
```

### 6. "ImportError: No module named 'tifffile'"

**Cause**: Not in CSC container.

**Fix**: `pip install --target=/projappl/project_465002500/pylibs tifffile` and add to PYTHONPATH.

### 7. Job sits in queue forever

**Cause**: Using `standard-g` or requesting too many resources.

**Fix**: Use `small-g` for single-GPU. Request only what you need:
```
#SBATCH --partition=small-g
#SBATCH --gpus=1
#SBATCH --mem=64G
```

### 8. Checkpoint loading fails after code changes

**Cause**: Model architecture changed between save and load.

**Fix**: Use `strict=False` in `load_state_dict()` for exploratory loads. Always version checkpoints with the code that produced them.
