# General HPC Tips & Cross-Cluster Patterns

Lessons that apply across SAGA, Betzy, and LUMI for the vcloud project.

## SSH Configuration

```bash
# ~/.ssh/config
Host saga
    HostName saga.sigma2.no
    User digifarm
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 7d

Host betzy
    HostName betzy.sigma2.no
    User digifarm
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 7d

Host lumi2
    HostName lumi.csc.fi
    User digifarm
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 7d
```

Create socket directory: `mkdir -p ~/.ssh/sockets`

With `ControlPersist 7d`, one SSH connection stays open for 7 days. All subsequent SSH/SCP/rsync commands reuse it (instant connection, no re-auth).

## Code Sync via rsync

Neither cluster has git credentials. All code transfers use rsync from the dev machine:

```bash
# To LUMI (CRITICAL: use '/data' not 'data' — see lumi.md)
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '/data' \
    ./ lumi2:/scratch/project_465002500/vcloud/code/

# To SAGA
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '/data' \
    ./ saga:/cluster/work/users/digifarm/vcloud/v0/
```

**Always exclude:**
- `.git/` — unnecessary on HPC
- `__pycache__/` — regenerated on first run
- `/data` — multi-GB data directories (leading slash = top-level only!)

**Never exclude without leading slash:**
- `data` without slash also removes `src/vcloud/data/` (Python source code!)

## Data Transfer Between Clusters

```bash
# LUMI → local (e.g., viewer data export)
ssh lumi2 'tar cf - -C /scratch/project_465002500/vcloud viewer_data_full' | tar xf -

# SAGA → LUMI (via local machine as relay, for dataset)
ssh saga 'tar cf - -C /cluster/work/users/digifarm/vcloud/v0 data/sen12mscrts_real/prepared' | \
    ssh lumi2 'tar xf - -C /scratch/project_465002500/vcloud/'
```

**Caution**: Don't transfer checkpoints between GPU architectures without testing. The format is portable, but bf16 precision behavior may differ between P100 and MI250X.

## SLURM Essentials

### Daily Commands

```bash
squeue -u digifarm                           # Your jobs
squeue -u digifarm --format="%.10i %.9P %.20j %.8T %.10M %.6D %R"  # Detailed
sinfo -p PARTITION --format="%n %G %t %C"    # Node/GPU status
sacct -j JOBID --format=JobID,Elapsed,MaxRSS,MaxVMSize,State  # Post-mortem
scancel JOBID                                # Cancel one
tail -f logs/vcloud_full_JOBID.out           # Follow output
```

### Job Script Best Practices

```bash
set -euo pipefail             # Fail fast on errors
#SBATCH --requeue             # Auto-resubmit on preemption
#SBATCH --output=logs/%x_%j.out   # %x = job name, %j = job ID
#SBATCH --error=logs/%x_%j.err
```

- Always `mkdir -p` for log and output directories
- Add echo statements between steps to track progress in logs
- Never use `-i` flag (interactive) in batch scripts
- Request only what you need (shorter `--time` = higher scheduling priority)

### Auto-Resume Pattern

```bash
RESUME_ARG=""
if [ -f "$STAGE_DIR/latest.pt" ]; then
    echo "Resuming from $STAGE_DIR/latest.pt"
    RESUME_ARG="--resume $STAGE_DIR/latest.pt"
fi
python3 scripts/train.py --stage $STAGE $RESUME_ARG
```

Save `latest.pt` every epoch (resume) and `best.pt` on validation improvement (inference).

### Stage Skipping

```bash
START_STAGE="${START_STAGE:-0}"
for STAGE in 0 1 2 3; do
    [ "$STAGE" -lt "$START_STAGE" ] && continue
    # ... run stage
done

# Usage:
sbatch --export=ALL,START_STAGE=2 train.sbatch
```

## Container vs Venv

| Aspect | Container (LUMI) | Venv (SAGA/Betzy) |
|--------|-----------------|-------------------|
| Reproducibility | Excellent | Fragile |
| Setup time | Minutes | Hours of debugging |
| Dependency conflicts | None | Common |
| GPU driver compat | Container handles it | Module system broken |
| Debugging | Exec into container | Direct shell |

**Always prefer containers when available.** LUMI's CSC containers saved countless hours.

## Monitoring Training Without WandB

WandB is disabled on LUMI (network restrictions in containers). Monitor via:

```bash
# Tail job output
tail -f /scratch/project_465002500/vcloud/logs/vcloud_full_JOBID.out

# Check latest checkpoint metrics
python3 -c "
import torch
c = torch.load('outputs_full/stage1/latest.pt', map_location='cpu')
print(f'Epoch {c[\"epoch\"]}, Loss {c.get(\"best_val_loss\",\"?\")}, PSNR {c.get(\"best_val_psnr\",\"?\")}')
"
```

## Choosing the Right Cluster

| Situation | Use | Why |
|-----------|-----|-----|
| Full model (>16GB VRAM) | **LUMI** | Only cluster with enough VRAM |
| Quick small-model iteration | SAGA | Shorter queues (when working) |
| Data download & preparation | SAGA | Good network, existing data |
| CPU preprocessing | Betzy | Less contention |
| Reliable long runs | **LUMI** | Most stable, biggest budget |

## Common Mistakes We Made (Ordered by Pain)

1. **rsync `--exclude 'data'` without leading slash** — silently broke imports on LUMI. Took days to debug because the error only appeared on LUMI, not locally. Always use `--exclude '/data'`.

2. **Forgot `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`** — OOM at ~40GB instead of ~60GB on MI250X. The memory allocator fragments badly without this.

3. **Forgot `-B /opt/cray/libfabric` in container bindings** — "No HIP GPUs available" error. Easy to miss, hard to debug if you don't know to look for it.

4. **Used SAGA module PyTorch** — Contained only CUDA stubs, not runtime libraries. Wasted a full day debugging `torch.cuda` failures.

5. **Left WandB enabled in LUMI container** — Job hung for 30+ minutes at startup trying to connect, then timed out. Set `WANDB_MODE=disabled`.

6. **Tried full model on P100 (16GB)** — Instant OOM. Should have calculated memory requirements first. Full model needs ~55GB even with all optimizations.

7. **Forgot `module purge` on SAGA** — Cryptic library conflicts. Always purge before loading.

8. **Assumed cluster work dirs persist forever** — Both SAGA and LUMI have purge policies for inactive files. Always keep important data backed up.

9. **Tried to apply memory optimizations one at a time** — Each alone wasn't enough. Needed all 5 simultaneously to fit in 64GB.

10. **Didn't test inner script outside of SLURM** — Could have caught path/import issues in an interactive session instead of waiting in the job queue.
