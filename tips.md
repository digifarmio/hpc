# General HPC Tips & Cross-Cluster Pitfalls

Lessons learned the hard way across LUMI, Saga, and Betzy.

## SSH Configuration

```bash
# ~/.ssh/config
Host saga
    HostName saga.sigma2.no
    User myuser
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 7d

Host betzy
    HostName betzy.sigma2.no
    User myuser
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 7d

Host lumi
    HostName lumi.csc.fi
    User myuser
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 7d
```

Create socket directory: `mkdir -p ~/.ssh/sockets`

With `ControlPersist 7d`, one SSH connection stays open for 7 days. All subsequent SSH/SCP/rsync commands reuse it (instant connection, no re-auth).

## Code Sync via rsync

HPC clusters often lack git credentials. rsync from the dev machine is the simplest approach:

```bash
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '/data' \
    ./ cluster:/path/to/project/
```

**Always exclude:**
- `.git/` -- unnecessary on HPC
- `__pycache__/` -- regenerated on first run
- `/data` -- multi-GB data directories (leading slash = top-level only!)

**Never exclude without leading slash:**
- `data` without slash also removes `src/mypackage/data/` (Python source code!)

This rsync pitfall is one of the hardest bugs to diagnose -- the error only appears on the cluster (missing Python module), never locally.

## Data Transfer Between Clusters

```bash
# Cluster A -> local
ssh clusterA 'tar cf - -C /path/to/project data_dir' | tar xf -

# Cluster A -> Cluster B (via local machine as relay)
ssh clusterA 'tar cf - -C /path data/' | ssh clusterB 'tar xf - -C /path/'
```

**Caution**: Don't transfer checkpoints between GPU architectures without testing. The format is portable, but bf16 precision behavior may differ between P100 and MI250X.

## SLURM Essentials

### Daily Commands

```bash
squeue -u $USER                                              # Your jobs
squeue -u $USER --format="%.10i %.9P %.20j %.8T %.10M %.6D %R"  # Detailed
sinfo -p PARTITION --format="%n %G %t %C"                    # Node/GPU status
sacct -j JOBID --format=JobID,Elapsed,MaxRSS,MaxVMSize,State  # Post-mortem
scancel JOBID                                                # Cancel one
tail -f logs/jobname_JOBID.out                               # Follow output
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
if [ -f "$OUTPUT_DIR/latest.pt" ]; then
    echo "Resuming from $OUTPUT_DIR/latest.pt"
    RESUME_ARG="--resume $OUTPUT_DIR/latest.pt"
fi
python3 scripts/train.py $RESUME_ARG
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

| Aspect | Container | Venv |
|--------|-----------|------|
| Reproducibility | Excellent | Fragile |
| Setup time | Minutes | Hours of debugging |
| Dependency conflicts | None | Common |
| GPU driver compat | Container handles it | Module system often broken |
| Debugging | Exec into container | Direct shell |

**Always prefer containers when available.** LUMI's CSC containers saved countless hours compared to the bare venv approach on Saga/Betzy.

## Monitoring Training Without WandB

WandB may not work inside Singularity containers on LUMI (network restrictions). Set `WANDB_MODE=disabled` and monitor via:

```bash
# Tail job output
tail -f logs/training_JOBID.out

# Check latest checkpoint metrics
python3 -c "
import torch
c = torch.load('outputs/latest.pt', map_location='cpu')
print(f'Epoch {c[\"epoch\"]}, Loss {c.get(\"best_val_loss\",\"?\")}, Metric {c.get(\"best_val_metric\",\"?\")}')
"
```

## Choosing the Right Cluster

| Situation | Use | Why |
|-----------|-----|-----|
| Full model (>16GB VRAM) | **LUMI** | Only cluster with enough VRAM (64GB MI250X) |
| Quick small-model iteration | Saga | Shorter queues (when GPU nodes are up) |
| Data download & preparation | Saga | Good network, GDAL CLI available |
| I/O-bound batch processing | Betzy `preproc` | Less contention, good scheduling |
| CPU preprocessing | Betzy | 128 cores/node, 242GB RAM |
| Reliable long GPU runs | **LUMI** | Most stable, largest budget |

## Common Mistakes (Ordered by Pain)

1. **rsync `--exclude 'data'` without leading slash** -- silently removes Python source subdirectories named `data/`. Took days to debug because the error only appeared on HPC. Always use `--exclude '/data'`.

2. **Forgot `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`** on LUMI -- OOM at ~40GB instead of ~60GB on MI250X. The memory allocator fragments badly without this.

3. **Forgot `-B /opt/cray/libfabric` in container bindings** on LUMI -- "No HIP GPUs available" error. Easy to miss, hard to debug.

4. **Used module-provided PyTorch on Saga** -- contained only CUDA stubs, not runtime libraries. Wasted a full day debugging `torch.cuda` failures. Use pip-installed PyTorch in a venv instead.

5. **Left WandB enabled in LUMI container** -- job hung for 30+ minutes at startup trying to connect. Set `WANDB_MODE=disabled`.

6. **Used GradScaler with bfloat16** -- skipped 75-87% of optimizer steps due to false-positive inf detections. GradScaler is only for float16. bfloat16 has the same dynamic range as float32 and needs no scaling.

7. **Forgot `module purge` on Saga** -- cryptic library conflicts from leftover module state.

8. **Assumed cluster work dirs persist forever** -- Saga, Betzy, and LUMI all have purge policies for inactive files (42-90 days). Always back up important data.

9. **Tried to apply memory optimizations one at a time** -- for large models on MI250X 64GB, all optimizations (bf16, gradient checkpointing, reduced resolution, chunked processing) must be applied simultaneously.

10. **Didn't test inner script outside of SLURM** -- could have caught path/import issues in an interactive `srun --pty bash` session instead of waiting hours in the job queue.

11. **Used `--mem` on Betzy's `normal` partition** -- crashes the job. Betzy allocates full nodes automatically on `normal`; remove `--mem` entirely.

12. **Didn't cap thread counts on Betzy** -- `os.cpu_count()` returns 256 (hyperthreads), breaking tools like tippecanoe that size internal structures on CPU count. Cap at 128.

13. **Used Lustre for temp-file-heavy tools** -- tippecanoe, SQLite WAL mode, and similar tools deadlock on Lustre. Use `/dev/shm` (126GB on Betzy) instead.

14. **PROJ_DATA mismatch in containers** -- system proj.db is outdated; rasterio/pyproj produce silently wrong coordinates. Always set `PROJ_DATA` to the container's rasterio proj_data path.
