# Betzy HPC Guide

Betzy is another NRIS/Sigma2 cluster in Norway, used as a backup when SAGA's GPU nodes were unavailable. It shares the same account and billing quota as SAGA.

## Cluster Details

| Item | Value |
|------|-------|
| Host | `betzy.sigma2.no` |
| SSH alias | `betzy` |
| User | `digifarm` |
| SLURM account | `nn12037k` (shared with SAGA) |
| GPU partition | `accel` |
| Max job time | 7 days |

## SSH Access

Same authentication as SAGA (NRIS one-time passwords via authenticator app).

```bash
Host betzy
    HostName betzy.sigma2.no
    User digifarm
```

## Environment Setup

Same module system and venv approach as SAGA:

```bash
module purge
module load Python/3.10.4-GCCcore-11.3.0
source /cluster/work/users/digifarm/vcloud/v0/venv/bin/activate
export PYTHONPATH="/cluster/work/users/digifarm/vcloud/v0/src:${PYTHONPATH:-}"
```

The same broken module PyTorch issue from SAGA applies here — use pip-installed PyTorch in the venv.

## Job Script

A ready-to-use script is at `slurm/phase4_training/train_full_betzy.sbatch`:

```bash
cd /cluster/work/users/digifarm/vcloud/v0
sbatch slurm/phase4_training/train_full_betzy.sbatch

# Resume from stage 2
sbatch --export=ALL,START_STAGE=2 slurm/phase4_training/train_full_betzy.sbatch
```

## When to Use Betzy

Betzy is primarily a **CPU-focused** cluster with GPU nodes in high demand.

**Use for:**
- CPU-heavy preprocessing (data conversion, statistics)
- Backup when SAGA GPU nodes are down
- Jobs needing many CPU cores but no GPU

**Do NOT use for:**
- Iterative GPU work (queue times: days to weeks)
- Primary training runs (use LUMI)
- Anything time-sensitive requiring GPU

## Issues

### Very Long GPU Queue Times

GPU queue times on Betzy ranged from days to weeks. By the time your job starts, you've likely moved on to another approach or cluster. This made Betzy essentially unusable for iterative GPU development.

### Shared Billing with SAGA

Same `nn12037k` account — GPU-hours consumed on either cluster count against the same quota. Monitor with `cost -p nn12037k`.

### Same Pitfalls as SAGA

- Module PyTorch is broken → use venv
- Always `module purge` before loading
- `/cluster/work/` has no backup
- No container approach (more brittle than LUMI)

### Filesystem

Betzy shares the same `/cluster/work/` path structure as SAGA (shared NRIS filesystem). Data prepared on SAGA should be accessible from Betzy. Verify with `ls` before assuming.
