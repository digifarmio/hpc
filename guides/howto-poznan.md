# How-To Guide: Poznan (Eagle)

Step-by-step recipes for common tasks on Poznan Eagle HPC (PSNC, Poland). H100 GPUs, Singularity containers, key-based SSH auth.

---

## Connect to Poznan

```bash
# ~/.ssh/config
Host poznan
    HostName eagle.man.poznan.pl
    User girish_digifarm
    IdentityFile ~/.ssh/Digifarm/poznan

# Connect (key-based, no password needed)
ssh poznan
```

Unlike Saga/Betzy (which use Sigma2 password auth), Poznan uses standard SSH key authentication.

---

## Transfer Code to Poznan

```bash
# From dev machine — sync entire Kvasir repo
rsync -avzh --progress \
    --exclude '.git' --exclude '.vscode' --exclude '*.pyc' \
    --exclude '__pycache__/' --exclude 'env/' --exclude 'venv/' \
    --exclude 'tmp/' --exclude 'output/' --exclude '/data' \
    /path/to/Kvasir/ poznan:/mnt/storage_3/home/girish_digifarm/pl0386-01/archive/dr/Kvasir
```

Or use the automated sync script that pushes to all three HPCs:

```bash
# From Kvasir repo root
bash scripts/sync_kvasir_repo_to_all_hpc.sh
```

This script uses `sshpass` for Sigma2 clusters and key-based auth for Poznan.

---

## Directory Structure

```
/mnt/storage_3/home/girish_digifarm/pl0386-01/archive/dr/
├── Kvasir/                              # Kvasir repo (synced, not a git clone)
│   └── src/workers/deep_resolver_v4/
│       └── ImageryProcessor/
│           ├── entry.py                 # Main entry point
│           ├── model_artifacts/         # Model weights + stats files
│           │   ├── sw3b4_n14_n4g2_o3zc_checkpoint_best.pth
│           │   ├── dataset_stats_naip2sen_v14_1m.json
│           │   └── ...
│           └── helpers/                 # Pipeline code
│               ├── infer_s2f.py         # Core inference engine
│               ├── dr_processor.py      # Full-tile DR pipeline
│               ├── pdr_processor.py     # AOI/subscription pipeline
│               └── ...
├── dr4saga1.sif                         # DR v4 Singularity container (~15 GB)
├── logs/                                # SLURM output logs
└── run_sr_*.sh                          # Generated SLURM job scripts
```

| Path | Use for |
|------|---------|
| `/mnt/storage_3/home/girish_digifarm/pl0386-01/archive/dr/` | All DR data, containers, scripts |
| `ImageryProcessor/tmp/` | Inference temp files (stacked bands, memmaps, output TIFFs) |

---

## GPU Partitions

| Partition | GPU Type | Nodes | GPUs Total | Max Time |
|-----------|----------|-------|------------|----------|
| **proxima** | NVIDIA H100 (4 per node) | 77 nodes (gpu13-gpu89) | 308 | 7 days |
| **tesla** | Mixed Tesla + H100 | 89 nodes | 386 | 7 days |
| **proxima-cpu** | None (CPU only) | 60 nodes | — | 7 days |

**Use `proxima` for DR inference.** Each node has 4× H100 GPUs with 80 GB VRAM and 64 CPUs.

**Important:** Singularity is only available on compute nodes, not login nodes. All container commands must run via SLURM (`sbatch` or `srun`).

---

## Run a DR Inference Job (Full Tile)

```bash
#!/bin/bash
#SBATCH --job-name=sr_33LWF_230626
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=proxima
#SBATCH --gpus=1
#SBATCH --output=dr/logs/slurm_sr_33LWF_230626-%j.out
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=morris@digifarm.io

export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export PG_URL=postgres://...
export AWS_REGION=eu-central-1

TILE=33LWF
S2_DATE="20230626"
OUTPUT_FILE="dr/Kvasir/src/workers/deep_resolver_v4/ImageryProcessor/tmp/sr_${TILE}_${S2_DATE}/S2x10_T${TILE}_${S2_DATE}_v8.tif"

# Skip if already processed (idempotent)
if [ -f "$OUTPUT_FILE" ]; then
    echo "DR file already exists, skipping."
    exit 0
fi

singularity exec --nv \
    --bind dr/Kvasir/src/workers/deep_resolver_v4/ImageryProcessor/:/mnt \
    --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    --env AWS_REGION=$AWS_REGION \
    dr/dr4saga1.sif \
    python /mnt/entry.py dr --date $S2_DATE --mgrs $TILE --server poznan
```

Submit:

```bash
sbatch run_sr_33LWF_230626.sh
```

Key notes:
- `--nv` enables NVIDIA GPU passthrough inside Singularity
- The container bind-mounts `ImageryProcessor/` to `/mnt` inside
- `entry.py dr` runs the full-tile Deep Resolution pipeline
- `--server poznan` tags Slack notifications with the cluster name
- Output: `S2x10_T<MGRS>_<DATE>_v8.tif` (10× super-resolution GeoTIFF)

---

## Run a DR Inference Job (AOI Mode)

```bash
singularity exec --nv \
    --bind dr/Kvasir/src/workers/deep_resolver_v4/ImageryProcessor/:/mnt \
    --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    --env AWS_REGION=$AWS_REGION \
    dr/dr4saga1.sif \
    python /mnt/entry.py dr --date 20230626 --aoi 18.5 54.3 19.0 54.7 --server poznan
```

AOI mode crops the input to the bounding box instead of processing the full MGRS tile. Much faster (~10 min vs 6+ hours).

---

## Post-Inference Cleanup

After a successful run, clean up temp files (stacked bands, memmaps) but keep the output:

```bash
OUTPUT_DIR="dr/Kvasir/src/workers/deep_resolver_v4/ImageryProcessor/tmp/sr_33LWF_20230626"
OUTPUT_FILE="S2x10_T33LWF_20230626_v8.tif"

if [ -d "$OUTPUT_DIR" ]; then
    find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 ! -name "$OUTPUT_FILE" -exec rm -rf {} +
    echo "Cleaned up temp files, kept $OUTPUT_FILE"
fi
```

---

## Start an Interactive GPU Session

```bash
srun --partition=proxima --gpus=1 --mem=64G --time=02:00:00 --pty bash
```

Inside the session:

```bash
singularity exec --nv \
    --bind /mnt/storage_3/home/girish_digifarm/pl0386-01/archive/dr/Kvasir/src/workers/deep_resolver_v4/ImageryProcessor/:/mnt \
    /mnt/storage_3/home/girish_digifarm/pl0386-01/archive/dr/dr4saga1.sif \
    python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA H100
```

---

## Monitor Jobs

```bash
squeue -u girish_digifarm
squeue -u girish_digifarm --format="%.10i %.9P %.20j %.8T %.10M %.6D %R"

# Post-mortem stats
sacct -j JOBID --format=JobID,Elapsed,MaxRSS,MaxVMSize,State

# Follow live output
tail -f dr/logs/slurm_sr_33LWF_230626-JOBID.out

# Cancel
scancel JOBID
```

---

## Performance Benchmarks

Based on real DR v4/v5 runs on Poznan H100:

| Metric | Full Tile (109,800²) | AOI (~10,000²) |
|--------|---------------------|----------------|
| GPU inference | ~1h 40m | ~5-10 min |
| PostProc block writing | ~5-7 hours | ~5 min |
| Total wall time | 6-8 hours | 10-20 min |
| Peak memory | ~47-99 GB | ~8-16 GB |
| Output size | ~750 MB - 2 GB | ~30-100 MB |

**H100 advantage:** 80 GB VRAM allows larger batch sizes. The 755 GB system RAM means full tiles process without sub-tiling (the pipeline skips the split-merge path when RAM ≥ 700 GB).

---

## Common Errors and Fixes

| Error | Fix |
|-------|-----|
| `singularity: command not found` on login | Run via SLURM (`srun` or `sbatch`); Singularity is only on compute nodes |
| PostProc killed at 95% (100% nodata output) | Increase `--time` in SLURM script; code now has dynamic timeout scaling with image size |
| QC FAIL: 100% nodata | Pipeline now blocks upload on QC failure; check if PostProc was force-killed in the log |
| OOM at ~99 GB | For very large tiles, the pipeline auto-splits when RAM < 700 GB. On H100 nodes (755 GB RAM) it processes whole — reduce `--mem` to trigger splitting if needed |
| `Missing or insufficient output stats for s2_lr_6` (repeated) | Fixed in latest code — early fallback to input_norm at startup instead of per-batch warning |
| Job pending for hours on `proxima` | Try `tesla` partition as fallback (also has H100 nodes) |
