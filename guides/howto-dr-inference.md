# How-To Guide: Running DR Inference on HPC

Cross-cluster guide for running the Deep Resolution (DR) v4/v5 super-resolution pipeline on Saga, Betzy, and Poznan. Covers the full-tile and AOI workflows, container setup, SLURM patterns, and hard-won lessons from production failures.

---

## Overview

The DR pipeline takes 10-band Sentinel-2 imagery at 10m resolution and produces a 10× super-resolved output at 1m. It runs inside a Singularity container (`dr4saga1.sif`) with GPU acceleration.

Two pipeline modes:
- **`dr` (full tile):** Downloads an entire MGRS tile, runs inference, uploads the result to S3. Used for bulk tile processing.
- **`pdr` (AOI/subscription):** Polls a database for pending imagery requests, crops to subscription AOI, runs inference, generates derivatives (NDVI, EVI, RGB, NIR), uploads to S3. Used for on-demand customer requests.

### Current Model (DRV5.1)

| Component | Value |
|-----------|-------|
| Model | `Swin2SRa_L1hf` (srt3 implementation) |
| Checkpoint | `sw3b4_n14_n4g2_o3zc_checkpoint_best.pth` |
| Stats file | `dataset_stats_naip2sen_v14_1m.json` |
| Output bands | 4 RGBN + 1 quality mask = 5 bands |
| Output dtype | int16 |
| Upscale factor | 10× |
| Compression | LZW with predictor |
| Output format | COG (Cloud-Optimized GeoTIFF) with 512×512 tiles + overviews |

---

## Cluster Comparison for DR

| | Poznan | Betzy | Saga |
|---|---|---|---|
| **GPU** | H100 80GB | A100 40GB | A100 (a100 partition) |
| **System RAM** | 755 GB | 242 GB | Varies |
| **GPU partition** | `proxima` | `accel` | `a100` |
| **SLURM account** | (none needed) | `nn12037k` | `nn12037k` |
| **Auth** | SSH key | Password (sshpass) | Password (sshpass) |
| **User** | `girish_digifarm` | `warachi` | `warachi` |
| **Base path** | `/mnt/storage_3/.../pl0386-01/archive/dr/` | `/cluster/work/users/warachi/dr/` | `/cluster/work/users/warachi/dr/` |
| **Container** | `dr4saga1.sif` | `dr4saga1.sif` | `dr4saga1.sif` |
| **Best for** | Large full tiles (H100 + 755 GB RAM) | Medium tiles, AOI jobs | Medium tiles, AOI jobs |

---

## Container Setup

The same container image (`dr4saga1.sif`, ~15 GB) is used on all three clusters. It includes PyTorch, rasterio, GDAL, scipy, and all DR dependencies.

```bash
# Container is pre-built and stored alongside the Kvasir repo:
ls -lh /path/to/dr/dr4saga1.sif
# -rw-r--r-- 1 user group 15G Feb  6 12:00 dr4saga1.sif
```

The container bind-mounts the `ImageryProcessor/` directory to `/mnt`:

```bash
singularity exec --nv \
    --bind /path/to/ImageryProcessor/:/mnt \
    /path/to/dr4saga1.sif \
    python /mnt/entry.py dr --date 20230626 --mgrs 33LWF --server poznan
```

This means model artifacts, temp files, and output all live on the host filesystem under `ImageryProcessor/`.

---

## Code Deployment

Code lives in the Kvasir repo. It is not a git clone on HPC — it's synced via rsync:

```bash
# Sync to all three HPCs at once
bash scripts/sync_kvasir_repo_to_all_hpc.sh

# Or sync to one cluster manually
rsync -avzh --progress \
    --exclude '.git' --exclude '*.pyc' --exclude '__pycache__/' \
    --exclude 'env/' --exclude 'venv/' --exclude 'tmp/' \
    /path/to/Kvasir/ poznan:/mnt/storage_3/.../dr/Kvasir
```

---

## SLURM Job Scripts

### Template: Full Tile Inference

```bash
#!/bin/bash
#SBATCH --job-name=sr_<MGRS>_<YYMMDD>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=<GPU_PARTITION>     # proxima (Poznan), accel (Betzy), a100 (Saga)
#SBATCH --gpus=1
#SBATCH --account=<ACCOUNT>            # nn12037k (Sigma2), omit for Poznan
#SBATCH --output=dr/logs/slurm_sr_<MGRS>_<YYMMDD>-%j.out
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=morris@digifarm.io

set -euo pipefail

export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export PG_URL=postgres://...
export AWS_REGION=eu-central-1

TILE=<MGRS>
S2_DATE="<YYYYMMDD>"
DR_BASE="<BASE_PATH>/Kvasir/src/workers/deep_resolver_v4/ImageryProcessor"
OUTPUT="${DR_BASE}/tmp/sr_${TILE}_${S2_DATE}/S2x10_T${TILE}_${S2_DATE}_v8.tif"

# Idempotent: skip if output already exists
if [ -f "$OUTPUT" ]; then
    echo "Output already exists: $OUTPUT"
    exit 0
fi

singularity exec --nv \
    --bind "${DR_BASE}:/mnt" \
    --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    --env AWS_REGION=$AWS_REGION \
    <BASE_PATH>/dr4saga1.sif \
    python /mnt/entry.py dr --date $S2_DATE --mgrs $TILE --server <CLUSTER_NAME>

# Optional: cleanup temp files, keep output
dr_dir="$(dirname "$OUTPUT")"
dr_basename="$(basename "$OUTPUT")"
if [ -d "$dr_dir" ]; then
    find "$dr_dir" -mindepth 1 -maxdepth 1 ! -name "$dr_basename" -exec rm -rf {} +
fi
```

### Template: AOI Inference

```bash
# Same SBATCH headers, but shorter time (AOI is much faster)
#SBATCH --time=02:00:00
#SBATCH --mem=32G

singularity exec --nv \
    --bind "${DR_BASE}:/mnt" \
    --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    --env AWS_REGION=$AWS_REGION \
    <BASE_PATH>/dr4saga1.sif \
    python /mnt/entry.py dr --date 20230626 --aoi 18.5 54.3 19.0 54.7 --server <CLUSTER_NAME>
```

### Template: Post-Processing / Finalization

Finalization scripts are chained with `--dependency` to run after inference completes:

```bash
#!/bin/bash -e
#SBATCH --job-name=fin_<MGRS>_v<VERSION>
#SBATCH --nodes=4                        # Betzy normal partition requires 4 nodes
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --account=nn12037k
#SBATCH --dependency=afterany:<INFERENCE_JOBID>
#SBATCH --output=dr/logs/slurm_fin_<MGRS>-%j.out

# Log billing hours, wall clock, and generate digi-map preview URLs
# Clean up inference temp directory
```

Submit with dependency:

```bash
INFERENCE_JOB=$(sbatch --parsable run_sr_33LWF_230626.sh)
sbatch --dependency=afterany:$INFERENCE_JOB run_sr_post_33LWF.sh
```

---

## Environment Variables

All DR jobs require these env vars passed into the container:

| Variable | Purpose |
|----------|---------|
| `AWS_ACCESS_KEY_ID` | S3 access for reading Sentinel-2 COGs and uploading results |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key |
| `AWS_REGION` | AWS region (`eu-central-1`) |
| `PG_URL` | PostgreSQL connection string for metadata DB |

These are set via `--env` flags in the `singularity exec` command. They are **not** baked into the container.

---

## Pipeline Architecture

```
entry.py dr --date YYYYMMDD --mgrs TILE
  │
  ├─ dr_processor.py: process_dr()
  │    │
  │    ├─ dr.py: make_dr() → infer()
  │    │    ├─ Stack Sentinel-2 bands (download from S3, reproject to UTM, stack 10+1 bands)
  │    │    ├─ Validate stacked data
  │    │    ├─ Area check: >40k ha AND RAM <700 GB → split into 2048×2048 tiles
  │    │    ├─ run_inference_pipeline() [multiprocess: CPU preproc → GPU infer → PostProc writer]
  │    │    └─ Merge tiles (if split)
  │    │
  │    ├─ Clip by AOI (if aoi mode)
  │    ├─ Build COG overviews (gdaladdo)
  │    ├─ Quality checks (7 checks: tiling, overviews, values, bands, visual, timing, deploy)
  │    ├─ Upload to S3 (blocked if QC fails)
  │    ├─ Record metadata to DB
  │    └─ Send Slack notifications
  │
  └─ [DONE]
```

### Inference Engine (infer_s2f.py)

The core engine uses 3 worker types in a multiprocess pipeline:

1. **CPU PreProc workers** (`num_preproc_cpus`): Extract patches from stacked image, normalize, save to `.npz` batch files on disk
2. **GPU Infer workers** (`num_gpus`): Load batches, run through Swin2SRa model, output super-resolved patches
3. **PostProc writer** (single process): Accumulate patches into memmap arrays, then write block-by-block to output GeoTIFF

For full tiles, PostProc is the bottleneck — writing 46,000+ blocks from memmaps to a compressed GeoTIFF takes 5-7 hours.

---

## Full Tile vs AOI: Size Comparison

| | AOI (small crop) | Full MGRS Tile |
|---|---|---|
| Input | ~1,000 × 1,000 px | 10,980 × 10,980 px |
| Output at 10× | ~10,000 × 10,000 px | 109,800 × 109,800 px |
| Output blocks (512²) | ~400 | 46,225 |
| Inference time | 5-10 min | 1-2 hours |
| PostProc write time | 5 min | **5-7 hours** |
| Peak memory | 8-16 GB | 47-99 GB |
| Memmap usage | In-memory (no memmap) | Disk-backed memmaps (~190 GB) |

The pipeline automatically decides whether to use memmaps based on output size:
- < 200M pixels → in-memory accumulation (fast)
- ≥ 200M pixels → disk-backed memmaps (necessary but slow I/O)

For very large tiles (>40k hectares) on nodes with < 700 GB RAM, the pipeline splits the input into 2048×2048 sub-tiles, processes each independently, and merges via VRT.

---

## Quality Checks

Every DR run ends with 7 automated quality checks:

| Check | What it validates |
|-------|-------------------|
| Tiling | 512×512 internal tiles (COG compliance) |
| Overviews | ≥4 pyramid levels present |
| Values | Mean/std within range, <95% nodata, no all-zero output |
| Bands | Expected count (5: RGBN + quality mask), correct dtype |
| Visual | Shannon entropy >0 (catches blank outputs) |
| Timing | GPU <2h, CPU <1h, total <3h |
| Deploy | File size <10 GB, compressed, COG format, throughput >50 ha/min |

**QC FAIL blocks upload.** If quality checks fail, the pipeline raises an error, sends a Slack notification with the QC report, and refuses to upload corrupt data to S3 or mark imagery as processed in the database.

---

## Lessons Learned (Hard Failures)

### 1. PostProc Timeout Killed at 95% → 100% Nodata Output

**What happened:** A full-tile job on Poznan H100 completed inference (153,664 patches in 1h39m) but PostProc was force-terminated at 95% block completion (43,795/46,225 blocks) because the hardcoded 25,000s timeout expired. The rasterio file handle was never closed, leaving a corrupt all-zeros GeoTIFF. The pipeline reported success and uploaded blank data to S3.

**Fix:** PostProc timeout now scales dynamically with image size: `max(25000, n_blocks × 2)`. For a 109,800² image, the timeout is ~92,000s (25.7h). When PostProc is force-terminated, the pipeline now marks `pipeline_failed = True` and raises a RuntimeError instead of silently succeeding.

**Lesson:** Never use hardcoded timeouts for operations that scale with input size. Always validate output before declaring success.

### 2. Batch Size Too Large → PyTorch INT_MAX Overflow

**What happened:** `TILE_BATCH_SIZE=64` on A100-40GB produced tensors with 2.21 billion elements, exceeding PyTorch's `INT_MAX` limit (2.15 billion). Job crashed with a cryptic CUDA error.

**Fix:** Reduced `TILE_BATCH_SIZE` to 12 (413M elements, well within limits).

**Lesson:** PyTorch's `aten::index` has a 2^31 element limit. Calculate `batch × channels × height × width × upscale²` before choosing batch size.

### 3. s2_lr_6 Stats Warning Spam (12,000+ Warnings)

**What happened:** The output stats file has 4 bands but `s2_lr_6_mode=output_norm` needs 10 bands. The code caught the IndexError per-batch and fell back, but logged 12,806 identical warnings — one per batch.

**Fix:** Early detection at pipeline startup. If output stats lack enough bands, downgrade to `input_norm` once with a single warning.

### 4. No Slack Notification for Post-Processing Errors

**What happened:** Errors during S3 upload, GeoTIFF processing, derivative generation, or DB writes were logged but never sent to Slack. The team had no visibility into production failures.

**Fix:** Wrapped all post-inference processing in try/except blocks that send error details to Slack before re-raising.

### 5. QC FAIL Didn't Block Upload

**What happened:** A 100% nodata file passed through QC (which logged a warning) and was uploaded to S3 as a valid product.

**Fix:** QC FAIL now raises a RuntimeError, blocking S3 upload and DB updates. In the PDR pipeline, QC was also moved to run *before* S3 upload (it previously ran after).

---

## Operational Checklist

Before submitting a DR job:

- [ ] Code is synced to the cluster (`scripts/sync_kvasir_repo_to_all_hpc.sh`)
- [ ] Container exists at the expected path (`dr4saga1.sif`)
- [ ] AWS credentials are valid (test with a small AOI first)
- [ ] `PG_URL` is reachable from compute nodes
- [ ] SLURM `--time` is sufficient (15h for full tiles, 2h for AOI)
- [ ] `--mem=128G` for full tiles (some tiles peak at 99 GB)
- [ ] Log directory exists (`mkdir -p dr/logs`)
- [ ] Check if output already exists (the skip-if-exists pattern prevents wasted compute)

After a job completes:

- [ ] Check Slack for QC report (pass/warn/fail)
- [ ] Verify output in S3 (or check why upload was blocked)
- [ ] Clean up temp files if not auto-cleaned
- [ ] Check `sacct` for actual resource usage to tune future jobs

---

## Quick Reference: Entry Point CLI

```bash
# Full tile mode
python /mnt/entry.py dr --date YYYYMMDD --mgrs TILE [--server NAME] [--s2_id N] [--product_id ID]

# AOI mode
python /mnt/entry.py dr --date YYYYMMDD --aoi MIN_LON MIN_LAT MAX_LON MAX_LAT [--server NAME]

# PDR mode (polls DB for pending imagery)
python /mnt/entry.py pdr [--server NAME]
```

| Flag | Purpose |
|------|---------|
| `--date` | Sentinel-2 acquisition date (YYYYMMDD) |
| `--mgrs` | MGRS tile ID (e.g., 33LWF). Mutually exclusive with `--aoi` |
| `--aoi` | Bounding box (min_lon min_lat max_lon max_lat). Mutually exclusive with `--mgrs` |
| `--server` | Cluster name for Slack reporting (e.g., `poznan`, `betzy`, `saga`) |
| `--s2_id` | Sentinel-2 scene index (0 or 1) when multiple scenes exist for the date |
| `--product_id` | Exact S2 COG product folder (e.g., `S2B_19JCL_20250402_0_L2A`) |
