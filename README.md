# HPC Usage Guide for Geospatial ML Pipelines

Practical guide for running large-scale geospatial pipelines on European HPC systems, covering the **romcrop** project (TESSERA embedding generation, crop classification), the **Sure2** project (satellite-based crop emergence insurance), and the **EMB** project (CDL crop segmentation from foundation model embeddings).

This guide covers real workflows, real errors, and real fixes. It is not a generic HPC tutorial.

---

## Systems Covered

| System | Country | Operator | GPUs | Container Runtime | Internet from Compute |
|--------|---------|----------|------|-------------------|-----------------------|
| [LUMI](LUMI.md) | Finland | CSC | AMD MI250X (ROCm) | `singularity` | Yes |
| [Saga](SAGA.md) | Norway | NRIS/Sigma2 | None used | `apptainer` | No |
| [Betzy](betzy.md) | Norway | NRIS/Sigma2 | NVIDIA A100 (CUDA) | `apptainer` | No |

## Quick Reference

### LUMI

```
Account:   project_465002500
Scratch:   /scratch/project_465002500/romcrop
Projappl:  /projappl/project_465002500/venvs/romcrop
SSH:       ssh lumi2
GPU:       standard-g partition (4x MI250X = 8 GCDs per node)
CPU:       small partition (single-node), standard (multi-node)
Python:    module load cray-python/3.11.7
PyTorch:   module load pytorch/2.7  (ROCm container via SING_IMAGE)
```

### Saga

```
Account:   nn12037k
Work:      /cluster/work/users/digifarm/sure/v0  (PURGED - 42 day policy)
SSH:       ssh saga
Container: apptainer build/exec (migrated from singularity)
GDAL:      CLI tools available (unlike LUMI)
```

## Guide Structure

1. **[LUMI.md](LUMI.md)** -- Most detailed guide. Covers the full TESSERA embedding generation pipeline: GPU inference with ROCm containers, two-environment architecture (venv + singularity), array jobs, and every error we hit.

2. **[SAGA.md](SAGA.md)** -- Saga-specific guide. Covers container building with apptainer, the no-internet constraint, GDAL CLI availability, and the Saga pipeline orchestrator.

3. **[sure2.md](sure2.md)** -- Sure2 crop emergence insurance engine. Covers Apptainer containerization, I/O-bound S2 COG processing, ERA5-Land/CHIRPS pre-loading, resumable batch processing, and 10 production bugs with fixes.

4. **[COMMON_PITFALLS.md](COMMON_PITFALLS.md)** -- Cross-system lessons learned. Covers env var override bugs, download timeouts, file extension mismatches, CatBoost label handling, memory estimation, disk purge policies, and SLURM array job management.

5. **[alfa-ml-inference.md](alfa-ml-inference.md)** -- ALFA land cover project. Covers CPU-only ML inference at scale on Betzy/Saga, in-job 32-shard parallelism, S3 reads from Norway, COG block optimization (48min→1min), mosaic building, XYZ tile generation, and Lustre filesystem pitfalls.

### EMB Project (CDL Crop Segmentation)

6. **[emb-betzy.md](emb-betzy.md)** -- Betzy guide for EMB. Storage layout (project vs work space), container PROJ mismatch, SLURM patterns for data prep (Phase 3) and DDP training (Phase 4) on 4x A100.

7. **[emb-lumi.md](emb-lumi.md)** -- LUMI guide for EMB. AMD MI250X differences, ROCm container setup (no --rocm flag!), SINGULARITY_BIND, DDP on 8 GCDs, deps venv on projappl.

8. **[emb-errors.md](emb-errors.md)** -- Comprehensive error catalog for EMB. Every bug we hit: GradScaler+bfloat16 (75% steps skipped), BatchNorm corruption, GLIBC mismatch, PROJ mismatch, quota exceeded, SAS token expiry, and more.

9. **[emb-training.md](emb-training.md)** -- GPU training guide for EMB. Mixed precision (bfloat16, no GradScaler), BatchNorm pitfalls, EMA buffer copying, DDP unwrapping, learning rates per architecture, cosine schedule, channel normalization.


## Project Context

The **romcrop** project generates 10m-resolution crop type maps for Romania using:

- **TESSERA** foundation model embeddings (128-band GeoTIFFs, ~1.4-48 GB each)
- **Sentinel-1** (SAR) and **Sentinel-2** (optical) satellite imagery
- **CatBoost** classifiers trained on TESSERA embeddings
- **Planetary Computer** API for satellite data access

The pipeline has two main phases:

1. **Embedding generation** (GPU-intensive) -- Download S1+S2 imagery, stack temporal data, retile into patches, run TESSERA foundation model inference on AMD MI250X GPUs, stitch output, convert to GeoTIFF. Runs on LUMI.

2. **Classification** (CPU-intensive) -- Sample embeddings at labeled locations, train CatBoost with Optuna hyperparameter optimization, run wall-to-wall inference, validate against ground truth. Runs on both LUMI and Saga.

## Code Transfer Workflow

Neither LUMI nor Saga had git credentials configured. All code transfers used rsync from the development EC2 instance:

```bash
# EC2 -> LUMI
rsync -avz /home/ubuntu/romcrop/ lumi2:/scratch/project_465002500/romcrop/ \
    --exclude '.git' --exclude '__pycache__' --exclude 'data/' --exclude 'outputs/'

# EC2 -> LUMI (TESSERA model and preprocessing tools)
rsync -avz /home/ubuntu/tessera/ lumi2:/scratch/project_465002500/romcrop/tessera/

# EC2 -> Saga
rsync -avz /home/ubuntu/romcrop/ saga:/cluster/work/users/digifarm/sure/v0/ \
    --exclude '.git' --exclude '__pycache__' --exclude 'data/' --exclude 'outputs/'
```

Always exclude `data/` and `outputs/` (multi-GB directories) and `.git/` (unnecessary on HPC and can be large). Use `--exclude '__pycache__'` to avoid transferring bytecode.

## SLURM Essentials

Commands used daily across both systems:

```bash
# Submit a job
sbatch script.sbatch

# Monitor running jobs
squeue -u $USER
squeue -u $USER --format="%.10i %.9P %.20j %.8T %.10M %.6D %R"  # detailed format

# Check finished job statistics (memory, runtime)
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,MaxVMSize

# Cancel jobs
scancel JOBID                          # cancel one job
scancel -u $USER                       # cancel all your jobs

# Adjust array job concurrency without resubmitting
scontrol update JobId=XXXXX ArrayTaskThrottle=4

# Follow live output
tail -f logs/jobname_JOBID.out
```

## Two-Environment Architecture

For ML pipelines that need both preprocessing (rasterio, dask, pyproj) and GPU inference (PyTorch with ROCm), use two separate environments:

1. **Virtual environment** (cray-python) for preprocessing and postprocessing
2. **Singularity/Apptainer container** for GPU inference

Pass environment paths to pipeline scripts via environment variables:

```bash
export PYBIN_PREPROC="$VENV_DIR/bin/python3"   # venv Python binary
export PYTORCH_SIF="/path/to/pytorch.sif"       # GPU container path
```

This avoids the impossible task of getting rasterio, GDAL bindings, and PyTorch ROCm all working in the same environment. Each tool uses the environment it needs.

## Memory Estimation Rules of Thumb

```
Rasterio read:      band_count x height x width x dtype_bytes
CatBoost training:  ~2x dataset size in memory
TESSERA GeoTIFF:    128 bands x H x W x 4 bytes (float32) = 1.4-48 GB per chunk
SLURM request:      Always 2x expected peak for safety
```

Never load a full multi-GB raster into memory. Use `rasterio.sample()` for point queries, or read one band at a time with windowed reads.
