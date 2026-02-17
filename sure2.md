# Sure2 HPC Guide — Crop Emergence Insurance Engine

Detailed instructions for running the Sure2 satellite-based crop emergence insurance pipeline on Saga and Betzy, including every bug encountered, every fix applied, and every lesson learned.

**Project dates:** Feb 2026
**Scale:** 13,510 corn fields x 5 years (2021-2025), 32 Romanian counties
**Total HPC time:** ~84 hours across Saga + Betzy

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Saga Setup (V1 Pipelines)](#saga-setup-v1-pipelines)
- [Betzy Setup (V2 Docker CLI Batch)](#betzy-setup-v2-docker-cli-batch)
- [Pipeline Execution Order](#pipeline-execution-order)
- [Environment Variables — The Critical Ones](#environment-variables--the-critical-ones)
- [Container Patterns](#container-patterns)
- [Drought V2 — Pre-Loading Architecture](#drought-v2--pre-loading-architecture)
- [Batch CLI — Resumable Parallel Processing](#batch-cli--resumable-parallel-processing)
- [Bugs and Fixes (All 10)](#bugs-and-fixes-all-10)
- [Performance Numbers](#performance-numbers)
- [Results Summary](#results-summary)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Retrieving Results](#retrieving-results)
- [What We'd Do Differently](#what-wed-do-differently)

---

## Architecture Overview

Sure2 has 3 independent pipelines, plus a combined "full assessment" Docker CLI:

```
[Individual Pipelines — Saga V1]

1. Planting Estimation  →  S2 EVI timeseries → planting date per field
2. Emergence Detection  →  S2 spatial EVI at DAP30 → % not emerged
3. Drought V2 Compute   →  ERA5-Land + CHIRPS → drought gate (D > 0.5)

[Combined Pipeline — Betzy V2]

Docker CLI batch  →  runs all 12 steps per field inside Apptainer container
```

The individual pipelines were developed first on Saga (faster iteration), then consolidated into the Docker CLI batch for production runs on Betzy.

---

## Saga Setup (V1 Pipelines)

### Cluster Details

| Item | Value |
|------|-------|
| Hostname | saga.sigma2.no |
| Username | digifarm |
| SLURM Account | nn12037k |
| Container | `/cluster/work/users/digifarm/sure/v0/containers/romcrop.sif` |
| Project Dir | `/cluster/work/users/digifarm/sure/v1/` |

### Directory Layout

```
/cluster/work/users/digifarm/sure/v1/
├── scripts/           # Pipeline scripts (29_, 30_, 31_, 32_)
├── demo/engine/       # Engine modules (preserves "from demo.engine..." imports)
├── inputs/            # corn_fields_by_county.geojson
├── outputs/
│   ├── planting_all/  # planting_dates_*.csv (13,510 fields)
│   ├── emergence_*/   # Per-year emergence CSVs
│   └── drought_v2_*/  # Per-year drought CSVs
├── cache/
│   ├── drought/       # ERA5-Land monthly .nc, CHIRPS daily .tif
│   └── climatology/   # Precomputed baselines
├── logs/              # SLURM stdout/stderr
├── slurm/             # .sbatch files
├── pylibs/            # Extra pip packages (V1, abandoned)
└── pylibs3/           # Extra pip packages (V2, working)
```

### Initial Deploy

```bash
# From local machine
rsync -avz demo/engine/ saga:/cluster/work/users/digifarm/sure/v1/demo/engine/
rsync -avz demo/pipelines/ saga:/cluster/work/users/digifarm/sure/v1/scripts/
rsync -avz demo/slurm/ saga:/cluster/work/users/digifarm/sure/v1/slurm/
rsync -avz demo/inputs/corn_fields_by_county.geojson saga:/cluster/work/users/digifarm/sure/v1/inputs/

# Install missing Python packages (inside container)
ssh saga
cd /cluster/work/users/digifarm/sure/v1
apptainer exec ../v0/containers/romcrop.sif pip install --target pylibs3 boto3 cdsapi xarray netCDF4
```

**IMPORTANT:** The engine lives at `v1/demo/engine/` (not `v1/engine/`) because the code uses `from demo.engine.emergence.s2_spatial import ...`. The directory structure must match the local repo layout.

**IMPORTANT:** Scripts go to `v1/scripts/` (not `v1/demo/pipelines/`). The script filenames stay the same (29_planting, 30_batch_drought, etc.) but the path prefix changes.

### Container Packages

**In romcrop.sif:** rasterio, scipy, numpy, pandas, shapely, pyproj, pystac_client, matplotlib

**Missing (installed to pylibs3):** boto3, cdsapi, xarray, netCDF4

---

## Betzy Setup (V2 Docker CLI Batch)

### Cluster Details

| Item | Value |
|------|-------|
| Hostname | betzy.sigma2.no |
| Username | digifarm |
| SLURM Account | nn12037k (shared with Saga) |
| Partition | preproc (I/O-bound, no GPU) |
| Container | `/cluster/work/users/digifarm/sure2/containers/sure2-assess-v2.sif` |
| Work Dir | `/cluster/work/users/digifarm/sure2/` |

### Why Betzy for the Docker CLI Batch?

The Docker CLI is a self-contained assessment tool (Cython-compiled, embedded data files). It doesn't need pylibs, external engine modules, or manual PYTHONPATH setup. We just need to:
1. Convert the Docker image to an Apptainer .sif
2. Bind an output directory
3. Pass field ID + planting date as arguments

Betzy's `preproc` partition is ideal because our workload is I/O-bound (S2 COG reads over HTTPS), not CPU- or GPU-bound.

### Building the Container

```bash
# On local machine: build Docker image
docker build -f Dockerfile.cli -t sure2-assess-v2 .

# Export as tarball
docker save sure2-assess-v2 | gzip > sure2-assess-v2.tar.gz

# Transfer to Betzy
rsync -avP sure2-assess-v2.tar.gz betzy:/cluster/work/users/digifarm/sure2/containers/

# On Betzy: convert to .sif (or let the sbatch script do it automatically)
apptainer build sure2-assess-v2.sif docker-archive://sure2-assess-v2.tar.gz
```

The Docker image uses a two-stage build:
1. **Builder stage:** Cython compiles all `demo/**/*.py` to `.so` binaries
2. **Runtime stage:** Slim image with only `.so` files, embedded data, no source code

### Directory Layout

```
/cluster/work/users/digifarm/sure2/
├── containers/           # sure2-assess-v2.sif (or .tar.gz)
├── data/                 # planting_dates_all_combined.csv
├── outputs/
│   └── cli_batch_v2/     # Per-field dirs: F2022_*/assessment_*.json
├── logs/                 # SLURM logs
├── tmp/                  # xargs temp files
└── patch_drought.py      # One-off fix script
```

---

## Pipeline Execution Order

### Saga V1 (individual pipelines, ~48h total)

```
Step 1: Planting (2h devel QOS)
    sbatch slurm/planting.sbatch
    Output: outputs/planting_all/planting_dates_all.csv
    ↓
Step 2: Emergence (per year, 24h each, can run in parallel)
    sbatch slurm/emergence_2021_remaining.sbatch
    sbatch slurm/emergence_2022_remaining.sbatch
    sbatch slurm/emergence_2023.sbatch
    sbatch slurm/emergence_2024.sbatch
    sbatch slurm/emergence_2025.sbatch
    ↓
Step 3: Drought Climatology (one-time, 48h)
    sbatch slurm/drought_climatology.sbatch
    Output: cache/climatology/era5land_swvl1_climatology.nc
            cache/climatology/chirps_precip_climatology.nc
    ↓
Step 4: Drought V2 Compute (per year, 12h, can run in parallel)
    sbatch slurm/drought_v2.sbatch 2021
    sbatch slurm/drought_v2.sbatch 2022
    ...
```

### Betzy V2 (combined Docker CLI, ~36h)

```
Option A: Single node, 20 workers
    sbatch slurm/batch_cli_v2.sbatch

Option B: 3-node array, 15 workers each = 45 concurrent
    sbatch slurm/betzy_batch_v2.sbatch

Post-run: Collect statistics
    bash slurm/betzy_collect_stats.sh

Fix-up: Patch failed drought assessments
    sbatch slurm/betzy_patch_drought.sbatch
```

---

## Environment Variables — The Critical Ones

Every single one of these matters. Miss one and jobs fail silently or produce wrong results.

### PROJ_DATA (SEVERITY: CRITICAL)

```bash
export PROJ_DATA="/usr/local/lib/python3.12/dist-packages/rasterio/proj_data"
```

**The story:** The system proj.db on Saga/Betzy nodes has `DATABASE.LAYOUT.VERSION.MINOR = 2`, but rasterio needs `>= 6`. Without this fix, `rasterio.warp.transform_geom()` produces SILENTLY WRONG coordinates. We spent hours debugging why field polygons didn't match S2 tiles before discovering this.

On the local dev machine, we worked around this differently — using `pyproj.Transformer` + `shapely.ops.transform` instead of rasterio's built-in transform. On HPC, the container's rasterio ships its own correct proj_data, so we just point PROJ_DATA to that.

### AWS_NO_SIGN_REQUEST

```bash
export AWS_NO_SIGN_REQUEST=YES
```

Sentinel-2 COGs on AWS Earth Search are public. Without this, rasterio tries to find AWS credentials (which don't exist on HPC) and fails with an opaque error.

### GDAL_DISABLE_READDIR_ON_OPEN

```bash
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
```

Without this, GDAL lists the parent S3 directory before opening each COG. Earth Search buckets have thousands of files per directory. This adds 5-10 seconds per read. With the fix: COG reads drop from ~12s to ~3s. **This is a 3-5x speedup for free.**

### PYTHONPATH (Saga only)

```bash
# V1 pylibs (abandoned — Lustre stale handle errors)
export PYTHONPATH="$PROJECT_DIR/pylibs"

# V2 pylibs (working)
export PYTHONPATH="$PROJECT_DIR/pylibs3"
```

### SURE2_ROOT (Saga only)

```bash
export SURE2_ROOT="$PROJECT_DIR"
```

Engine modules use `from demo.engine.emergence.s2_spatial import ...`. The dir structure must mirror the local repo.

### Full Apptainer invocation (Saga)

```bash
apptainer exec \
  --bind "$PROJECT_DIR" \
  --env PYTHONPATH="$PROJECT_DIR/pylibs3" \
  --env SURE2_ROOT="$PROJECT_DIR" \
  --env PROJ_DATA="/usr/local/lib/python3.12/dist-packages/rasterio/proj_data" \
  --env AWS_NO_SIGN_REQUEST=YES \
  --env GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR \
  "$CONTAINER" \
  python3 "$PROJECT_DIR/scripts/30_batch_drought.py" ...
```

### Full Apptainer invocation (Betzy — isolated)

```bash
apptainer run --containall --cleanenv \
  --bind "$field_dir:/data/results" \
  --env AWS_NO_SIGN_REQUEST=YES \
  --env GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR \
  --env TMPDIR=/tmp \
  "$CONTAINER" \
  --demo-field "$fid" --planting-date "$planting" --era5-api-key "$ERA5_KEY" --no-sr --no-stats
```

Key difference: `--containall --cleanenv` strips ALL host environment and filesystem. Only explicitly bound paths and env vars are visible inside. More reproducible but requires careful setup.

---

## Container Patterns

### Saga: Open bind (shared filesystem)

```bash
apptainer exec --bind "$PROJECT_DIR" "$CONTAINER" python3 script.py
```

The whole project directory is visible inside the container. Scripts can read inputs and write outputs directly. Simple but less isolated.

### Betzy: Contained isolation

```bash
apptainer run --containall --cleanenv \
  --bind "$output_dir:/data/results" \
  "$CONTAINER" \
  --demo-field F2022_420140 --planting-date 2022-05-13
```

Only the output directory is bound. The container has embedded input data (geojson, planting CSV, climatology files). This is how we ship the Docker CLI to partners — it's self-contained.

### Converting Docker to Apptainer

```bash
# Build Docker image locally
docker build -f Dockerfile.cli -t sure2-assess-v2 .

# Export tarball
docker save sure2-assess-v2 | gzip > sure2-assess-v2.tar.gz

# Transfer to HPC
rsync -avP sure2-assess-v2.tar.gz betzy:/cluster/work/.../containers/

# Convert (auto-detected in sbatch script)
apptainer build sure2-assess-v2.sif docker-archive://sure2-assess-v2.tar.gz
```

**Note:** The .sif file is read-only. ALL output must go through `--bind` mounts. We bind per-field output directories to `/data/results` inside the container.

---

## Drought V2 — Pre-Loading Architecture

The drought batch pipeline (`30_batch_drought.py`) uses a critical performance pattern: pre-load ALL climate data into RAM before the per-field loop.

### Why

The naive approach (open file, extract point, close file, repeat) hits Lustre's random I/O latency (~100ms per small read). With 13,510 fields reading the same monthly files, that's 1.35M file operations.

The pre-loading approach loads everything once (~700MB RAM), then does pure numpy extraction per field (~0.1ms each).

### Implementation

```python
# Phase 1: Bulk regional downloads (1 CDS call per month, CHIRPS via HTTP)
_download_era5land(cache_dir, year, month, region)
_download_chirps(cache_dir, year, month, day)

# Phase 1.5: Pre-load ALL data into RAM
era5_data = {}
for f in era5_files:
    ds = xr.open_dataset(f).load()  # .load() forces into numpy arrays
    era5_data[month] = ds

chirps_data = {}
for f in chirps_files:
    with rasterio.open(f) as src:
        chirps_data[date] = src.read(1)  # Full raster in memory

# Phase 2: Per-field extraction (pure numpy, no I/O)
for field in fields:
    smp = _extract_era5_from_preloaded(era5_data, field.centroid)  # ~0.1ms
    pdef = _extract_chirps_from_preloaded(chirps_data, field.centroid)  # ~0.1ms
    D = compute_drought_score(smp, pdef, ...)
```

### Results

- 6-7 fields/second throughput
- ~8 minutes per year (2,700 fields)
- ~40 minutes total for all 5 years
- RAM usage: ~700MB (ERA5-Land 47MB + CHIRPS 650MB)

---

## Batch CLI — Resumable Parallel Processing

### Architecture

```
batch_cli_v2.sbatch
  ├── Python: build field list (skip completed)
  ├── xargs -P $WORKERS: parallel processing
  │   ├── process_field() → apptainer run per field
  │   ├── process_field() → apptainer run per field
  │   └── ...
  └── Python: collect statistics
```

### Resumability

```python
# Build field list, skipping completed
for fid, date in all_fields:
    existing = glob.glob(os.path.join(out_dir, fid, "assessment_*.json"))
    if existing:
        skip += 1
        continue
    print(f"{fid},{date}")
```

And inside `process_field()`:
```bash
# Double-check resume (race condition guard between parallel workers)
if ls "$field_dir"/assessment_*.json &>/dev/null 2>&1; then
    return 0
fi
```

This means:
- Jobs can be interrupted (`scancel`) and resubmitted safely
- Multiple array tasks can process the same field list without conflicts
- Post-crash restarts skip all completed work

### Array Job Distribution (Betzy)

```bash
#SBATCH --array=0-2  # 3 tasks

# Each task gets every 3rd field:
# Task 0: fields[0, 3, 6, 9, ...]
# Task 1: fields[1, 4, 7, 10, ...]
# Task 2: fields[2, 5, 8, 11, ...]
my_fields = fields[task_id::num_tasks]
```

3 nodes x 15 workers each = 45 concurrent field assessments. Each field takes 2-5 minutes.

---

## Bugs and Fixes (All 10)

### Bug 1: PROJ_DATA Incompatibility (CRITICAL)

**Symptom:** Coordinates silently wrong. Fields don't overlap with S2 tiles. EVI reads return all zeros.

**Root cause:** System proj.db version mismatch.

**Time to diagnose:** ~4 hours

**Fix:** `export PROJ_DATA="/usr/local/lib/python3.12/dist-packages/rasterio/proj_data"`

**Lesson:** Always verify CRS transforms produce reasonable coordinates. Print centroids and compare with expected lat/lon.

### Bug 2: xr.concat HDF Error (HIGH)

**Symptom:** `NetCDF: HDF error` when loading ERA5-Land monthly files.

**Root cause:** pylibs xarray version incompatible with container's system netCDF4. The C-level HDF5 library crashes when xarray lazily concatenates datasets.

**Fix (extract.py):** Process monthly files separately, concatenate DataFrames:
```python
dfs = []
for f in files:
    ds = xr.open_dataset(f)
    df = extract_to_dataframe(ds)
    dfs.append(df)
result = pd.concat(dfs)
```

**Fix (climatology.py):** Force eager loading before concat:
```python
datasets = [xr.open_dataset(f).load() for f in files]
combined = xr.concat(datasets, dim='time')  # Safe: data already in RAM
```

**Lesson:** When mixing pylibs packages with container system packages, lazy evaluation (dask/xarray) is fragile. Force `.load()` to materialize data eagerly.

### Bug 3: Threading Segfault (HIGH)

**Symptom:** `Segmentation fault` or `Bus error` with ThreadPoolExecutor + rasterio.

**Root cause:** GDAL and HDF5 C libraries are not thread-safe in this container build.

**Fix:** Sequential processing. No ThreadPoolExecutor.

```python
# BAD:
with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(read_cog, items))

# GOOD:
results = [read_cog(item) for item in items]
```

**Lesson:** ProcessPoolExecutor (separate processes) is fine. The issue is threads sharing GDAL's internal state.

### Bug 4: rasterio.mask Segfault with CHIRPS (HIGH)

**Symptom:** Segfault when using `rasterio.mask.mask()` on CHIRPS GeoTIFFs.

**Root cause:** Unknown — possibly CHIRPS file structure + container's rasterio build.

**Fix:** Centroid point reads instead of polygon masking. CHIRPS pixels are 5.5km — larger than typical 4ha corn fields.

```python
centroid = shape(geom).centroid
row, col = src.index(centroid.x, centroid.y)
value = src.read(1)[row, col]
```

### Bug 5: Lustre Stale File Handle (MEDIUM)

**Symptom:** `OSError: [Errno 116] Stale file handle` when importing from pylibs.

**Root cause:** Lustre caches file metadata aggressively. Replacing pylibs while jobs reference them causes stale handles.

**Fix:** Create fresh directory (`pylibs3`) instead of updating in-place.

**Lesson:** On Lustre, treat pip install targets as immutable. Never modify a pylibs directory while jobs might reference it.

### Bug 6: SMAP Earthaccess IndexError (LOW)

**Symptom:** `IndexError: list index out of range` in earthaccess.

**Root cause:** SMAP data availability is intermittent. Empty search results.

**Fix:** Graceful skip. SMAP is 25% of the drought score weight — redistribute to remaining 3 sources.

### Bug 7: Corrupt ERA5-Land Downloads (MEDIUM)

**Symptom:** xarray can't open ERA5-Land files. Some are 0 bytes.

**Root cause:** CDS API download interrupted by previous job crash, leaving `*_download` temp files.

**Fix:**
```bash
find cache/drought/ -name '*_download' -delete
find cache/drought/ -size 0 -delete
```

### Bug 8: compute.py Climate Variable Bug (HIGH)

**Symptom:** All fields get identical PDEF scores.

**Root cause:** Line 230 used `era5_clim` instead of `chirps_clim` for precipitation deficit. Copy-paste error.

**Fix:** Change variable name. One character difference, massive impact.

**Lesson:** When all fields produce identical values for a component, suspect a shared-data bug.

### Bug 9: --containall Hiding Everything (MEDIUM)

**Symptom:** Container can't find input files, env vars not set.

**Root cause:** `--containall --cleanenv` strips ALL host environment and filesystem.

**Fix:** Explicitly `--bind` every needed path and `--env` every needed variable.

**Lesson:** Test with a single field first: `apptainer run --containall --cleanenv --bind ... $CONTAINER --demo-field F2022_420140 --planting-date 2022-05-13`

### Bug 10: Python Logging Buffering (LOW)

**Symptom:** SLURM log files empty for long periods, then dump everything at once.

**Root cause:** Python stdout is line-buffered when connected to a TTY, but fully buffered when piped to a file.

**Fix:** `python3 -u script.py` or `export PYTHONUNBUFFERED=1`

---

## Performance Numbers

### Individual Pipelines (Saga V1)

| Pipeline | Fields | Workers | Wall Time | Per-Field | Notes |
|----------|--------|---------|-----------|-----------|-------|
| Planting | 13,510 | 16 | ~70 min | ~0.5s | S2 timeseries, I/O bound |
| Emergence | ~2,700/yr | 6 | ~45 min/yr | ~1s | S2 spatial COGs |
| Drought Clim. | 1 | 1 | ~4h | N/A | CDS API rate-limited |
| Drought V2 | ~2,700/yr | 6 | ~8 min/yr | ~0.15s | Pre-loaded, CPU-bound |

### Docker CLI Batch (Betzy V2)

| Config | Fields | Workers | Wall Time | Per-Field | Notes |
|--------|--------|---------|-----------|-----------|-------|
| 1 node, 20 workers | 13,510 | 20 | ~48h | ~4.5 min | Includes ERA5 download |
| 3 nodes, 15 workers each | 13,510 | 45 | ~24h | ~4.5 min | Array job |

### Resource Recommendations

| Pipeline | CPUs | RAM | Walltime | Partition |
|----------|------|-----|----------|-----------|
| Planting | 16 | 64G | 2h | devel |
| Emergence | 8 | 32G | 24h | normal |
| Drought V2 | 8 | 32G | 12h | normal |
| Climatology | 8 | 64G | 48h | normal |
| CLI Batch (Saga) | 20 | 64G | 48h | normal |
| CLI Batch (Betzy) | 16 | 32G | 24h | preproc |
| Drought Patch | 10 | 16G | 24h | preproc |

---

## Results Summary

### Full 5-Year Run (13,510 field-years)

| Metric | Value |
|--------|-------|
| Total field-years | 13,510 |
| S2 planting PASS | 68.2% (9,209) |
| Claim eligible (>=30% not emerged) | 13.9% (1,277 of PASS) |
| Drought gate TRUE | 12.7% |
| Mean drought score | 0.344 |

### By Year

| Year | S2 PASS | Claim % | Drought Gate % | D Score | Interpretation |
|------|---------|---------|----------------|---------|----------------|
| 2021 | 71.6% | 12.2% | 1.4% | 0.235 | Wet year |
| 2022 | 78.6% | 14.3% | 20.7% | 0.442 | Drought year |
| 2023 | 58.5% | 15.4% | 11.8% | 0.332 | Average |
| 2024 | 63.0% | 16.9% | 23.3% | 0.413 | Drought year |
| 2025 | 71.8% | 10.1% | 1.7% | 0.260 | Wet year |

### Hotspot Counties (highest claim rates)

| County | Claims >=30% | Drought Gate |
|--------|-------------|--------------|
| Buzau | 33.7% | 19.2% |
| Vaslui | 30.8% | 55.0% |
| Vrancea | 29.9% | 22.1% |
| Galati | 29.2% | 28.3% |
| Braila | 27.5% | 44.0% |

---

## Monitoring and Debugging

### Check Job Queue

```bash
ssh saga 'squeue -u digifarm'
ssh betzy 'squeue -u digifarm'
```

### View Logs

```bash
# Saga: stdout/stderr split
ssh saga 'tail -50 /cluster/work/users/digifarm/sure/v1/logs/drought_v2_*.out'

# Betzy: combined log
ssh betzy 'tail -100 /cluster/work/users/digifarm/sure2/logs/sure2_v2_*.log'

# Per-field log (Betzy batch)
ssh betzy 'cat /cluster/work/users/digifarm/sure2/outputs/cli_batch_v2/F2022_420140/stdout.log'
```

### Check Progress (Betzy batch)

```bash
# Completed assessments
ssh betzy 'ls /cluster/work/users/digifarm/sure2/outputs/cli_batch_v2/F*/assessment_*.json 2>/dev/null | wc -l'

# Failed fields (dir exists but no assessment JSON)
ssh betzy 'for d in /cluster/work/users/digifarm/sure2/outputs/cli_batch_v2/F*/; do
  if ! ls "$d"assessment_*.json &>/dev/null; then echo "FAIL: $(basename $d)"; fi
done | wc -l'

# Detailed stats (post-run)
ssh betzy 'bash /cluster/work/users/digifarm/sure2/betzy_collect_stats.sh'
```

### Cancel Jobs

```bash
scancel JOB_ID          # Cancel specific job
scancel -u digifarm     # Cancel all your jobs
```

---

## Retrieving Results

```bash
# Saga individual pipeline results
rsync -avz saga:/cluster/work/users/digifarm/sure/v1/outputs/ demo/outputs/hpc/

# Betzy batch results (large! ~4GB for 13K fields)
rsync -avz betzy:/cluster/work/users/digifarm/sure2/outputs/cli_batch_v2/ demo/outputs/hpc/cli_batch_v2/

# Just the assessment JSONs (~200MB)
rsync -avz --include='*/' --include='assessment_*.json' --exclude='*' \
  betzy:/cluster/work/users/digifarm/sure2/outputs/cli_batch_v2/ demo/outputs/hpc/cli_batch_v2/
```

---

## What We'd Do Differently

### 1. Start with Docker CLI on Betzy from day one

The individual Saga V1 pipelines were useful for development and debugging, but the Docker CLI batch on Betzy was simpler operationally. One container, one sbatch script, fully self-contained. We'd skip the Saga V1 pipelines for production runs.

### 2. Pre-load ALL data for ALL pipelines

The drought V2 pre-loading pattern (bulk download, load to RAM, extract per field) was 10-50x faster than the per-field I/O approach. We'd apply this to the emergence pipeline too — download S2 COGs for the whole region, then extract per field from RAM.

### 3. Use --containall from the start

The Saga open-bind pattern (`--bind $PROJECT_DIR`) is convenient but fragile. Host environment leaks (PROJ_DATA, PYTHONPATH conflicts) caused multiple hard-to-debug issues. The Betzy `--containall --cleanenv` pattern is more work to set up but eliminates an entire class of bugs.

### 4. Test with 10 fields before submitting 13,510

Every batch run had at least one bug that required re-running. A 10-field test run takes ~45 minutes and catches most issues. We did this eventually, but not consistently enough.

### 5. Log rotation

SLURM log files for 13K-field runs are enormous. Would add per-field logging to separate files from the start, with a summary log for the batch.

---

## File Reference

| Local Path | HPC Path (Saga) | Purpose |
|------------|-----------------|---------|
| `demo/slurm/planting.sbatch` | `v1/slurm/planting.sbatch` | Planting estimation |
| `demo/slurm/emergence_YEAR.sbatch` | `v1/slurm/emergence_YEAR.sbatch` | Year-specific emergence |
| `demo/slurm/drought_v2.sbatch` | `v1/slurm/drought_v2.sbatch` | Drought V2 |
| `demo/slurm/drought_climatology.sbatch` | `v1/slurm/drought_climatology.sbatch` | One-time climatology |
| `demo/slurm/batch_cli_v2.sbatch` | Betzy only | Full CLI batch (1 node) |
| `demo/slurm/betzy_batch_v2.sbatch` | Betzy only | Full CLI batch (array) |
| `demo/slurm/betzy_patch_drought.sbatch` | Betzy only | Fix failed drought |
| `demo/slurm/betzy_collect_stats.sh` | Betzy only | Post-run stats |
| `demo/pipelines/30_batch_drought.py` | `v1/scripts/30_batch_drought.py` | Drought V2 pipeline |
| `demo/pipelines/32_build_climatology.py` | `v1/scripts/32_build_climatology.py` | Climatology builder |
| `Dockerfile.cli` | N/A (built locally) | Docker CLI image |
| `build_cython.py` | N/A (built locally) | Cython compilation |
