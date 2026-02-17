# Fields Pipeline on HPC (Betzy & Saga)

Full-US crop field tile generation pipeline: 29.1M field polygons x 15 years (2008-2024), producing PMTiles for a web map viewer. This documents our real experience running the pipeline on NRIS/Sigma2 clusters, including every error hit and fix applied.

## Pipeline Overview

Per year, the pipeline does:

1. **Download CDL** -- USDA Cropland Data Layer raster (~2.5GB zip per year)
2. **Zonal stats** -- Compute majority crop type per field polygon (single-threaded Python + rasterio, ~2-4h per year, ~700MB parquet output)
3. **Build PMTiles** -- Stream 58M GeoJSON features (polygon + centroid per field) through tippecanoe, convert MBTiles to PMTiles (~1.5h, ~17GB output)
4. **Upload to S3** -- `aws s3 cp` at ~300 MiB/s (~1 min per 17GB file)
5. **Update manifest** -- JSON list of available years, CloudFront invalidation

Total per year: ~4-6 hours. 15 years on 4 nodes: ~8-10 hours.

---

## Cluster Choice: Saga vs Betzy

We initially ran on **Saga**, then switched to **Betzy** after hitting issues.

### Saga (first attempt)

| Item | Value |
|------|-------|
| Partition | `normal` (single-node OK) |
| CPUs | Up to 128 per node |
| RAM | Up to 178GB per node |
| Walltime | 7 days max |
| `--mem` flag | Required |

**Saga advantages:**
- Single-node jobs allowed (simpler scripts)
- Flexible resource requests

**Why we left Saga:**
- PEP 668 errors when building container (externally-managed-environment)
- Node failures mid-job (multiple occurrences)
- Walltime insufficient for sequential processing of 15 years on 1 node
- TMPDIR issues (default /tmp too small)

### Betzy (production run)

| Item | Value |
|------|-------|
| Partition | `normal` (min 4 nodes!) |
| CPUs per node | 128 physical cores, 256 logical (hyperthreading) |
| RAM per node | ~242GB |
| Walltime | 4 days max |
| `--mem` flag | Do NOT use (crashes the job) |

**Betzy advantages:**
- 4 nodes = 4 years in parallel
- More RAM per node (242GB vs 178GB)
- More stable than Saga for our workload

**Betzy gotchas (see detailed sections below):**
- **Minimum 4 nodes** on normal partition -- cannot request fewer
- **Do NOT use `--mem`** -- Betzy allocates full node memory automatically
- 128 physical cores but 256 logical CPUs -- tools see 256 via `os.cpu_count()`
- `/dev/shm` is 126GB (useful!), `/tmp` is only 1.5GB (useless)

---

## Apptainer Container

### Definition File

```
Bootstrap: docker
From: ghcr.io/osgeo/gdal:ubuntu-small-3.10.1

%post
    apt-get update -qq && apt-get install -y -qq \
        python3-pip curl unzip sqlite3 git \
        build-essential libsqlite3-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

    pip3 install --no-cache-dir --break-system-packages \
        geopandas==1.1.2 pandas==2.2.2 numpy==1.26.4 \
        shapely==2.1.2 pyarrow==14.0.2 rasterio==1.3.11 \
        pmtiles==3.5.0 pyproj ujson mapbox_vector_tile

    cd /tmp && git clone --depth 1 https://github.com/felt/tippecanoe.git \
    && cd tippecanoe && make -j"$(nproc)" && make install \
    && cd / && rm -rf /tmp/tippecanoe

    cd /tmp && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip \
    && unzip -q awscliv2.zip && ./aws/install && rm -rf /tmp/aws /tmp/awscliv2.zip
```

### Building on Betzy

```bash
ssh betzy
cd /cluster/work/users/digifarm/fields/saga/containers
apptainer build fields.sif fields.def
```

Build takes ~5 minutes. The `--break-system-packages` flag is required for the GDAL base image (uses PEP 668 externally-managed Python). On Saga, we initially hit PEP 668 errors without this flag.

### Running Commands in Container

```bash
apptainer exec --bind /cluster "$CONTAINER" python3 script.py
apptainer exec --bind /cluster "$CONTAINER" tippecanoe ...
apptainer exec --bind /cluster "$CONTAINER" aws s3 cp ...
```

The `--bind /cluster` is essential -- without it, the container cannot see any project data on the Lustre filesystem.

---

## SLURM Script: Multi-Node Parallel Processing

### The 4-Node Requirement

Betzy normal partition requires `--nodes=4` minimum. Our solution: process 4 years simultaneously, one per node, using `srun --exclusive`:

```bash
#SBATCH --job-name=fields_all
#SBATCH --nodes=4
#SBATCH --partition=normal
#SBATCH --time=4-00:00:00
#SBATCH --account=nn12037k
# NOTE: Do NOT add --mem on Betzy!

YEARS=(2008 2009 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024)

for year in "${YEARS[@]}"; do
  # Wait if all 4 nodes are busy
  while (( $(jobs -r | wc -l) >= 4 )); do
    wait -n 2>/dev/null || true
  done

  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=128 \
    bash "$YEAR_SCRIPT" "$year" ... \
    > "${WORKDIR}/logs/year_${year}.log" 2>&1 &
done

# Wait for all years
wait
```

Key points:
- `srun --exclusive` gives exclusive access to one node per year
- `--cpus-per-task=128` requests all physical cores
- Background `&` lets the loop continue to fill all 4 nodes
- `wait -n` blocks when all nodes are busy, releases when any year finishes
- Each year's stdout/stderr goes to a separate log file

### Validation Functions

Built into the year script to enable skip-if-valid logic:

```bash
is_valid_zonal() {
  # Check parquet footer (last 4 bytes = "PAR1") + row count
  run python3 - "$1" "$EXPECTED_ROWS" <<'PY'
import pyarrow.parquet as pq
pf = pq.ParquetFile(sys.argv[1])
if pf.metadata.num_rows != int(sys.argv[2]):
    raise SystemExit(4)
PY
}

is_valid_pmtiles() {
  # Check PMTiles header is readable
  run python3 - "$1" <<'PY'
from pmtiles.reader import MmapSource, Reader
with open(sys.argv[1], "rb") as f:
    r = Reader(MmapSource(f))
    _ = r.header()
PY
}
```

This lets the pipeline resume after cancellation -- completed years are skipped.

---

## Bug #1: Tippecanoe "shards not a power of 2" (CRITICAL)

### Symptom

```
Internal error: 745 shards not a power of 2
```

Tippecanoe crashes after streaming all 29.1M features (~30 min of work wasted per attempt).

### Root Cause

Tippecanoe computes internal shard count from system parameters:

```
CPUS = sysconf(_SC_NPROCESSORS_ONLN)  # 256 on Betzy (128 cores x 2 HT)
CPUS = round_down_to_power_of_2(CPUS) # 256 (already power of 2)
MAX_FILES = test_open_files() * 3 / 4  # ~1500
TEMP_FILES = (MAX_FILES - 10) / 2      # 745
if TEMP_FILES > CPUS * 4:
    TEMP_FILES = CPUS * 4              # 1024 > 745, so NOT capped!
child_shards = TEMP_FILES / threads     # 745 / 1 = 745 (NOT power of 2!)
```

On Betzy with 256 logical CPUs: `CPUS*4 = 1024 > 745`, so `TEMP_FILES` stays at 745 (not a power of 2). The power-of-2 check on `child_shards` then fails.

### Fix

Set `TIPPECANOE_MAX_THREADS=128` environment variable before running tippecanoe:

```python
env = os.environ.copy()
env["TIPPECANOE_MAX_THREADS"] = "128"
subprocess.Popen(["tippecanoe", ...], env=env)
```

This forces `CPUS = 128`, so `CPUS*4 = 512 < 745`, and `TEMP_FILES = 512` (power of 2).

### Why Other Fixes Don't Work

- **`-j` flag**: In tippecanoe, `-j` is a **feature filter** (GeoJSON expression), NOT thread count. We initially tried `-j256` which was silently interpreted as a filter expression.
- **Patching thread count in Python**: The `threads` variable in tile.cpp is separate from `TEMP_FILES`. Even with a power-of-2 thread count, `TEMP_FILES / threads` can still be non-power-of-2 if `TEMP_FILES` isn't.
- **CHANGELOG says "fixed"**: Tippecanoe's CHANGELOG mentions fixing this bug, but the fix only applies when `CPUS*4` actually caps `TEMP_FILES`. On high-core machines where `CPUS*4` exceeds the open file limit, the bug persists.

### Key Lesson

**Always set `TIPPECANOE_MAX_THREADS=128` on any machine with more than 186 logical CPUs.** The threshold is `(MAX_FILES-10)/2 / 4` -- if CPU count exceeds this, the bug triggers.

---

## Bug #2: Lustre I/O Stalls with Tippecanoe Temp Files (CRITICAL)

### Symptom

Tippecanoe starts, streams all features (~30 min), then sits at 1 thread with 0% CPU for hours. MBTiles stays at 28KB (header only). Process shows `cl_sync_io_wait` in `/proc/PID/wchan`.

### Root Cause

Tippecanoe creates ~500-700 temporary files for sorting and distributing features across shards. With `--temporary-directory=/cluster/work/...` (Lustre), each file operation goes through the parallel filesystem. Lustre has high latency for small random I/O and metadata operations.

With 4 tippecanoe processes across 4 nodes all writing to the same Lustre directory, I/O contention is catastrophic. The process effectively deadlocks on filesystem operations.

### Diagnosis

```bash
# On compute node:
PID=$(pgrep tippecanoe)
cat /proc/$PID/status | grep Threads    # Shows: 1 (should be 128)
cat /proc/$PID/wchan                    # Shows: cl_sync_io_wait (Lustre wait)
ls /proc/$PID/fd/ | wc -l              # Shows: ~719 (hundreds of open files)
ls -la /proc/$PID/fd/ | tail -5        # Shows: all pointing to (deleted) files on Lustre
```

### Fix

Use `/dev/shm` (RAM-backed tmpfs) instead of Lustre for tippecanoe temp files:

```bash
TILE_TMP="/dev/shm/tippecanoe_${YEAR}"
mkdir -p "$TILE_TMP"

python3 build_us_full_pmtiles.py \
  --tmp-dir "$TILE_TMP" ...

rm -rf "$TILE_TMP"
```

**Betzy `/dev/shm`**: 126GB available per node (half of 242GB RAM). More than enough for tippecanoe's ~10-20GB of temp data.

### Performance Impact

| Metric | Lustre (broken) | /dev/shm (fixed) |
|--------|-----------------|-------------------|
| CPU usage | 25% (I/O bound) | 90-143% |
| Threads | 1 (stuck) | 2-128 (active) |
| Tile generation | Never completes | ~45 min |
| MBTiles growth | Stuck at 28KB | 0 → 18GB |

### Key Lesson

**Never use Lustre for applications that create hundreds of small temp files.** This applies to tippecanoe, SQLite (WAL mode), and any tool with heavy random I/O. Use `/dev/shm` (RAM tmpfs) or local NVMe when available.

**Betzy filesystem summary:**
- `/dev/shm` -- 126GB RAM tmpfs, perfect for temp files
- `/tmp` -- 1.5GB tmpfs, too small for most workloads
- `/cluster/work/` -- Lustre, great for large sequential I/O, terrible for small random I/O

---

## Bug #3: Betzy Normal Partition Requires 4 Nodes

### Symptom

```
sbatch: error: --nodes >= 4 required for normal jobs on Betzy
```

### Root Cause

Betzy's normal partition enforces a minimum of 4 full nodes per job. This is a policy choice to prevent underutilization of the large-node cluster.

### Fix

Redesign the script to use all 4 nodes productively. Our solution: process 4 years in parallel using `srun --exclusive --nodes=1` inside a 4-node sbatch allocation. See the SLURM Script section above.

---

## Bug #4: `--mem` Flag Crashes Jobs on Betzy

### Symptom

Job fails immediately at submission or allocation with memory-related errors.

### Root Cause

Betzy's normal partition allocates full nodes. The `--mem` flag is incompatible with full-node allocation -- SLURM cannot reconcile a specific memory request with exclusive node access.

### Fix

Simply omit `--mem` from the sbatch script. Each node provides its full ~242GB automatically.

---

## Bug #5: Saga PEP 668 Container Build Failure

### Symptom

```
error: externally-managed-environment
```

When running `pip install` inside an Apptainer container built from a modern Python base image.

### Fix

Add `--break-system-packages` to pip install:

```
pip3 install --no-cache-dir --break-system-packages package1 package2 ...
```

---

## Bug #6: Saga Node Failures Mid-Job

### Symptom

Random year processes die with `slurmstepd: error: *** STEP CANCELLED ***` messages. No application error -- the compute node itself becomes unavailable.

### Mitigation

- Build validation into the pipeline (skip-if-valid-exists pattern)
- Use `set -uo pipefail` (not `-e`) in the outer script so one failed year doesn't kill all others
- The multi-node Betzy approach is more resilient: if one node fails, only 1 of 4 years is lost

---

## Performance Characteristics

### Zonal Stats (Python + rasterio)

- Single-threaded (rasterio.mask per field polygon)
- ~29.1M fields per year
- ~2-4 hours per year
- ~700MB output parquet
- CPU-bound, uses ~1 core

### GeoJSON Streaming (Python → tippecanoe stdin)

- Reads 3 parquet files (raw geometry, clean area, zonal stats) in parallel batches
- Converts WKB geometry to GeoJSON
- Writes 2 features per field (polygon + centroid) to tippecanoe stdin
- Rate: ~16-22k fields/s
- ~22-30 min per year
- Memory: ~1.5GB Python process

### Tippecanoe Tile Generation

- Reads GeoJSON from stdin, sorts by tile index, generates vector tiles
- Uses internal parallelism (128 threads with TIPPECANOE_MAX_THREADS=128)
- ~45 min per year for 58M features across z4-z14
- Peak memory: ~17-20GB
- Output: ~18GB MBTiles (SQLite)

### MBTiles → PMTiles Conversion

- Python pmtiles library
- ~20 min per year for 18GB MBTiles
- Output: ~17GB PMTiles

### S3 Upload

- `aws s3 cp` from Betzy → S3
- ~300 MiB/s throughput
- ~1 min per 17GB file

### Total Per Year

| Phase | Time | Notes |
|-------|------|-------|
| CDL download | ~5 min | Skip if zonal exists |
| Zonal stats | 2-4h | Skip if valid parquet exists |
| GeoJSON streaming | 22-30 min | |
| Tippecanoe tiling | ~45 min | |
| MBTiles → PMTiles | ~20 min | |
| S3 upload | ~1 min | |
| **Total** | **~4-6h** | |

With 4 nodes processing in parallel: **15 years in ~8-10 hours**.

---

## Data Transfer

### EC2 → Betzy

```bash
# Large parquets (21GB + 19GB)
rsync -avP data/intermediate/fields_clean.parquet \
  betzy:/cluster/work/users/digifarm/fields/data/
rsync -avP data/intermediate/fields_raw.parquet \
  betzy:/cluster/work/users/digifarm/fields/data/

# Scripts and container definition
rsync -avP scripts/ betzy:/cluster/work/users/digifarm/fields/scripts/
rsync -avP saga/ betzy:/cluster/work/users/digifarm/fields/saga/
```

### Betzy → EC2 (post-processing)

```bash
# Download zonals for legend/history rebuild
rsync -avP betzy:/cluster/work/users/digifarm/fields/outputs/zonal/ outputs/zonal/
```

### AWS Credentials on Betzy

AWS credentials propagate into Apptainer containers automatically (they inherit `~/.aws/`). Just ensure the profile exists on the HPC login node:

```bash
ssh betzy
cat >> ~/.aws/credentials <<'EOF'
[kostya]
aws_access_key_id = AKIA...
aws_secret_access_key = ...
EOF
```

---

## Filesystem Layout on Betzy

```
/cluster/work/users/digifarm/fields/
    data/
        fields_clean.parquet        # 21GB, field polygons with area
        fields_raw.parquet          # 19GB, field polygons with WKB geometry
        cdl/                        # Downloaded CDL rasters (cleaned up per year)
    outputs/
        zonal/
            zonal_us_YYYY.parquet   # ~700MB each, majority crop per field
        tiles/
            fields_us_YYYY_full.pmtiles  # ~17GB each
            fields_us_YYYY_full.mbtiles  # intermediate, deleted after conversion
        web/
            us_full_years.json      # manifest of available years
    scripts/                        # Python scripts (rsync'd from EC2)
    saga/
        containers/fields.sif       # ~451MB Apptainer container
        slurm/                      # SLURM scripts
    logs/
        year_YYYY.log               # per-year application logs
        process_all_JOBID.out       # SLURM stdout
        process_all_JOBID.err       # SLURM stderr
    tmp/
        process_year_node.sh        # runtime-generated year script
```

---

## Monitoring

### From EC2 (remote)

```bash
# Job status
ssh betzy "squeue -u digifarm --format='%i %j %T %M %N'"

# Per-year progress
ssh betzy "for y in 2008 2009 2012 2013; do echo -n \"\$y: \"; tail -1 /cluster/work/users/digifarm/fields/logs/year_\${y}.log; done"

# Check completed outputs
ssh betzy "ls -lh /cluster/work/users/digifarm/fields/outputs/tiles/*.pmtiles"

# Check for errors
ssh betzy "grep -rl 'ERROR\|shards not a power' /cluster/work/users/digifarm/fields/logs/year_*.log"

# Check tippecanoe process on compute node
ssh betzy "srun --overlap --jobid=JOBID --nodelist=NODE --ntasks=1 bash -c 'ps aux | grep tippecanoe'"
```

### Debugging Stuck Tippecanoe

```bash
# On compute node via srun --overlap:
PID=$(pgrep tippecanoe)

# Check thread count (should be >1 during tiling)
grep Threads /proc/$PID/status

# Check if I/O bound
cat /proc/$PID/wchan
# "cl_sync_io_wait" = Lustre stall (bad)
# "futex_wait_queue" = normal thread synchronization (good)

# Check open file count
ls /proc/$PID/fd/ | wc -l

# Check /dev/shm usage
df -h /dev/shm
du -sh /dev/shm/tippecanoe_*
```

---

## Saga vs Betzy: Key Differences

| Feature | Saga | Betzy |
|---------|------|-------|
| Min nodes (normal) | 1 | **4** |
| `--mem` flag | Required | **Must NOT use** |
| CPUs per node | Up to 128 | 128 physical / 256 logical |
| RAM per node | Up to 178GB | ~242GB |
| Max walltime | 7 days | 4 days |
| `/dev/shm` | Unknown | 126GB |
| `/tmp` | Unknown | 1.5GB (too small) |
| Node stability | Occasional failures | More stable |
| `os.cpu_count()` | Varies | Returns 256 (not 128!) |

---

## NRIS/Sigma2 Shared Infrastructure

- **Account**: `nn12037k` (shared across Saga and Betzy)
- **Project space**: `/cluster/projects/nn12037k/` (shared between clusters)
- **Work space**: `/cluster/work/users/digifarm/` (per-cluster, NOT shared)
- **Purge policy**: Files unused for ~42-90 days may be deleted from `/cluster/work/`
- **SSH**: OTP-based authentication via authenticator app
- **Container runtime**: `apptainer` (not `singularity`) on both clusters

---

## Summary of Lessons Learned

1. **Lustre is terrible for small random I/O** -- Use `/dev/shm` for temp files from tools like tippecanoe, SQLite, or anything creating hundreds of small files.

2. **`os.cpu_count()` lies on Betzy** -- Returns 256 (hyperthreads) not 128 (physical cores). Cap thread counts at 128 to avoid the tippecanoe shards bug.

3. **`TIPPECANOE_MAX_THREADS=128` is mandatory** on high-core machines -- Without it, tippecanoe crashes with "shards not a power of 2" after wasting 30+ min streaming data.

4. **`-j` in tippecanoe is NOT thread count** -- It's a feature filter (GeoJSON expression). Thread count is controlled by `TIPPECANOE_MAX_THREADS` env var.

5. **Betzy requires 4 nodes minimum** -- Design your job to use all 4 productively (process 4 things in parallel).

6. **Never use `--mem` on Betzy** -- Full-node allocation provides all memory automatically.

7. **Build skip-if-valid-exists logic** into every pipeline step -- Jobs get cancelled, nodes fail, walltime expires. Idempotent steps save hours of recomputation.

8. **Validation before cleanup** -- Always validate output files before deleting inputs (CDL zips, intermediate MBTiles). Our parquet validation checks footer bytes + row count.

9. **Per-year log files are essential** -- With 4 years running in parallel, a single log file is unreadable. Redirect each srun to its own log file.

10. **`srun --overlap` for debugging running jobs** -- Lets you inspect processes on compute nodes without killing the running job. Invaluable for diagnosing stuck tippecanoe.

11. **Container env vars propagate into Apptainer** -- No special flags needed. `export FOO=bar` before `apptainer exec` makes FOO available inside.

12. **AWS credentials just work in Apptainer** -- `~/.aws/` is visible inside the container by default.
