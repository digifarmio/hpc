# Feedback for NRIS / Sigma2 Administrators (Saga & Betzy)

*From: digifarm (account nn12037k)*
*Covering: Geospatial ML pipelines and crop insurance batch processing (2025-2026)*

---

## What Works Well

### Betzy's `preproc` partition

The `preproc` partition is perfectly suited for I/O-bound workloads like satellite COG processing, climate data downloads, and batch API calls. Scheduling is fast, resources are appropriate, and wall-time limits (1 day) are reasonable for iterative development.

### Saga's `devel` QOS for fast iteration

The 2-hour devel QOS on Saga gave us quick scheduling turnaround for testing satellite processing pipelines. Running 100 fields, inspecting results, and resubmitting took minutes rather than hours.

### Shared NRIS authentication and account system

The single `nn12037k` account working across both clusters, with the same OTP-based SSH authentication, made switching between Saga and Betzy seamless.

### Apptainer availability

Having `apptainer` as the container runtime on both clusters was essential. Container-based workflows were the only reliable approach for geospatial Python stacks involving rasterio, GDAL, pyproj, and PyTorch.

---

## Saga Issues

### 1. Broken module-provided PyTorch (HIGH priority)

**Problem:** The `PyTorch/1.12.1-foss-2022a-CUDA-11.7.0` module contains only CUDA stub libraries, not runtime libraries (`libcurand.so.10`, `libcufft.so.10`, etc.). Any code using `torch.cuda` fails or produces wrong results.

**Impact:** We lost a full day debugging this before discovering we had to pip-install PyTorch in a venv instead. Other users likely hit the same issue.

**Suggestion:** Either fix the PyTorch module to include runtime CUDA libraries, remove/deprecate it with a note pointing users to the pip-install approach, or provide a tested Apptainer container with working PyTorch + CUDA.

### 2. GPU node availability (HIGH priority)

**Problem:** GPU nodes on the `accel` partition were frequently in `down` or `drain` state. Jobs sat in the queue for days with no GPU nodes accepting work.

**Impact:** This made Saga unusable for GPU workloads during the affected periods, forcing us to switch to Betzy (with its own queue time issues) or LUMI.

**Suggestion:** Improve monitoring/alerting for GPU node health, or provide a status page showing GPU partition availability. Users shouldn't have to run `sinfo -p accel` repeatedly to discover the partition is entirely down.

### 3. PROJ database version on system nodes

**Problem:** The system-installed proj.db on Saga compute nodes has an old schema version (`DATABASE.LAYOUT.VERSION.MINOR = 2`). Rasterio requires `>= 6`. When rasterio falls back to the system proj.db, coordinate transforms produce **silently wrong results** -- not errors, just incorrect coordinates.

**Impact:** We spent 4 hours debugging why field polygons didn't match Sentinel-2 tiles before discovering this. The silent failure mode is particularly dangerous for geospatial workloads.

**Suggestion:** Update the system-installed PROJ library and proj.db to a current version, or at minimum document this known incompatibility for container-based geospatial workflows. The workaround is `export PROJ_DATA=<container's rasterio proj_data path>`.

### 4. Node failures mid-job

**Problem:** We experienced multiple node failures during long-running jobs on Saga. The SLURM step would be cancelled without application-level errors, losing hours of computation.

**Suggestion:** If node health is degrading, drain nodes proactively rather than letting jobs start on failing hardware.

### 5. Login and Lustre availability issues (MEDIUM priority)

**Problem:** During our usage period (January-February 2026), Saga experienced several login node and Lustre filesystem outages, typically occurring during night hours and weekend days. SSH connections would fail or hang, and running jobs that depended on Lustre would stall or crash.

**Impact:** Weekend and overnight batch jobs were particularly affected -- jobs submitted Friday evening might fail due to a Lustre outage Saturday morning, with no way to detect or recover until Monday. This made unattended long-running jobs unreliable.

**Suggestion:** Improve monitoring and notification for infrastructure outages (email/status page). Consider adding health checks that drain the job queue gracefully before planned maintenance windows.

---

## Betzy Issues

### 1. 4-node minimum on `normal` partition (HIGH priority)

**Problem:** The `normal` partition enforces `--nodes >= 4`. Many legitimate workloads need 1-2 nodes -- data preprocessing, single-node parallel processing, container builds.

**Impact:** We had to redesign our pipeline to process 4 years in parallel just to satisfy the node minimum. While this turned out well, the constraint is arbitrary for many workloads. We sometimes ran 4 nodes with 3 sitting idle.

**Suggestion:** Either (a) reduce the minimum to 1 node, (b) create a `single` partition for 1-node jobs, or (c) document recommended alternatives for users who need fewer nodes (e.g., `preproc` for CPU-only, `accel` for GPU).

### 2. `--mem` flag crashes jobs on `normal` partition (MEDIUM priority)

**Problem:** Using `--mem` (any value) in a `normal` partition job causes immediate failure. Betzy allocates full nodes automatically, and the explicit memory request conflicts with this.

**Impact:** Users migrating scripts from Saga (where `--mem` is required) hit this immediately. The error message doesn't clearly explain the issue.

**Suggestion:** Improve the error message to explicitly state "normal partition uses full-node allocation; remove --mem from your script." Currently the error is cryptic.

### 3. GPU queue times of days-to-weeks (HIGH priority)

**Problem:** The `accel` partition has only 4 nodes (16 A100 GPUs total). Queue times routinely stretched to days. For DDP training across 4 GPUs (one full node), we sometimes waited a week.

**Impact:** Made iterative GPU development impossible on Betzy. We could only submit carefully validated configurations and hope they were correct.

**Suggestion:** Expand the GPU partition, or implement a priority system that gives short debug jobs (1-2 hours) faster scheduling over long training jobs.

### 4. Disk quota on project space is too restrictive for shared accounts (MEDIUM priority)

**Problem:** User `digifarm` has ~1.74 TB quota across ALL `/cluster/projects/` directories. With multiple projects sharing the account, this is quickly exhausted. `rsync` fails silently because it creates temp files that push over quota. Writing logs, checkpoints, or any significant data to project space fails.

**Impact:** We had to set up a complex symlink system from project space to work space, and learned the hard way that `rsync` to project space fails (must use `scp` or `rsync --inplace`).

**Suggestion:** Either increase per-user project quotas, or make the quota error messages more informative (currently `rsync` just fails silently). Also, `rsync --inplace` should be documented as a workaround for tight-quota destinations.

### 5. `/tmp` is only 1.5GB (LOW priority)

**Problem:** `/tmp` on Betzy nodes is only 1.5GB. Many tools default to using `/tmp` for scratch space. Tippecanoe, SQLite, and similar tools that create temporary files will fail or hang.

**Impact:** We had to explicitly redirect all temp operations to `/dev/shm` (126GB, works well) or work space. `/tmp` was essentially useless.

**Suggestion:** Either increase `/tmp` size or clearly document that users should set `TMPDIR=/dev/shm` or similar for applications with significant temp file usage.

### 6. `os.cpu_count()` returns 256 (hyperthreads), causing tool bugs

**Problem:** Betzy nodes have 128 physical cores but 256 logical CPUs (hyperthreading). Python's `os.cpu_count()` and C's `sysconf(_SC_NPROCESSORS_ONLN)` both return 256. This breaks tools like tippecanoe that use CPU count for internal sizing.

**Impact:** Tippecanoe crashed with "shards not a power of 2" after processing millions of features (~30 minutes of wasted work per attempt). The fix required reading tippecanoe source code to understand the interaction.

**Suggestion:** Document this in the Betzy user guide, recommending that users cap thread counts at 128 for CPU-intensive tools. Consider adding a note about `TIPPECANOE_MAX_THREADS` and similar environment variables for known affected tools.

---

## Cross-Cluster Issues

### 1. Lustre performance for small random I/O

**Problem:** Lustre (`/cluster/work/`) has excellent throughput for large sequential reads/writes but catastrophic latency for small random I/O. Tools like tippecanoe (hundreds of temp files), SQLite (WAL mode), and any application creating many small files will stall or deadlock on Lustre.

**Impact:** Our tippecanoe runs went from "never completes" on Lustre to 45 minutes on `/dev/shm`. A 100x+ difference.

**Suggestion:** Add a prominent warning in the filesystem documentation: "Lustre is not suitable for applications that create many small temporary files. Use /dev/shm for scratch space in such workloads."

### 2. Stale file handle errors on Lustre

**Problem:** Modifying a directory (e.g., running `pip install --target pylibs/`) while running jobs reference files in that directory causes `OSError: Stale file handle`. Lustre's metadata caching causes references to become invalid.

**Impact:** We had to create a new directory (`pylibs3`) instead of updating the existing one, and could not figure out why imports were failing until we understood Lustre's caching behavior.

**Suggestion:** Document this in the user guide: "On Lustre, treat pip install targets and shared library directories as immutable while jobs reference them."

### 3. File purge policy lacks advance warning

**Problem:** Both Saga and Betzy purge inactive files on `/cluster/work/` after ~42-90 days. There is no advance notification.

**Suggestion:** Send email warnings 7 days before purging, similar to how some other HPC sites handle this.

---

## Feature Requests

1. **GPU partition status dashboard.** A web page showing current GPU node availability, queue depth, and estimated wait times for Saga `accel` and Betzy `accel`. This would save users from submitting jobs to down partitions.

2. **Pre-built geospatial container images.** NRIS-maintained Apptainer images with common geospatial stacks (GDAL + rasterio + PyTorch + CUDA) would save every geospatial user from building their own containers. CSC does this well for LUMI.

3. **Documentation for container-based workflows.** The current docs focus on the module system. A dedicated page on "Running containerized workloads on NRIS clusters" covering Apptainer best practices, PROJ_DATA, GPU passthrough (`--nv`), and filesystem binding would help the growing number of container users.

4. **Single-node partition on Betzy.** Even a small one (4-8 nodes) would serve users who need Betzy's hardware (242GB RAM, fast interconnect) but don't need 4+ nodes.

---

## Summary

Saga and Betzy are capable systems with distinct strengths. Saga excels at fast-iteration development; Betzy excels at large-scale batch processing. The main pain points are: broken module PyTorch, GPU scarcity, Lustre's poor small-I/O performance (which needs documentation), Betzy's restrictive partition policies, and the silent PROJ_DATA mismatch that affects all geospatial workloads.

The most impactful improvements would be: fixing or deprecating the broken PyTorch module (#1 Saga), documenting Lustre's small-I/O limitations, and providing pre-built geospatial containers. These three changes alone would save new users days of debugging.

Thank you for operating these systems and for the responsive support when we've needed it.
