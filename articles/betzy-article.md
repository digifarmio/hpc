# Running Geospatial ML on Betzy: GPU Training, Batch Processing, and Lustre's Limits

*How we used Norway's Betzy cluster for A100 GPU training and crop insurance batch processing -- and what Betzy's unique constraints taught us about HPC job design.*

---

Betzy, NRIS/Sigma2's compute-focused cluster in Norway, played two main roles in our geospatial ML work: DDP training on A100 GPUs for crop segmentation and batch-running 13,510 crop insurance assessments. Each workload exposed different aspects of Betzy's personality.

## The 4-Node Surprise

The first thing you learn about Betzy is that its `normal` partition requires a minimum of 4 nodes per job. You cannot request fewer. This is a policy choice to prevent underutilization of the large-node cluster, but it forces you to redesign single-node workflows.

Our solution: process multiple independent work items simultaneously, one per node, using `srun --exclusive --nodes=1` inside a 4-node allocation. A simple shell loop fills all 4 nodes with `srun &` background processes and uses `wait -n` to release a node when any item finishes:

```bash
for item in "${ITEMS[@]}"; do
  while (( $(jobs -r | wc -l) >= 4 )); do wait -n; done
  srun --exclusive --nodes=1 bash process_item.sh "$item" &
done
wait
```

Turning a constraint into parallelism.

## Lustre vs. RAM: A 100x Performance Difference

Any tool that creates hundreds of small temporary files will deadlock on Lustre. We discovered this when a data processing tool started, ran for 30 minutes streaming input, then sat at 1 thread with 0% CPU for hours. The output file stayed at 28KB (header only). `/proc/PID/wchan` showed `cl_sync_io_wait` -- the process was blocked on Lustre I/O.

The diagnosis: the tool creates 500-700 temporary files for internal sorting. With the temp directory on Lustre (`/cluster/work/`), each file operation goes through the parallel filesystem. Lustre excels at large sequential I/O but has terrible latency for small random operations. With multiple processes across multiple nodes all writing to the same Lustre directory, I/O contention was catastrophic.

The fix: use `/dev/shm` (RAM-backed tmpfs) for temp files. Betzy provides a generous 126GB of `/dev/shm` per node. Performance went from "never completes" to 45 minutes.

**This lesson generalizes: never use Lustre for applications that create hundreds of small temp files.** SQLite in WAL mode, tippecanoe, and similar tools should always use `/dev/shm` or local NVMe on HPC.

**Betzy filesystem summary:**
- `/dev/shm` -- 126GB RAM tmpfs, perfect for temp files
- `/tmp` -- 1.5GB, useless for most workloads
- `/cluster/work/` -- Lustre, great for large sequential I/O, terrible for small random I/O

## Hyperthreading Breaks Tools That Size on CPU Count

Betzy nodes have 128 physical cores but expose 256 logical CPUs (hyperthreading). Python's `os.cpu_count()` and C's `sysconf(_SC_NPROCESSORS_ONLN)` both return 256. Tools that use CPU count for internal sizing -- thread pools, shard counts, temp file allocation -- may break in subtle ways.

We hit this with tippecanoe (the vector tile generator), which computes internal shard counts from the CPU count. On Betzy, the calculation yielded a non-power-of-2 shard count, causing a crash after 30 minutes of processing. The fix: `export TIPPECANOE_MAX_THREADS=128`. The `-j` flag, which looks like it controls threads, is actually a GeoJSON feature filter -- not thread count.

**General advice:** Cap thread counts at 128 on Betzy for any CPU-intensive tool.

## A100 Training: Powerful but Scarce

Betzy's `accel` partition has 4 nodes with 4 NVIDIA A100 40GB GPUs each -- 16 GPUs total for the entire cluster. Queue times routinely stretched to days. For our EMB crop segmentation project, we ran DDP training across 4 A100s with `torchrun --nproc_per_node=4`.

The container setup requires two critical environment variables that aren't obvious:

- **`PROJ_DATA`** must point to rasterio's copy of the PROJ database (the container ships two incompatible versions)
- **`PYTHONPATH`** must include the project directory (the container doesn't know about your code)

Miss either one and you get opaque errors -- EPSG lookups fail, or imports break.

## The Storage Split

Betzy has two storage tiers with very different behavior:

- **Project space** (`/cluster/projects/nn12037k/`) -- tiny shared quota, persistent. Use only for code and configs. `rsync` fails here because it creates temp files that exceed quota; use `scp` instead.
- **Work space** (`/cluster/work/users/`) -- large quota, per-cluster. All data, outputs, shards, and checkpoints go here.

We connected them with symlinks so scripts using relative paths from the project directory still find data on work space. Getting this wrong causes `Disk quota exceeded` errors that silently kill long-running jobs.

## Crop Insurance at Scale

For the Sure2 project, we ran 13,510 field assessments as a batch on Betzy's `preproc` partition. Each assessment runs in an isolated Apptainer container with `--containall --cleanenv`, binding only the output directory. The container is self-contained -- Cython-compiled code, embedded data, no external dependencies.

The `--containall` approach eliminates the host environment leak bugs that plagued our Saga runs (PROJ_DATA, PYTHONPATH conflicts). It's more work to set up -- every path and variable must be explicitly bound -- but it removes an entire class of hard-to-debug issues.

Resumability was critical: a Python script builds the field list while skipping completed assessments, and each worker double-checks before processing (race condition guard between parallel workers). Jobs can be cancelled and resubmitted safely at any time.

## Three Things We'd Tell New Betzy Users

1. **Read the partition rules carefully.** `normal` requires 4 nodes minimum. `--mem` must NOT be used on `normal` (full-node allocation handles memory automatically). `accel` allows `--mem`. Getting these wrong wastes hours in failed submissions.

2. **`os.cpu_count()` returns 256, not 128.** Hyperthreading means tools see double the physical core count. This breaks tools that use CPU count for internal sizing. Cap thread counts at 128.

3. **Build skip-if-valid-exists logic into every pipeline step.** Node failures, walltime expiry, and job cancellations are routine. Idempotent steps with output validation save hours of recomputation. Check file existence AND validity (not just "does it exist" but "is the parquet footer intact and row count correct").
