# Feedback for LUMI / CSC Administrators

*From: digifarm / konstanv (project_465002500)*
*Covering: ~100,000 GPU-hours of geospatial ML training on MI250X (2025-2026)*

---

## What Works Exceptionally Well

### CSC-provided PyTorch containers

The pre-built containers (`pytorch_2.5.1_lumi_rocm6.2.4_flash-attn-3.0.0.post1.sif` and `lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif`) were the single most valuable feature of LUMI. They eliminated days of environment setup that we spent on other clusters dealing with CUDA stub mismatches, GDAL conflicts, and broken module-provided PyTorch. Having Flash Attention 3.0 pre-compiled for MI250X saved significant effort.

### ROCm/HIP transparency

The fact that `torch.cuda.*` maps seamlessly to HIP internals made porting CUDA code trivial. We never had to maintain separate code paths for AMD vs NVIDIA. `torch.autocast("cuda", dtype=torch.bfloat16)` works perfectly.

### Filesystem layout

The scratch/projappl/flash three-tier layout is well-designed. Scratch for active work, projappl for persistent lightweight data (extra pip packages), and flash for temporary high-speed I/O. The separation made it easy to organize projects.

### Internet access from compute nodes

Being able to download data or install packages from within a SLURM job is a significant advantage over NRIS clusters (Saga/Betzy) where compute nodes have no internet. This saved us from complex pre-staging workflows.

---

## Issues and Suggestions

### 1. `--rocm` flag injects incompatible GLIBC library (HIGH priority)

**Problem:** `singularity exec --rocm` injects the host's `libdrm.so`, which requires GLIBC 2.33. The CSC PyTorch containers ship GLIBC 2.31. This causes an immediate crash:

```
/lib/x86_64-linux-gnu/libdrm.so.2: version `GLIBC_2.33' not found
```

**Current workaround:** We use `SINGULARITY_BIND` manually, replicating what `singularity-AI-bindings/24.03` does minus the `--rocm` flag.

**Suggestion:** Either (a) update the container base to include GLIBC 2.33+, (b) update the `singularity-AI-bindings` module to not use `--rocm`, or (c) document this prominently in the LUMI AI documentation. Currently users discover this by trial and error.

### 2. `module load` broken on compute nodes (HIGH priority)

**Problem:** LUMI compute nodes are missing `lua5.3`, so the Lmod module system fails entirely:

```
/usr/share/lmod/lmod/init/bash: line 12: /usr/bin/lua5.3: No such file or directory
```

This means `module load singularity-AI-bindings/24.03` and similar commands cannot be used in SLURM scripts. Users must set all paths manually.

**Suggestion:** Install `lua5.3` on compute nodes, or provide a non-module-based initialization script that users can `source` in SLURM scripts. The current situation forces users to reverse-engineer what modules do and replicate it manually.

### 3. Memory allocator fragmentation without `expandable_segments` (MEDIUM priority)

**Problem:** Without `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, PyTorch on MI250X OOMs at ~40GB instead of the available ~64GB. The memory allocator fragments badly by default.

**Suggestion:** Set this environment variable by default in CSC PyTorch containers, or document it in the AI container documentation. This single variable recovered 20GB of usable VRAM for us.

### 4. WandB hangs in containers (LOW priority)

**Problem:** WandB initialization hangs for 30+ minutes inside Singularity containers, presumably due to network restrictions or proxy settings. Users must set `WANDB_MODE=disabled`.

**Suggestion:** Either document this limitation or configure the container network to allow WandB connections. For many ML researchers, WandB is a primary experiment tracking tool and its absence is felt.

### 5. 42-day scratch purge policy lacks warnings

**Problem:** Files untouched for 42 days on scratch are purged. We lost intermediate data when switching between project phases.

**Suggestion:** Send email notifications before purging, similar to what some other EuroHPC sites do. A 7-day warning would give users time to `touch` files they still need.

### 6. Billing transparency for `small-g`

**Problem:** 1 wall-hour on `small-g` with 8 GCDs = 8 GPU-hours billed. This billing rate isn't obvious from the SLURM output and we burned through budget faster than expected.

**Suggestion:** Include billing rate in the SLURM job start notification, e.g., "This job will consume 8 GPU-hours per wall-hour."

---

## Feature Requests

1. **Pre-installed rasterio/GDAL in a CSC container variant.** Many geospatial ML users need both PyTorch and rasterio/GDAL. Currently these require separate environments (venv + container). A container with both would save significant setup time.

2. **Container recipe documentation.** Publishing the Dockerfiles/def files used to build CSC containers would help users who need to add packages or debug library issues.

3. **Example SLURM scripts for common AI patterns** in the LUMI docs -- single-GPU training, multi-GCD DDP, array jobs for hyperparameter sweeps. The current documentation is thorough but abstract; concrete copy-paste examples would help.

---

## Summary

LUMI is an excellent platform. The CSC containers and MI250X hardware make ambitious ML workloads feasible. The main friction points are the `--rocm` / GLIBC mismatch, broken module system on compute nodes, and a few missing environment defaults. Fixing items #1 and #2 would eliminate the most common pitfalls for new users.

Thank you for maintaining the platform and the container images.
