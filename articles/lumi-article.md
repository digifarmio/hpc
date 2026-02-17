# Running Geospatial ML Pipelines on LUMI: Lessons from 100,000 GPU-Hours

*A practitioner's account of training deep learning models for satellite imagery on Europe's fastest supercomputer -- what worked, what broke, and what we wish we'd known.*

---

Running large-scale geospatial ML on HPC is not like running it on a cloud GPU. Over the past year we've spent 100,000 GPU-hours on LUMI training crop classification, cloud removal, and segmentation models on AMD MI250X GPUs. Here's what we learned.

## The Good: CSC Containers Save Days of Setup

LUMI's CSC-provided PyTorch containers are the single best thing about the platform. On other clusters we spent days fighting broken module-provided PyTorch, CUDA stub mismatches, and GDAL/rasterio dependency hell. On LUMI, one line gives you a fully working PyTorch 2.5.1 + ROCm 6.2.4 + Flash Attention 3.0 stack:

```
SIF="/appl/local/csc/soft/ai/images/pytorch_2.5.1_lumi_rocm6.2.4_flash-attn-3.0.0.post1.sif"
```

If you need extra packages, install them to `/projappl/` and add to PYTHONPATH. The container approach eliminated entire categories of environment bugs we hit on Saga and Betzy.

## The Trap: Never Use `--rocm` with Singularity

This cost us hours. Running `singularity exec --rocm` on LUMI injects the host's `libdrm.so`, compiled against GLIBC 2.33. The PyTorch container ships GLIBC 2.31. Result: an immediate crash before any Python code runs.

The fix is to use `SINGULARITY_BIND` to mount exactly the paths you need, replicating what `singularity-AI-bindings/24.03` does minus the `--rocm` flag:

```bash
export SINGULARITY_BIND="/var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/pfs,/scratch,/projappl,/project,/flash,/appl"
```

Complicating matters: `module load` doesn't work on LUMI compute nodes (missing `lua5.3`), so you can't load the bindings module in your SLURM script either. Manual path setup is the only reliable approach.

## AMD MI250X: Looks Like CUDA, Mostly Acts Like CUDA

PyTorch's ROCm integration means `torch.cuda.is_available()` returns True, `model.cuda()` works, and `torch.autocast("cuda", dtype=torch.bfloat16)` works. Use `"cuda"` as the device string everywhere, not `"hip"`.

The big difference is memory behavior. Each MI250X has 2 GCDs (Graphics Compute Dies) that appear as independent GPUs with ~64GB each. Without `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, the memory allocator fragments badly and you OOM at 40GB instead of 60GB. This single environment variable recovered 20GB of usable VRAM.

## The Five Memory Optimizations That Had to Work Simultaneously

Our CFDT model (154M parameters, 16 temporal dates, 256x256 patches) needed ~90GB in fp32 -- far beyond the MI250X's 64GB. After extensive profiling, we applied five optimizations that *only worked when combined*:

1. **bfloat16 mixed precision** -- the biggest single win, roughly halving activation memory
2. **Gradient checkpointing** on the per-date encoder -- trades ~30% more compute for ~40% less activation memory
3. **Reduced-resolution change gate** -- downsample 256x256 to 64x64 for the O(T^2) date-pair computation
4. **Target-only multi-scale features** -- keep full 4-level pyramid only for the target date
5. **Chunked change gate convolution** -- process date pairs in chunks of 64

Total: ~90GB down to ~55GB, fitting comfortably in 64GB. We tried each optimization individually first, and none alone was sufficient. The lesson: profile the full memory budget before deciding what to optimize.

## The Bug That Took Days: GradScaler + bfloat16

This was the most painful bug across all our HPC work. GradScaler, designed for float16 training, silently crippled our bfloat16 training by skipping 75-87% of optimizer steps. The model appeared to train (loss slowly decreased) but barely learned. Validation metrics were frozen near random.

The root cause: GradScaler's initial scale factor (65536) combined with bfloat16's reduced mantissa precision causes frequent false-positive infinity detections in gradients. GradScaler dutifully skips those steps, not knowing the infinities aren't real.

**Rule: if you're using `torch.autocast("cuda", dtype=torch.bfloat16)`, never use GradScaler.** bfloat16 has the same dynamic range as float32 -- there's no underflow risk to protect against.

## DDP on 8 GCDs: Works, But Budget Carefully

Each LUMI node has 4 MI250X GPUs with 2 GCDs each = 8 logical GPUs. `torchrun --standalone --nproc_per_node=8` works out of the box.

The billing catch: 1 wall-hour on `small-g` with 8 GCDs = 8 GPU-hours billed. A 3-day job = 576 GPU-hours. We burned through budget faster than expected until we learned to test on Saga first and only submit validated configurations to LUMI.

## Practical Tips

- **Always use the outer/inner script pattern.** The outer `.sbatch` handles SLURM and container setup; the inner `.sh` runs inside the container with PYTHONPATH, env vars, and Python. This keeps concerns separated and makes debugging much easier.
- **Set `WANDB_MODE=disabled`.** WandB hangs for 30+ minutes inside Singularity containers trying to connect through network restrictions. Monitor via log files and checkpoint metrics instead.
- **Build auto-resume into every training script.** SLURM preemption, walltime expiry, and node failures are routine. Save `latest.pt` every epoch and `best.pt` on validation improvement. With `--requeue`, jobs restart automatically.
- **Test your inner script interactively before submitting.** Catch path and import errors in a `srun --pty bash` session instead of waiting hours in the job queue.
- **Use `/flash/` for ultra-fast temporary storage** when available, but never rely on it persisting.

## The Bottom Line

LUMI is an exceptional platform for large-scale GPU training. The CSC containers and AMD MI250X's 64GB VRAM make ambitious models feasible. The main challenges are AMD-specific quirks (memory allocator, no `--rocm`), billing rates that demand efficiency, and the universal HPC reality that environment variables and container bindings are where most time is lost. Get those right and the actual training just works.
