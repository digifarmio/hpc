# HPC Usage Guide for Geospatial ML Pipelines

Practical guide for running large-scale geospatial ML pipelines on European HPC systems. Covers real workflows, real errors, and real fixes -- not a generic HPC tutorial.

These documents were created from the experience of running [Claude Code](https://claude.ai/code) (Opus 4.6) on several geospatial projects over a few weeks in January/February 2026:

- **romcrop/vcloud** -- TESSERA embedding generation + CatBoost crop classification (LUMI)
- **Sure2** -- satellite-based crop emergence insurance engine (Saga, Betzy)
- **EMB** -- CDL crop segmentation from foundation model embeddings (Betzy A100, LUMI MI250X)
- **ALFA** -- land cover ML inference at scale (Betzy, Saga)

---

## Systems Covered

| System | Country | Operator | GPUs | Container Runtime | Internet from Compute |
|--------|---------|----------|------|-------------------|-----------------------|
| LUMI | Finland | CSC | AMD MI250X (ROCm) | `singularity` | Yes |
| Saga | Norway | NRIS/Sigma2 | NVIDIA P100 (CUDA) | `apptainer` | No |
| Betzy | Norway | NRIS/Sigma2 | NVIDIA A100 (CUDA) | `apptainer` | No |

---

## Documentation

### User Guides

Step-by-step recipes for common tasks on each cluster:

- **[How-To: LUMI](guides/howto-lumi.md)** -- Container setup, GPU bindings, single-GPU and 8-GCD DDP jobs, bfloat16 mixed precision, interactive sessions, common errors
- **[How-To: Saga](guides/howto-saga.md)** -- Venv/container setup, CPU and GPU jobs, PROJ_DATA fix, pre-loading pattern, resumable batch processing
- **[How-To: Betzy](guides/howto-betzy.md)** -- Storage layout, A100 GPU training, 4-node `normal` partition pattern, `/dev/shm` for temp files, batch jobs with resumability

### Tips & Pitfalls

- **[Tips & Pitfalls](tips.md)** -- SSH config, rsync gotchas, SLURM patterns, container vs venv, cluster selection guide, and 14 common mistakes ranked by pain

### Articles

In-depth write-ups on running real workloads on each system:

- **[LUMI Article](articles/lumi-article.md)** -- CSC containers, `--rocm` trap, MI250X memory optimization, GradScaler+bfloat16 bug, DDP billing
- **[Saga Article](articles/saga-article.md)** -- Fast iteration strengths, broken module PyTorch, PROJ_DATA silent coordinate corruption, Lustre pre-loading pattern
- **[Betzy Article](articles/betzy-article.md)** -- 4-node constraint as parallelism, Lustre vs `/dev/shm`, hyperthreading bugs, A100 training, storage split

### Admin Feedback

Detailed feedback for cluster operators based on our experience:

- **[Feedback: LUMI / CSC](articles/feedback-lumi-csc.md)** -- `--rocm` GLIBC mismatch, broken module system on compute nodes, memory allocator defaults, WandB restrictions
- **[Feedback: Sigma2 / NRIS (Saga & Betzy)](articles/feedback-sigma2-nris.md)** -- Broken PyTorch module, GPU availability, Lustre I/O limitations, Betzy partition rules, login/infrastructure issues
