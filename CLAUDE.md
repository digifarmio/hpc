# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is an **operational knowledge base** (not a software project) documenting real-world HPC deployment of geospatial ML pipelines on European supercomputers. There is no source code, build system, or test suite — only markdown guides recording workflows, errors, and fixes.

## File Organization

| File | Scope |
|------|-------|
| `README.md` | Index, cross-cluster quick reference, SLURM essentials |
| `lumi.md` | Primary training cluster (AMD MI250X, ROCm, CSC containers) |
| `saga.md` | Development cluster (Apptainer, no-internet compute nodes) |
| `betzy.md` | CPU/backup cluster |
| `sure2.md` | Crop insurance pipeline — 10 production bugs with fixes |
| `emb-betzy.md` | EMB project on Betzy (A100 GPUs, data prep + DDP training) |
| `emb-lumi.md` | EMB project on LUMI (MI250X, ROCm container, DDP on 8 GCDs) |
| `emb-errors.md` | **Critical reference** — 16 errors with root causes, diagnosis times, and fixes |
| `emb-training.md` | GPU training specifics (mixed precision, BatchNorm, EMA, learning rates) |
| `fields-pipeline.md` | US crop field tile pipeline (29.1M polygons, tippecanoe, PMTiles) |
| `tips.md` | Cross-cluster patterns, SSH config, common mistakes ranked by pain |

## Projects Documented

- **romcrop/vcloud**: TESSERA embedding generation (LUMI GPU) + CatBoost crop classification
- **Sure2**: Satellite-based crop emergence insurance (Saga/Betzy, Apptainer containers)
- **EMB**: CDL crop segmentation from foundation model embeddings (Betzy A100 + LUMI MI250X)
- **Other small projects**: US crop field history tile pipeline, ALFA land cover inference

## HPC Systems

| System | GPUs | Container Runtime | Key Constraint |
|--------|------|-------------------|----------------|
| LUMI (Finland, CSC) | AMD MI250X (ROCm) | `singularity` | No `--rocm` flag (causes GLIBC mismatch) |
| Saga (Norway, NRIS) | None used | `apptainer` | No internet from compute nodes |
| Betzy (Norway, NRIS) | NVIDIA A100 (CUDA) | `apptainer` | 4-node minimum on GPU partition |

## Key Patterns When Editing These Docs

- Every error entry follows the format: **Symptom** → **Root cause** → **Fix** (with diagnosis time)
- Code blocks contain real SLURM scripts, container commands, and Python snippets — keep them accurate
- Cross-references between files use relative markdown links (e.g., `[LUMI.md](LUMI.md)`)
- Environment variable settings are critical — small changes (e.g., `PROJ_DATA` path, `SINGULARITY_BIND` list) can break pipelines
- Performance numbers are real benchmarks — do not round or approximate when editing

## Most Important Operational Lessons

These are documented across multiple files and are the hardest-won knowledge:

1. **GradScaler + bfloat16 = disaster** — skips 75-87% of optimizer steps (emb-training.md, emb-errors.md)
2. **LUMI: never use `--rocm` flag** — use `SINGULARITY_BIND` instead (emb-lumi.md, emb-errors.md)
3. **rsync `--exclude 'data'` vs `--exclude '/data'`** — without leading slash, silently deletes source code (tips.md)
4. **PROJ_DATA must point to rasterio's copy** when container has two proj.db versions (emb-errors.md)
5. **All 5 memory optimizations needed simultaneously** — none sufficient alone for MI250X 64GB (tips.md, lumi.md)
