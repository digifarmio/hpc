# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is an **operational knowledge base** (not a software project) documenting real-world HPC deployment of geospatial ML pipelines on European supercomputers. There is no source code, build system, or test suite -- only markdown guides recording workflows, errors, and fixes.

Created from the experience of running Claude Code (Opus 4.6) on several geospatial projects in January/February 2026.

## File Organization

```
README.md                          # Index with project list and doc links
tips.md                            # Cross-cluster tips, pitfalls, common mistakes
guides/
  howto-lumi.md                    # Step-by-step recipes for LUMI
  howto-saga.md                    # Step-by-step recipes for Saga
  howto-betzy.md                   # Step-by-step recipes for Betzy
articles/
  lumi-article.md                  # In-depth article on LUMI experience
  saga-article.md                  # In-depth article on Saga experience
  betzy-article.md                 # In-depth article on Betzy experience
  feedback-lumi-csc.md             # Admin feedback for CSC/LUMI
  feedback-sigma2-nris.md          # Admin feedback for NRIS/Sigma2
```

## HPC Systems

| System | GPUs | Container Runtime | Key Constraint |
|--------|------|-------------------|----------------|
| LUMI (Finland, CSC) | AMD MI250X (ROCm) | `singularity` | No `--rocm` flag (causes GLIBC mismatch) |
| Saga (Norway, NRIS) | NVIDIA P100 (CUDA) | `apptainer` | No internet from compute nodes |
| Betzy (Norway, NRIS) | NVIDIA A100 (CUDA) | `apptainer` | 4-node minimum on `normal` partition |

## Key Patterns When Editing These Docs

- Code blocks contain real SLURM scripts, container commands, and Python snippets -- keep them accurate
- Environment variable settings are critical -- small changes (e.g., `PROJ_DATA` path, `SINGULARITY_BIND` list) can break pipelines
- Performance numbers are real benchmarks -- do not round or approximate when editing

## Most Important Operational Lessons

1. **GradScaler + bfloat16 = disaster** -- skips 75-87% of optimizer steps
2. **LUMI: never use `--rocm` flag** -- use `SINGULARITY_BIND` instead
3. **rsync `--exclude 'data'` vs `--exclude '/data'`** -- without leading slash, silently deletes source code
4. **PROJ_DATA must point to rasterio's copy** when container has two proj.db versions
5. **Lustre is terrible for small random I/O** -- use `/dev/shm` for temp-file-heavy tools
