# Running Satellite Pipelines on Saga: Fast Iteration, Fragile Infrastructure

*How we used Norway's Saga cluster for geospatial data preparation and small-model validation -- and why we moved production work elsewhere.*

---

Saga, operated by NRIS/Sigma2 in Norway, served as our development cluster for two geospatial projects: satellite-based crop insurance assessment (Sure2) and cloud removal model validation (vcloud). Its strength was fast iteration; its weakness was everything else.

## The Role: Development and Data Prep

Saga was never our production training cluster -- that was LUMI. Instead, Saga served three purposes:

1. **Data preparation** -- downloading and processing the SEN12MS-CR-TS benchmark dataset (33GB), building Sentinel-2 COG processing pipelines, and downloading ERA5-Land climate data
2. **Small-model validation** -- running a reduced version of our cloud removal model (64x64 patches, 4 temporal dates) on P100 GPUs to validate architecture choices before committing LUMI GPU-hours
3. **Crop insurance pipeline development** -- iterating on planting estimation, emergence detection, and drought computation across 13,510 Romanian corn fields

## What Made Saga Good for Iteration

Saga's `devel` QOS lets you run 2-hour interactive jobs with quick scheduling. For a pipeline that processes satellite imagery field-by-field, this was ideal -- run 100 fields, inspect results, fix bugs, resubmit. The turnaround cycle was minutes, not the hours we'd face on LUMI.

The cluster also has good internet connectivity from compute nodes and GDAL CLI tools available system-wide, making it straightforward to test data download and rasterio-based processing pipelines.

## The Environment Setup Pain

Saga's module-provided PyTorch was broken. The `PyTorch/1.12.1-foss-2022a-CUDA-11.7.0` module contained only CUDA stub libraries, not runtime libraries. We spent a full day debugging `torch.cuda` failures before discovering we needed to pip-install PyTorch in a venv instead.

The venv approach on Saga is fragile. Module conflicts between pip packages and system-provided packages caused cryptic errors unless we religiously ran `module purge` before every session. This is the kind of thing containers solve completely -- LUMI's CSC containers never had this problem.

## PROJ_DATA: The Silent Coordinate Killer

Across all our HPC work, the PROJ database mismatch was one of the most dangerous bugs because it fails *silently*. The system proj.db on Saga nodes has an old schema version. When rasterio uses it for coordinate transforms, the results are subtly wrong -- not errors, just incorrect coordinates. Fields don't overlap with Sentinel-2 tiles. EVI reads return all zeros. Everything looks like a data problem.

We spent 4 hours debugging this before discovering the fix: point `PROJ_DATA` to the container's own copy:

```bash
export PROJ_DATA="/usr/local/lib/python3.12/dist-packages/rasterio/proj_data"
```

This applies to any container-based rasterio workflow on NRIS clusters.

## The Drought Pre-Loading Pattern

Our most impactful performance optimization came from rethinking I/O patterns for the Lustre filesystem. The naive approach -- open a climate NetCDF file, extract one point, close, repeat for 13,510 fields -- hits Lustre's random I/O latency at ~100ms per read. That's 1.35 million file operations per year of data.

The pre-loading approach loads all ERA5-Land and CHIRPS data into RAM (~700MB), then does pure numpy point extraction at ~0.1ms per field. Result: 6-7 fields/second, ~8 minutes per year for 2,700 fields. The insight is general: on Lustre, batch your I/O into large sequential reads, then process in memory.

## The Bugs That Mattered

**Threading segfaults.** GDAL and HDF5 C libraries in our container build were not thread-safe. `ThreadPoolExecutor` with rasterio caused immediate segfaults. The fix was sequential processing -- `ProcessPoolExecutor` (separate processes) works fine, but threads sharing GDAL state don't.

**Lustre stale file handles.** Replacing a `pylibs/` directory while running jobs reference it causes `OSError: Stale file handle`. On Lustre, treat pip install targets as immutable -- create a fresh `pylibs3/` instead of modifying `pylibs/` in place.

**Python output buffering.** SLURM log files stayed empty for hours, then dumped everything at once. Python stdout is fully buffered when piped to a file. The fix: `python3 -u` or `export PYTHONUNBUFFERED=1`.

## GPU Availability: The Dealbreaker

GPU nodes on Saga were frequently down or in drain state. Jobs sat in the `accel` queue for days waiting for a working P100 node. This was the single biggest frustration. When GPU nodes were available, Saga was great; when they weren't, it was unusable.

Check `sinfo -p accel` before submitting GPU jobs. If everything shows `down` or `drain`, switch clusters immediately rather than hoping it'll recover.

## The Verdict

Saga was excellent for CPU-bound data preparation and iterative pipeline development. Its fast scheduling, internet access, and GDAL availability made it ideal for geospatial ETL work. But the broken module PyTorch, fragile venv approach, intermittent GPU availability, and 42-day file purge policy made it unsuitable as a primary training cluster.

Our best pattern: develop and test on Saga, validate small models on Saga GPU nodes (when available), then run production training on LUMI. Always test with 10 items before submitting 13,510.
