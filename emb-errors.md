# EMB Project: Error Catalog

Every error we encountered while running the CDL crop segmentation pipeline on Betzy and LUMI, with root causes and fixes. This is the most valuable document in the collection.

---

## Container and Environment Errors

### GLIBC_2.33 not found (LUMI)

**Symptom:**
```
/lib/x86_64-linux-gnu/libdrm.so.2: version `GLIBC_2.33' not found (required by /usr/lib64/libdrm.so.2)
```

**Root cause:** Using `singularity exec --rocm` on LUMI injects the host's `libdrm.so`, which was compiled against GLIBC 2.33. The LUMI PyTorch container ships GLIBC 2.31. The version mismatch causes an immediate crash before any Python code runs.

**Fix:** Remove the `--rocm` flag entirely. Instead, use the `SINGULARITY_BIND` environment variable to mount only the specific paths needed:

```bash
export SINGULARITY_BIND="/var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/pfs,/scratch,/projappl,/project,/flash,/appl"
singularity exec "${CONTAINER}" ...
```

This replicates what the `singularity-AI-bindings/24.03` module does, minus the `--rocm` flag that injects the incompatible library.

---

### PROJ database mismatch (Betzy)

**Symptom:**
```
pyproj.exceptions.CRSError: Invalid projection: EPSG:5070: (Internal Proj Error: proj_create: crs not found)
```

Or more generally: any EPSG lookup returns "unknown" or fails.

**Root cause:** The `emb.sif` container has two copies of the PROJ database:
- rasterio's copy: `/opt/conda/lib/python3.11/site-packages/rasterio/proj_data/proj.db` (PROJ 9.7.1, minor=6)
- pyproj's copy: at pyproj's own path (PROJ 9.5.1, minor=4)

When pyproj loads its older proj.db, it cannot parse entries written for the newer schema. Any CRS lookup by EPSG code fails.

**Fix:** Force both rasterio and pyproj to use rasterio's (newer) PROJ database:

```bash
apptainer exec --bind /cluster \
  --env "PROJ_DATA=/opt/conda/lib/python3.11/site-packages/rasterio/proj_data" \
  "${CONTAINER}" ...
```

---

### ModuleNotFoundError: No module named 'src'

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
```

**Root cause:** The container does not include the project directory on PYTHONPATH. When running `python -m src.train`, Python cannot find the `src` package.

**Fix (Betzy):**
```bash
apptainer exec --env "PYTHONPATH=/cluster/projects/nn12037k/emb" ...
```

**Fix (LUMI):**
```bash
export PYTHONPATH=/projappl/project_465002500/venvs/emb-deps:/scratch/project_465002500/emb:${PYTHONPATH:-}
```

On LUMI you need both the deps venv (for webdataset, pyyaml) and the project dir (for `src`).

---

### Disk quota exceeded (Betzy)

**Symptom:**
```
OSError: [Errno 122] Disk quota exceeded: '/cluster/projects/nn12037k/emb/outputs/shards/...'
```

Or `rsync` fails silently when syncing to project space.

**Root cause:** User `digifarm` has ~1.74 TB quota across ALL `/cluster/projects/` directories. Most of it is consumed by other projects (nn6000k). Writing any significant data to `/cluster/projects/nn12037k/emb/` exceeds the quota.

`rsync` creates temporary files before renaming, so even syncing small files can fail if the temp file creation pushes over quota.

**Fix:**
1. Write all data to work space: `/cluster/work/users/digifarm/emb/`
2. Create symlinks from project space to work space for paths that scripts expect
3. Use `scp` (not rsync) for transferring individual files to project space
4. If rsync is needed to project space, use `--inplace` to avoid temp files

---

### module load fails on LUMI compute nodes

**Symptom:**
```
bash: module: command not found
```
or
```
/usr/share/lmod/lmod/init/bash: line 12: /usr/bin/lua5.3: No such file or directory
```

**Root cause:** LUMI compute nodes do not have `lua5.3` installed, which the Lmod module system requires. The `module` command is simply unavailable.

**Fix:** Do not use `module load` in SLURM scripts. Set all paths manually:

```bash
# Instead of: module load singularity-AI-bindings/24.03
export SINGULARITY_BIND="/var/spool/slurmd,/opt/cray,..."

# Instead of: module load cray-python
# Use the container's Python directly
singularity exec "${CONTAINER}" python ...
```

---

## WebDataset Errors

### TypeError: object of type 'WebLoader' has no len()

**Symptom:**
```
TypeError: object of type 'WebLoader' has no len()
```

**Root cause:** WebDataset creates `IterableDataset` instances, which do not support `len()`. The training loop tried to compute `steps_per_epoch = len(train_loader)` to set up the cosine schedule.

**Fix:** Wrap in try/except and estimate from sample count:

```python
try:
    steps_per_epoch = len(train_loader)
except TypeError:
    # WebDataset IterableDatasets have no len(); estimate from sample count
    steps_per_epoch = data_cfg.get("train_samples", 50000) // batch_size
```

---

### Workers get 0 shards (validation hangs)

**Symptom:** Validation hangs indefinitely or completes instantly with no metrics. Some data-loading workers sit idle.

**Root cause:** WebDataset uses `split_by_worker` (within a rank) and `split_by_node` (across DDP ranks). If the number of shards per rank is less than `num_workers`, some workers get zero shards and produce no data. Example: 8 val shards / 4 ranks = 2 shards per rank. With `num_workers=4`, 2 workers get nothing.

**Fix:** Use `num_workers=0` for validation when shard count is small:

```python
val_wds = WebDatasetLoader(
    shard_urls=data_cfg.get("val_shards", ""),
    ...
    num_workers=0,   # Avoid split_by_worker leaving workers empty
    shuffle_buffer=0,
)
```

---

## Training Numerical Errors

### GradScaler skipping 75-87% of optimizer steps with bfloat16

**This was the single biggest bug in the project.** It was subtle, hard to diagnose, and catastrophically harmful.

**Symptom:** Training appears to run normally but learns extremely slowly. Validation metrics are frozen or barely improving. Loss decreases but at a fraction of the expected rate. Logging shows GradScaler frequently printing "Gradient overflow. Skipping step."

**Root cause:** `torch.amp.GradScaler` was designed for **float16** autocast, which has a narrow dynamic range (max ~65504). GradScaler scales loss up before backward, then checks for inf/NaN gradients before optimizer step. If it finds any, it skips the step and reduces the scale factor.

**bfloat16 has the same dynamic range as float32** (max ~3.4e38). GradScaler is completely unnecessary for bfloat16. But when used with bfloat16, the initial scale factor (65536) combined with bfloat16's reduced mantissa precision causes frequent false-positive inf gradient detections. GradScaler then skips 75-87% of all optimizer steps.

The model barely learns. Validation metrics are nearly random. The training loop reports losses that slowly decrease (because the 13-25% of steps that do execute make some progress), masking the severity of the problem.

**Fix:** Remove GradScaler entirely when using bfloat16 autocast. Replace:

```python
# WRONG: GradScaler with bfloat16
scaler = torch.amp.GradScaler("cuda")
with autocast("cuda", dtype=torch.bfloat16):
    outputs = model(emb)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

With:

```python
# CORRECT: plain backward with bfloat16
with autocast("cuda", dtype=torch.bfloat16):
    outputs = model(emb)
    loss = criterion(outputs, targets)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

GradScaler is ONLY appropriate for `dtype=torch.float16`.

---

### val_loss=NaN from epoch 0 (BatchNorm running stats corruption)

**Symptom:** `val_loss=NaN` starting from the very first epoch. Training loss appears finite.

**Root cause:** This is a downstream consequence of the GradScaler bug. The chain of causation:

1. GradScaler skips the optimizer step (false inf detection)
2. But the forward pass already executed, including BatchNorm
3. BN's `running_mean` and `running_var` are updated during the forward pass (they are updated in-place, regardless of whether the optimizer step happens)
4. When the model produces NaN activations (which GradScaler detected as inf gradients), those NaN values propagate into BN running stats
5. During training, BN uses per-batch statistics (fine, since each batch is computed fresh)
6. During validation, `model.eval()` switches BN to use the corrupted running stats
7. NaN running stats produce NaN outputs for every validation sample

**What made this hard to diagnose:** Sometimes the last training batch before validation would produce valid activations, overwriting the NaN running stats with valid values. So the corruption was intermittent and hard to reproduce.

**Fix attempts (in order of discovery):**

1. **Reset BN stats to (0, 1) before validation.** Problem: only helps if running stats are currently NaN. If the last training batch happened to produce valid stats, the reset doesn't trigger, and the stats are still unreliable.

2. **Force all BN layers to train mode during validation.** This makes BN use per-batch statistics even during validation, bypassing running stats entirely:

```python
model.eval()
for m in model.modules():
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.train()
```

This works for SegFormer. But crashes DeepLab.

3. **BN train mode + skip batches with batch_size < 2.** DeepLab's ASPP module has a global average pooling branch that produces 1x1 spatial output. BN in train mode requires >1 values per channel to compute batch statistics. When the last batch has only 1 sample and hits the ASPP pooling, BN gets a 1x1 spatial tensor and crashes.

```python
# Skip tiny batches in validation
if emb.shape[0] < 2:
    continue
```

4. **Ultimate fix: remove GradScaler** (see above). Without GradScaler, optimizer steps never get skipped, running stats never accumulate NaN, and the whole problem disappears. Fixes 2 and 3 are still in the code as defense-in-depth.

---

### EMA doesn't copy BatchNorm buffers

**Symptom:** The EMA model produces worse results than the base model, or produces NaN. BN running stats in the EMA model are stuck at their initial values (mean=0, var=1).

**Root cause:** The initial EMA implementation only iterated over `model.parameters()`:

```python
# WRONG: only copies learnable parameters
@torch.no_grad()
def update(self, model):
    for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
        ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
```

`model.parameters()` returns only learnable parameters (weights, biases). It does NOT return buffers. BatchNorm's `running_mean`, `running_var`, and `num_batches_tracked` are registered as buffers, not parameters. The EMA model's BN layers kept their initial values (mean=0, var=1) forever.

**Fix:** Also copy buffers:

```python
@torch.no_grad()
def update(self, model):
    for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
        ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
    # Copy buffers (BatchNorm running stats, etc.)
    for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
        ema_b.copy_(model_b)
```

Note: buffers are copied directly (not EMA-smoothed) because BN running stats are already exponentially smoothed by BN's own momentum parameter.

---

### DDP wrapper leaks into validation

**Symptom:** Validation produces slightly different results than expected, or DDP communication errors during validation.

**Root cause:** When EMA is disabled, `eval_model = model` passes the DDP-wrapped model to the validation function. DDP's forward hooks (which synchronize gradients) interfere with `@torch.no_grad()` validation.

**Fix:** Always unwrap the DDP wrapper before validation:

```python
raw_model = model.module if hasattr(model, "module") else model
eval_model = raw_model
if ema is not None:
    ema_model = copy.deepcopy(raw_model)
    ema_model.load_state_dict(ema.state_dict())
    ema_model.to(device)
    eval_model = ema_model

val_metrics = validate(eval_model, val_loader, criterion, device, n_classes_l3)
```

---

### DeepLab train_loss collapses to 0.000

**Symptom:** DeepLab training loss drops to exactly 0.000 within the first few epochs. Model outputs are all NaN.

**Root cause:** The initial learning rate of 5e-4 (copied from SegFormer config) is too aggressive for DeepLab's architecture. DeepLab with ASPP has a different optimization landscape than SegFormer with ConvNeXt. At LR=5e-4, the model diverges within the first epoch.

The reported loss of 0.000 is misleading. All forward passes produce NaN outputs. The loss computation produces NaN. The NaN-safe accumulation (`if math.isfinite(loss_val): epoch_loss += loss_val`) skips all NaN losses, so the denominator is also 0, and `0.0 / max(0, 1)` = 0.0.

**Fix:** Reduce learning rate to 1e-4 for DeepLab:

```yaml
# configs/train/aef_deeplab.yaml
training:
  lr: 1.0e-4   # was 5.0e-4
```

SegFormer is fine at 3e-4. DeepLab needs 1e-4.

---

### NCCL ALLREDUCE timeout (600 seconds)

**Symptom:**
```
RuntimeError: NCCL error in: ../torch/lib/c10d/ProcessGroupNCCL.cpp:XXX, NCCL ALLREDUCE timeout (600000ms)
```

**Root cause:** This is a secondary symptom, not the root cause. When one DDP rank diverges (producing NaN tensors) while other ranks produce valid tensors, the ALLREDUCE operation to synchronize gradients hangs because the ranks have inconsistent data. After 600 seconds, NCCL times out.

**Fix:** Fix the underlying model divergence (usually the GradScaler or LR issues described above). The NCCL timeout is just the messenger.

---

## Data Access Errors

### SAS token expiration for CDL (Planetary Computer)

**Symptom:**
```
CPLE_OpenFailed: /vsicurl/https://planetarycomputer.microsoft.com/.../...: not recognized as a supported file format.
```

Or GDAL/rasterio silently returns empty data.

**Root cause:** Microsoft Planetary Computer SAS tokens expire after approximately 1 hour. During shard building with 256 workers, each worker processes ~728 samples. If the token refresh interval is too large (e.g., 2000), workers run out of valid tokens before refreshing.

**Fix:** Set token refresh interval to 200 (well under the ~728 samples per worker):

```python
# In build_sample_shards.py
SAS_REFRESH_INTERVAL = 200  # NOT 2000
```

Also implement retry-on-error: if a CDL read fails for year <= 2021 (Planetary Computer years), refresh the SAS token and retry once:

```python
def process_sample(self, sample):
    try:
        cdl = self.cdl_reader.read(...)
    except Exception:
        if sample.year <= 2021:
            self.refresh_sas_token()
            cdl = self.cdl_reader.read(...)  # retry once
        else:
            raise
```

---

### Source Cooperative rate limits (AEF)

**Symptom:**
```
HTTP 502 Bad Gateway
```
or
```
CPLE_HttpError: HTTP error code: 502
```

When downloading AEF COGs from `data.source.coop`.

**Root cause:** Source Cooperative limits concurrent HTTP connections to approximately 12-20. With more concurrent workers, the server returns 502 errors.

**Fix:**
1. Limit AEF shard-building workers to 12
2. Implement checkpoint/resume so multi-day builds can be restarted
3. Set GDAL retry options:

```bash
--env "GDAL_HTTP_TIMEOUT=60"
--env "GDAL_HTTP_MAX_RETRY=3"
--env "GDAL_HTTP_RETRY_DELAY=2"
```

---

### AEF COG bottom-up orientation

**Symptom:** Reading AEF embeddings with `rasterio.windows.from_bounds()` returns data from the wrong spatial location, or the data appears vertically flipped.

**Root cause:** AEF COGs have bottom-up (south-up) orientation, indicated by `transform.e > 0` (positive Y pixel size). Most rasterio code assumes north-up (negative Y pixel size). `from_bounds()` computes incorrect window coordinates for bottom-up rasters.

**Fix:** Use pixel-based windowing via `src.index()`:

```python
# WRONG for bottom-up rasters:
window = rasterio.windows.from_bounds(west, south, east, north, src.transform)

# CORRECT:
row_start, col_start = src.index(west, north)  # top-left in pixel coords
row_end, col_end = src.index(east, south)       # bottom-right
window = rasterio.windows.Window.from_slices(
    (min(row_start, row_end), max(row_start, row_end)),
    (min(col_start, col_end), max(col_start, col_end))
)
data = src.read(window=window)
```

---

### TESSERA GeoTIFF endpoint is landmask only

**Symptom:** Reading TESSERA data from the GeoTIFF URL returns a single-band binary mask instead of 128-dimensional embeddings.

**Root cause:** The TESSERA API serves GeoTIFFs that contain ONLY a landmask (1=land, 0=water). The actual embeddings are stored as `.npy` files in a separate directory structure:

```
.../global_0.1_degree_representation/{year}/grid_{lon}_{lat}/
    grid_{lon}_{lat}.npy         # int8, shape (H, W, 128), ~119 MB
    grid_{lon}_{lat}_scales.npy  # float32, shape (H, W), ~3.7 MB
```

**Fix:** Always use the `.npy` files for embeddings. Dequantize with:

```python
raw = np.load(f"grid_{lon}_{lat}.npy")       # int8, (H, W, 128)
scales = np.load(f"grid_{lon}_{lat}_scales.npy")  # float32, (H, W)
embeddings = raw.astype(np.float32) * scales[:, :, np.newaxis]  # broadcast
```

---

## Summary: Error Frequency and Severity

| Error | Severity | Time to diagnose | Cluster |
|-------|----------|-----------------|---------|
| GradScaler + bfloat16 | **Critical** | Days | Both |
| BN running stats corruption | **Critical** | Days | Both |
| GLIBC_2.33 / --rocm flag | High | Hours | LUMI |
| PROJ database mismatch | High | Hours | Betzy |
| Disk quota exceeded | High | Minutes | Betzy |
| EMA missing buffers | Medium | Hours | Both |
| DeepLab LR too high | Medium | Hours | Both |
| SAS token expiration | Medium | Hours | Betzy |
| WebDataset no len() | Low | Minutes | Both |
| Workers get 0 shards | Low | Minutes | Both |
| AEF bottom-up orientation | Low | Hours | Both |
| TESSERA GeoTIFF vs .npy | Low | Minutes | Both |
