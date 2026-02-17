# EMB Project: GPU Training Guide

Practical guide for training CDL crop segmentation models on GPUs. Covers mixed precision, BatchNorm pitfalls, EMA, DDP, and the specific lessons learned from running SegFormer and DeepLab on A100 (Betzy) and MI250X (LUMI).

---

## Architecture Overview

The EMB project trains two model architectures on two embedding sources, for a total of 4 training configurations:

| Config | Embedding | Architecture | in_dim | LR | Batch size |
|--------|-----------|-------------|--------|-----|-----------|
| `aef_segformer.yaml` | AEF (64-D) | SegFormer (ConvNeXt) | 64 | 3e-4 | 32 |
| `aef_deeplab.yaml` | AEF (64-D) | DeepLab v3+ (ASPP) | 64 | 1e-4 | 64 |
| `tessera_segformer.yaml` | TESSERA (128-D) | SegFormer (ConvNeXt) | 128 | 3e-4 | 32 |
| `tessera_deeplab.yaml` | TESSERA (128-D) | DeepLab v3+ (ASPP) | 128 | 1e-4 | 64 |

All configs use:
- Multi-head CE loss: L1 (binary, weight=0.1) + L2 (7 groups, weight=0.3) + L3 (61 classes, weight=1.0)
- Label smoothing: 0.05
- IGNORE_INDEX = 255 for invalid pixels
- AdamW optimizer with weight_decay=0.01
- Cosine schedule with 500-step linear warmup
- Gradient clipping: max_norm=1.0
- Early stopping: patience=15 epochs

---

## Mixed Precision with bfloat16

### The correct approach

Use `torch.amp.autocast` with `dtype=torch.bfloat16` for the forward pass. Do NOT use `GradScaler`.

```python
# Forward pass in mixed precision
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(emb)
    losses = criterion(outputs, targets)
    loss = losses["loss"] / accum_steps

# Backward and optimize -- NO GradScaler
loss.backward()

if (batch_idx + 1) % accum_steps == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
```

### Why NOT GradScaler

GradScaler was designed for **float16**, which has a narrow dynamic range (max ~65504). It scales the loss up before backward to prevent underflow in float16 gradients, then checks for inf/NaN before the optimizer step. If it finds any, it skips the step.

**bfloat16 has the same dynamic range as float32** (max ~3.4e38). There is no underflow risk. GradScaler is unnecessary. But when used with bfloat16, it causes **75-87% of optimizer steps to be skipped** due to false-positive inf gradient detections. This cripples training.

Symptoms of GradScaler + bfloat16:
- Very slow learning (model barely improves over epochs)
- Validation metrics frozen or barely above random
- GradScaler logs "Gradient overflow. Skipping step" frequently
- Model eventually diverges
- BN running stats accumulate NaN (see next section)

**Rule: if your autocast dtype is bfloat16, never use GradScaler.**

### When GradScaler IS appropriate

Only use GradScaler with `dtype=torch.float16`:

```python
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda", dtype=torch.float16):  # float16, not bfloat16
    outputs = model(emb)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

We use bfloat16 everywhere in this project because it is simpler and numerically safer.

---

## BatchNorm Pitfalls in Mixed Precision Training

### The corruption chain

BatchNorm maintains `running_mean` and `running_var` buffers that are updated during every forward pass (during training mode). These updates happen regardless of whether the optimizer step executes.

If GradScaler skips an optimizer step (or if any numerical instability produces NaN activations), the NaN values propagate into BN running stats. During training, BN uses per-batch statistics (computed fresh each forward pass, unaffected). During validation, `model.eval()` switches BN to use the corrupted running stats, producing NaN outputs.

### Defense: BN train mode during validation

Force BN layers to use per-batch statistics even during validation:

```python
@torch.no_grad()
def validate(model, loader, criterion, device, n_classes_l3):
    model.eval()
    # Force BN to train mode -- uses per-batch stats, bypasses running stats
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
    ...
```

### ASPP compatibility (DeepLab)

DeepLab's ASPP module has a global average pooling branch that reduces spatial dimensions to 1x1. BN in train mode requires >1 values per channel to compute batch statistics. If the last validation batch has only 1 sample, the ASPP branch produces a 1x1 spatial tensor, and BN crashes.

**Fix:** Skip batches with batch_size < 2 during validation:

```python
for batch in loader:
    emb = batch["embedding"].to(device, non_blocking=True)
    if emb.shape[0] < 2:
        continue  # BN train mode needs >1 value per channel
    ...
```

This is only needed when BN is forced to train mode. If running stats are healthy (i.e., GradScaler is not being used with bfloat16), `model.eval()` uses running stats and handles batch_size=1 fine.

### The ultimate fix

Remove GradScaler when using bfloat16. Without GradScaler, optimizer steps are never skipped, running stats never accumulate NaN, and the entire corruption chain is broken. The BN-train-mode and batch-skip-2 workarounds remain as defense-in-depth.

---

## EMA (Exponential Moving Average)

### What EMA does

EMA maintains a shadow copy of model weights that is a smoothed average of training weights over time. The EMA model typically generalizes better than the final training weights.

```python
class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
```

### Critical: copy buffers, not just parameters

`model.parameters()` returns only learnable parameters (Conv weights, Linear biases, etc.). It does NOT return buffers. BatchNorm's `running_mean`, `running_var`, and `num_batches_tracked` are buffers.

If EMA only copies parameters, the EMA model's BN layers keep their initial values (mean=0, var=1) forever, producing terrible normalization.

```python
@torch.no_grad()
def update(self, model):
    # EMA-smooth the parameters
    for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
        ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
    # Direct-copy the buffers (BN running stats are already smoothed by BN momentum)
    for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
        ema_b.copy_(model_b)
```

Buffers are copied directly (not EMA-smoothed) because BN's own momentum parameter already provides exponential smoothing of running stats.

### EMA validation

To validate with EMA weights, deep-copy the base model and load EMA state:

```python
raw_model = model.module if hasattr(model, "module") else model
eval_model = raw_model

if ema is not None:
    ema_model = copy.deepcopy(raw_model)
    ema_model.load_state_dict(ema.state_dict())
    ema_model.to(device)
    eval_model = ema_model

val_metrics = validate(eval_model, val_loader, ...)
```

### EMA config

```yaml
training:
  ema:
    enabled: true
    decay: 0.9999    # standard value for ~100-epoch training
```

EMA is disabled on LUMI (`enabled: false`) to reduce memory usage and simplify debugging.

---

## DDP (Distributed Data Parallel)

### Setup

```python
def setup_ddp():
    if "RANK" not in os.environ:
        return 0  # single GPU
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank
```

Launch with `torchrun`:

```bash
# Betzy: 4 A100 GPUs
torchrun --nproc_per_node=4 -m src.train --config configs/train/aef_segformer.yaml

# LUMI: 8 MI250X GCDs
torchrun --standalone --nproc_per_node=8 -m src.train --config configs/train/tessera_segformer_lumi.yaml
```

### Always unwrap before validation

DDP wraps the model with gradient synchronization hooks. These interfere with `@torch.no_grad()` validation. Always unwrap:

```python
raw_model = model.module if hasattr(model, "module") else model
```

### WebDataset shard splitting

WebDataset splits shards across DDP ranks automatically via `wds.split_by_node`:

```python
pipeline = wds.WebDataset(shard_urls, shardshuffle=True, nodesplitter=wds.split_by_node)
```

Requirements:
- Number of shards must be >= number of DDP ranks
- For training: 32 shards / 4 ranks = 8 shards/rank (good)
- For validation: 8 shards / 4 ranks = 2 shards/rank (use `num_workers=0`)

### Gradient accumulation

To simulate larger batches across GPU-memory-limited setups:

```python
accum_steps = config["training"]["gradient_accumulation_steps"]

for batch_idx, batch in enumerate(train_loader):
    loss = compute_loss(...) / accum_steps
    loss.backward()

    if (batch_idx + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
```

Effective global batch size = `batch_size * accum_steps * num_gpus`:
- Betzy: `32 * 1 * 4 = 128`
- LUMI: `16 * 2 * 8 = 256`

---

## Learning Rate by Architecture

| Architecture | LR | Notes |
|-------------|-----|-------|
| SegFormer (ConvNeXt backbone) | 3e-4 | Standard for transformer-like architectures |
| DeepLab v3+ (ASPP) | 1e-4 | Reduced from initial 5e-4 after divergence |

DeepLab at 5e-4 diverges within the first epoch, producing NaN outputs. The NaN-safe loss accumulation reports 0.000 loss, masking the divergence. See error catalog for full analysis.

---

## Cosine Schedule with Warmup

```python
def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_lr / optimizer.defaults["lr"],
                   0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
```

- 500 warmup steps is reasonable for our dataset (~129k train samples, ~4000 steps/epoch at batch=32)
- `min_lr=1e-6` prevents LR from reaching exactly 0
- Schedule steps on every optimizer step (not every batch, when using gradient accumulation)

---

## Channel Normalization

### TESSERA

TESSERA embeddings require per-channel normalization. Statistics are computed from 500 randomly sampled tiles:

```bash
# Compute stats (Phase 3c)
sbatch slurm/phase3_sampling/compute_tessera_stats.sbatch
```

This produces:
- `tessera_channel_mean.npy` -- float32, shape (128,)
- `tessera_channel_std.npy` -- float32, shape (128,)

Stats paths are specified in the training config:

```yaml
data:
  channel_mean_path: "/cluster/work/users/digifarm/emb/configs/data/tessera_channel_mean.npy"
  channel_std_path: "/cluster/work/users/digifarm/emb/configs/data/tessera_channel_std.npy"
```

Normalization is applied during data loading (WebDataset pipeline), not in the model:

```python
def _transform(sample):
    embedding = sample["embedding.npy"].astype(np.float32)
    if channel_mean is not None:
        embedding -= channel_mean[:, None, None]  # (D,) -> (D, 1, 1)
    if channel_std is not None:
        embedding /= (channel_std[:, None, None] + 1e-8)
    ...
```

### AEF

AEF embeddings are unit-length (L2-normalized) int8 vectors. They do not use channel normalization. The model config has `input_norm: true` which applies learned layer normalization inside the model.

**Important:** AEF vectors must be re-normalized after pooling to 30m resolution (if done). Averaging unit-length vectors does not produce a unit-length vector.

### Stats must be on same filesystem as training

LUMI and Betzy have different filesystem paths. Stats files must exist at the path specified in the config:

```yaml
# Betzy config
channel_mean_path: "/cluster/work/users/digifarm/emb/configs/data/tessera_channel_mean.npy"

# LUMI config
channel_mean_path: "/scratch/project_465002500/emb/configs/data/tessera_channel_mean.npy"
```

Copy stats when moving between clusters.

---

## Loss Function

Multi-head cross-entropy over three label hierarchy levels:

```python
class MultiHeadCELoss(nn.Module):
    def forward(self, outputs, targets):
        loss_l1 = self.ce_l1(outputs["l1"], targets["label_l1"])  # binary
        loss_l2 = self.ce_l2(outputs["l2"], targets["label_l2"])  # 7 groups
        loss_l3 = self.ce_l3(outputs["l3"], targets["label_l3"])  # 61 classes
        total = 0.1 * loss_l1 + 0.3 * loss_l2 + 1.0 * loss_l3
        return {"loss": total, "loss_l1": loss_l1, "loss_l2": loss_l2, "loss_l3": loss_l3}
```

All CE losses use `ignore_index=255` (IGNORE_INDEX) to mask invalid pixels (nodata in embeddings or labels).

---

## Checkpointing

### What is saved

```python
state = {
    "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": epoch,
    "step": global_step,
    "best_metric": best_metric,
    "config": config,
}
if ema is not None:
    state["ema"] = ema.state_dict()
```

### Checkpoint schedule

- `best.pt`: saved whenever `val_miou` exceeds previous best
- `epoch_NNNN.pt`: saved every 10 epochs (configurable via `save_every`)
- `last.pt`: saved at end of training

### Resume

```yaml
training:
  resume: "/cluster/work/users/digifarm/emb/outputs/runs/tessera_segformer/last.pt"
```

---

## Monitoring

### Wandb (Betzy only)

```yaml
wandb:
  enabled: true
  project: "cdl-segmentation"
  name: "tessera_segformer"
```

Disabled on LUMI by default. Can be enabled if wandb is installed in the deps venv.

### Console logging

Training logs every 50 steps (brief) and every 500 steps (detailed):

```
2025-01-15 10:23:45 [INFO] src.train: Epoch 3  step 12500  loss=1.2345  lr=2.85e-04
```

Validation logs after each epoch:

```
2025-01-15 10:30:12 [INFO] src.train: Epoch 3  train_loss=1.1234  val_loss=1.3456  mIoU=0.2345  macro-F1=0.3456  OA=0.6789  [423.1s]
```

### SLURM log locations

```
# Betzy
/cluster/work/users/digifarm/emb/logs/train_{JOBID}.log
/cluster/work/users/digifarm/emb/logs/train_{JOBID}.err

# LUMI
/scratch/project_465002500/emb/logs/train_{JOBID}.log
/scratch/project_465002500/emb/logs/train_{JOBID}.err
```

---

## Quick Reference: Training Commands

### Betzy (4x A100)

```bash
# Submit training
sbatch --export=CONFIG=configs/train/tessera_segformer.yaml \
  slurm/phase4_training/train.sbatch

# Interactive debugging
srun --partition=accel --gpus=1 --mem=32G --time=4:00:00 --account=nn12037k --pty bash
cd /cluster/projects/nn12037k/emb
apptainer exec --nv --bind /cluster containers/emb.sif \
  python -m src.train --config configs/train/tessera_segformer.yaml
```

### LUMI (8x MI250X GCDs)

```bash
# Submit training
sbatch --export=CONFIG=configs/train/tessera_segformer_lumi.yaml \
  slurm/phase4_training/train_lumi.sbatch

# Interactive debugging
srun --partition=small-g --gpus-per-node=1 --mem=64G --time=4:00:00 --account=project_465002500 --pty bash
cd /scratch/project_465002500/emb
export SINGULARITY_BIND="/var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/pfs,/scratch,/projappl,/project,/flash,/appl"
singularity exec /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
  bash -c "export PYTHONPATH=/projappl/project_465002500/venvs/emb-deps:/scratch/project_465002500/emb && python -m src.train --config configs/train/tessera_segformer_lumi.yaml"
```
