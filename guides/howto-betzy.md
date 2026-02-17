# How-To Guide: Betzy

Step-by-step recipes for common tasks on Betzy (NRIS/Sigma2, Norway). A100 GPUs, Apptainer containers, 4-node minimum on `normal` partition.

---

## Connect to Betzy

```bash
# First time: add to ~/.ssh/config
Host betzy
    HostName betzy.sigma2.no
    User digifarm
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 7d

# Create socket directory
mkdir -p ~/.ssh/sockets

# Connect (OTP-based auth via authenticator app)
ssh betzy
```

---

## Transfer Code to Betzy

### To work space (large files, data, outputs)

```bash
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '/data' --exclude 'outputs/' \
    /home/ubuntu/myproject/ betzy:/cluster/work/users/digifarm/myproject/
```

### To project space (code only -- tight quota!)

`rsync` creates temp files that can exceed quota. Use `scp` or `rsync --inplace`:

```bash
# Individual files
scp myfile.py betzy:/cluster/projects/nn12037k/myproject/src/

# Bulk sync (code only, --inplace avoids temp files)
rsync -avz --inplace \
    --exclude '.git' --exclude '__pycache__' --exclude 'data/' --exclude 'outputs/' \
    --exclude '*.parquet' --exclude '*.sif' \
    /home/ubuntu/myproject/ betzy:/cluster/projects/nn12037k/myproject/
```

---

## Understand Betzy's Storage Layout

```
/cluster/projects/nn12037k/myproject/    # Project space: code, configs, container
    src/
    configs/
    containers/mycontainer.sif
    data/ -> /cluster/work/.../data/     # Symlink to work space

/cluster/work/users/digifarm/myproject/  # Work space: all data and outputs
    data/
    outputs/
    logs/
```

| Path | Quota | Use for | Watch out |
|------|-------|---------|-----------|
| `/cluster/projects/nn12037k/` | ~1.74TB shared | Code, configs, .sif | rsync fails; use scp |
| `/cluster/work/users/digifarm/` | Large | Data, outputs, logs | 42-90 day purge |

Create symlinks to bridge them:

```bash
ln -s /cluster/work/users/digifarm/myproject/data /cluster/projects/nn12037k/myproject/data
ln -s /cluster/work/users/digifarm/myproject/outputs /cluster/projects/nn12037k/myproject/outputs
```

---

## Build an Apptainer Container

```bash
ssh betzy
cd /cluster/work/users/digifarm/myproject/containers

# From definition file
apptainer build mycontainer.sif mycontainer.def

# From Docker tarball
apptainer build mycontainer.sif docker-archive://myimage.tar.gz
```

If the definition uses `pip install`, add `--break-system-packages` for modern Python base images (PEP 668).

---

## Run a GPU Training Job (A100)

```bash
#!/bin/bash
#SBATCH --job-name=my_training
#SBATCH --account=nn12037k
#SBATCH --partition=accel
#SBATCH --time=7-00:00:00
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p /cluster/work/users/digifarm/myproject/logs

PROJECT_DIR=/cluster/projects/nn12037k/myproject
CONTAINER=${PROJECT_DIR}/containers/mycontainer.sif

apptainer exec --nv --bind /cluster \
  --env "PYTHONPATH=${PROJECT_DIR}" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PROJ_DATA=/opt/conda/lib/python3.11/site-packages/rasterio/proj_data" \
  "${CONTAINER}" \
  torchrun --nproc_per_node=4 -m src.train --config configs/train/model.yaml
```

Key notes:
- `--nv` enables NVIDIA GPU passthrough (Betzy uses `--nv`, LUMI does NOT)
- `--mem=0` requests all memory on the node
- `accel` partition: 4 nodes, 4 A100 40GB GPUs each
- Queue times can be days -- validate configs thoroughly before submitting

---

## Run a CPU Job on `preproc` Partition

For I/O-bound data processing:

```bash
#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --account=nn12037k
#SBATCH --partition=preproc
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

apptainer exec --bind /cluster \
  --env "PYTHONPATH=/cluster/projects/nn12037k/myproject" \
  --env "PROJ_DATA=/opt/conda/lib/python3.11/site-packages/rasterio/proj_data" \
  --env "GDAL_HTTP_TIMEOUT=60" \
  --env "GDAL_HTTP_MAX_RETRY=3" \
  /cluster/projects/nn12037k/myproject/containers/mycontainer.sif \
  python -m src.prepare_data
```

---

## Run a Multi-Node Job on `normal` Partition

Betzy's `normal` partition requires `--nodes >= 4`. Process 4 items in parallel:

```bash
#!/bin/bash
#SBATCH --job-name=parallel_job
#SBATCH --account=nn12037k
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --time=4-00:00:00
# NOTE: Do NOT use --mem on normal partition!

set -uo pipefail

ITEMS=(item1 item2 item3 item4 item5 item6 item7 item8)

for item in "${ITEMS[@]}"; do
  # Wait if all 4 nodes are busy
  while (( $(jobs -r | wc -l) >= 4 )); do
    wait -n 2>/dev/null || true
  done

  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=128 \
    bash process_item.sh "$item" \
    > "logs/${item}.log" 2>&1 &
done

wait
echo "All items complete"
```

**Critical rules for `normal` partition:**
- `--nodes >= 4` (minimum 4 nodes)
- Do NOT use `--mem` (full-node allocation provides all memory automatically)
- Each node has 128 physical cores, 256 logical CPUs, ~242GB RAM

---

## Use /dev/shm for Temp-File-Heavy Tools

Lustre (`/cluster/work/`) has catastrophic latency for small random I/O. Tools like tippecanoe, SQLite, and anything creating hundreds of temp files must use RAM-backed storage:

```bash
# Use /dev/shm (126GB per node)
TMPDIR="/dev/shm/mytool_$$"
mkdir -p "$TMPDIR"

# Run tool with temp dir on /dev/shm
TIPPECANOE_MAX_THREADS=128 tippecanoe \
    --temporary-directory="$TMPDIR" \
    -o output.mbtiles \
    -l layer_name \
    ...

rm -rf "$TMPDIR"
```

**Important:** Set `TIPPECANOE_MAX_THREADS=128` on Betzy. Without it, tippecanoe crashes with "shards not a power of 2" because `os.cpu_count()` returns 256 (hyperthreads) and the internal shard calculation fails.

Betzy filesystem summary:
- `/dev/shm` -- 126GB RAM tmpfs, perfect for temp files
- `/tmp` -- 1.5GB, useless for most workloads
- `/cluster/work/` -- Lustre, great for large sequential I/O, terrible for small random I/O

---

## Run a Batch Job with Resumability

Process many items in parallel with skip-if-complete logic:

### Python: build work list

```python
import os, glob

completed = 0
work_list = []
for item_id in all_items:
    if glob.glob(f"outputs/{item_id}/result_*.json"):
        completed += 1
        continue
    work_list.append(item_id)

print(f"Skip {completed}, process {len(work_list)}")
with open("remaining.txt", "w") as f:
    for item in work_list:
        f.write(f"{item}\n")
```

### Bash: parallel processing with xargs

```bash
process_item() {
    local item_id="$1"
    local output_dir="outputs/${item_id}"
    mkdir -p "$output_dir"

    # Race condition guard
    if ls "$output_dir"/result_*.json &>/dev/null 2>&1; then
        return 0
    fi

    apptainer exec --bind /cluster "$CONTAINER" \
        python -m src.process --item "$item_id" --output "$output_dir"
}
export -f process_item

cat remaining.txt | xargs -P 15 -I {} bash -c 'process_item "$@"' _ {}
```

---

## Start an Interactive Session

### CPU

```bash
srun --account=nn12037k --partition=preproc --mem=16G --time=01:00:00 --pty bash
```

### GPU

```bash
srun --account=nn12037k --partition=accel --gpus=1 --mem=32G --time=04:00:00 --pty bash
```

Verify GPU inside container:

```bash
apptainer exec --nv --bind /cluster containers/mycontainer.sif \
    python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA A100-SXM4-40GB
```

---

## Set Up AWS Credentials

AWS credentials propagate into Apptainer containers automatically (they inherit `~/.aws/`):

```bash
ssh betzy
mkdir -p ~/.aws
cat >> ~/.aws/credentials <<'EOF'
[myprofile]
aws_access_key_id = AKIA...
aws_secret_access_key = ...
EOF
```

Inside containers:

```bash
apptainer exec --bind /cluster "$CONTAINER" \
    aws s3 cp output.pmtiles s3://mybucket/tiles/ --profile myprofile
```

---

## Load Modules in SLURM Scripts

Modules aren't available by default in non-login shells:

```bash
# Option 1: source module init
source /etc/profile.d/modules.sh
module load CUDA/12.9.1

# Option 2: bash login shell
bash -l -c "module load CUDA/12.9.1 && python train.py"
```

Usually unnecessary if using containers (they ship their own CUDA).

---

## Monitor Jobs

```bash
# Check queue
squeue -u digifarm

# Detailed format
squeue -u digifarm --format="%.10i %.9P %.20j %.8T %.10M %.6D %R"

# Post-mortem stats
sacct -j JOBID --format=JobID,Elapsed,MaxRSS,MaxVMSize,State

# Follow per-item log
tail -f logs/item1.log

# Check billing quota
cost -p nn12037k

# Debug a running process on a compute node
srun --overlap --jobid=JOBID --nodelist=NODE --ntasks=1 bash -c 'ps aux | grep python'

# Check /dev/shm usage on a compute node
srun --overlap --jobid=JOBID --nodelist=NODE --ntasks=1 bash -c 'df -h /dev/shm'
```

---

## Partition Quick Reference

| Partition | `--mem` | `--gpus` | Min nodes | Max wall-time | Best for |
|-----------|---------|----------|-----------|---------------|----------|
| `accel` | OK (`--mem=0` for all) | Required | 1 | 7 days | GPU training (A100) |
| `preproc` | OK | No | 1 | 1 day | I/O-bound data prep |
| `normal` | **DO NOT use** | No | **4** | 4 days | Large CPU jobs |

---

## Common Errors and Fixes

| Error | Fix |
|-------|-----|
| `Disk quota exceeded` on project space | Write data to `/cluster/work/`; symlink from project space |
| `rsync` fails to project space | Use `scp` or `rsync --inplace` |
| `--mem` crashes job on `normal` | Remove `--mem`; full-node allocation handles it |
| `shards not a power of 2` (tippecanoe) | `export TIPPECANOE_MAX_THREADS=128` |
| Tippecanoe hangs at 1 thread | Use `/dev/shm` for `--temporary-directory`, not Lustre |
| `EPSG:5070 not found` (PROJ) | Set `PROJ_DATA` to rasterio's proj_data path inside container |
| `ModuleNotFoundError: src` | Set `PYTHONPATH` to project dir in apptainer `--env` |
| Job rejected: `--nodes >= 4` | Use `preproc` for 1-node, or redesign for 4-node parallelism |
| `os.cpu_count()` returns 256 | Cap threads at 128 (physical cores); hyperthreading misleads tools |
