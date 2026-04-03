# Running Llamole on AWS SageMaker

This guide documents how to run Llamole eval and training as SageMaker Training Jobs.

## Why SageMaker Training Jobs?

Llamole requires a 7B parameter model (Qwen2-7B, Llama-3.1-8B, or Mistral-7B). In bfloat16:
- **Inference/eval**: ~14 GB VRAM → single A10G (24 GB) is fine
- **Training with LoRA**: ~34 GB VRAM → needs 4x A10G or an A100

SageMaker Training Jobs spin up on demand, run to completion, and terminate — no idle cost.

---

## Prerequisites

### 1. AWS credentials

Configure an AWS profile (e.g. `genai`) with SageMaker permissions:

```bash
aws configure --profile genai
```

### 2. SageMaker config

Copy the example config and fill in your values:

```bash
cp sagemaker/.env.example sagemaker/.env
# Edit sagemaker/.env with your role ARN, S3 bucket, VPC subnet IDs, security group ID
```

Required values:
- `SAGEMAKER_ROLE` — IAM role ARN with SageMaker execution permissions
- `SAGEMAKER_S3_BUCKET` — S3 bucket for job outputs and source code
- `SAGEMAKER_VPC_SUBNETS` — comma-separated subnet IDs (private subnets with NAT gateway for internet access)
- `SAGEMAKER_VPC_SG` — security group ID

> **Note:** `sagemaker/.env` is gitignored and must never be committed.

### 3. Python dependencies (local, for submission only)

```bash
pip install boto3 sagemaker
```

### 4. HuggingFace token

Set your HF token (needed to download Qwen2-7B-Instruct and graph model weights):

```bash
export HF_TOKEN=hf_...
```

---

## Submitting Jobs

### Eval job

Runs inference on the example material dataset (~5 examples):

```bash
python sagemaker/submit_job.py --mode eval
# or with a specific config:
python sagemaker/submit_job.py --mode eval --config qwen_drug
```

Instance: `ml.g5.2xlarge` (1x A10G, 24 GB VRAM) — ~$1.21/hr  
Typical duration: ~15 min (including model download)  
Cost: ~$0.30

### Training job (smoke test)

Runs 1 epoch of SFT on 61 examples to verify the training pipeline:

```bash
python sagemaker/submit_job.py --mode train
# or with a full training config:
python sagemaker/submit_job.py --mode train --config qwen_lora
```

Instance: `ml.g5.12xlarge` (4x A10G, 96 GB VRAM) — ~$5.67/hr  
Typical duration: smoke test ~30 min, full training ~3 hrs  
Cost: smoke test ~$3, full training ~$17

### Options

```
--mode eval|train         What to run (default: eval)
--config NAME             Config name without .yaml (default: qwen_material / qwen_lora_smoketest)
--instance INSTANCE_TYPE  Override instance type
--wait                    Block until job completes and stream logs
--hf-token TOKEN          HuggingFace token (or use HF_TOKEN env var)
```

---

## Monitoring Jobs

Check status:
```bash
aws --profile genai sagemaker describe-training-job \
  --region us-east-1 \
  --training-job-name <job-name> \
  --query '{Status:TrainingJobStatus,Secondary:SecondaryStatus}'
```

Stream logs:
```bash
# Find log stream name
aws --profile genai logs describe-log-streams \
  --region us-east-1 \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix <job-name> \
  --query 'logStreams[*].logStreamName'

# Tail logs
aws --profile genai logs get-log-events \
  --region us-east-1 \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name <job-name>/algo-1-<timestamp> \
  --limit 50 \
  --query 'events[*].message' --output text
```

---

## Retrieving Results

### Eval results

Eval results are currently printed to CloudWatch logs only. To retrieve them:

```bash
# Download the output tarball
aws --profile genai s3 cp \
  s3://<bucket>/llamole/output/<job-name>/output/output.tar.gz \
  /tmp/llamole-results/

tar -xzf /tmp/llamole-results/output.tar.gz -C /tmp/llamole-results/
```

### Training results (LoRA adapter)

After a successful training job, the LoRA adapter is saved to S3:

```bash
aws --profile genai s3 cp \
  s3://<bucket>/llamole/output/<job-name>/output/output.tar.gz \
  /tmp/llamole-adapter/

tar -xzf /tmp/llamole-adapter/output.tar.gz -C /tmp/llamole-adapter/
# Adapter is at /tmp/llamole-adapter/saves/Qwen2-7B-Instruct-Adapter-smoketest/
```

Copy the adapter to the local `saves/` directory to use for eval:

```bash
cp -r /tmp/llamole-adapter/saves/Qwen2-7B-Instruct-Adapter-smoketest saves/
```

---

## Instance Selection Guide

| Task | Instance | GPUs | VRAM | Price/hr | Notes |
|---|---|---|---|---|---|
| Eval / Inference | `ml.g5.2xlarge` | 1x A10G | 24 GB | $1.21 | Default for eval |
| SFT training | `ml.g5.12xlarge` | 4x A10G | 96 GB | $5.67 | Default for training |
| Full-scale training | `ml.p4d.24xlarge` | 8x A100 | 320 GB | $32.77 | For full dataset runs |

If `ml.g5.2xlarge` shows `Pending` for >10 min (capacity issue), try `ml.g5.4xlarge` — different capacity pool, same GPU.

---

## How It Works Internally

### Source upload

`submit_job.py` uses `source_dir="."` which uploads the entire repo to S3 as a tarball. Files listed in `.sourceignore` (project root) are excluded:

```
saves/          # model weights — too large
.git/
data/molqa_*.json  # large datasets
requirements.txt   # managed by entry.py instead
```

### Dependency management

The SageMaker base container is `pytorch-training:2.1.0-gpu-py310-cu121`. The repo's `requirements.txt` is excluded from the upload to avoid conflicts (the base container has its own pinned versions).

`sagemaker/entry.py` manages all dependencies explicitly:
1. **Removes `transformer_engine`** — the container ships this compiled against PyTorch 2.1.0; it breaks when torch is upgraded to 2.4.0 (ABI mismatch in `.so` file)
2. **Installs pinned packages** — `transformers==4.44.0`, `trl==0.9.6`, `peft==0.12.0`, etc.

Key pinned versions that must not be changed:
- `transformers==4.44.0` — 4.45+ removed `AutoModelForVision2Seq` used in `src/model/loader.py`
- `trl==0.9.6` — 1.0+ removed `AutoModelForCausalLMWithValueHead` used in training
- `numpy==1.26.4` — `accelerate==0.33.0` requires `numpy<2.0.0`

### Configs

| Config | Path | Purpose |
|---|---|---|
| Eval (material) | `config/generate/qwen_material.yaml` | Default eval config |
| Eval (drug) | `config/generate/qwen_drug.yaml` | Drug dataset eval |
| SFT full | `config/train/qwen_lora.yaml` | Full 4-epoch training |
| SFT smoke test | `config/train/qwen_lora_smoketest.yaml` | 1-epoch test, small batch |

---

## Common Errors and Fixes

### `InstallRequirementsError` — numpy/huggingface_hub conflict
The base container auto-installs `requirements.txt` if found. Fixed by excluding it via `.sourceignore` at the project root AND ensuring `requirements.txt` itself has compatible version ranges.

### `ImportError: transformer_engine_extensions.so: undefined symbol`
The container's `transformer_engine` is compiled against PyTorch 2.1.0. Upgrading torch breaks its ABI. Fixed by uninstalling `transformer_engine` in `entry.py` before installing packages.

### Job stuck in `Pending` for >10 minutes
`ml.g5.xlarge` has limited capacity in `us-east-1`. Use `ml.g5.2xlarge` instead — same GPU, different capacity pool.

### `ValueError: Unknown dataset: molqa_material_examples`
The eval router in `src/eval/workflow.py` must include example dataset names. Already fixed — see the condition in `run_eval()`.
