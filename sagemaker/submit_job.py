"""Submit Llamole eval as a SageMaker Training Job.

Usage:
    python sagemaker/submit_job.py                        # eval with qwen_material
    python sagemaker/submit_job.py --config qwen_drug     # eval with qwen_drug config
    python sagemaker/submit_job.py --wait                 # block until job completes

The job will:
  1. Spin up ml.g5.xlarge (A10G 24GB VRAM)
  2. Install Llamole dependencies
  3. Auto-download models from HuggingFace
  4. Run eval on example dataset
  5. Upload results to S3 and terminate

Cost: ~$1.00/hr for ml.g5.xlarge. A typical eval run takes ~20-40 min → ~$0.50-$1.00 total.
"""
import argparse
import os
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import Session
from pathlib import Path


# ── Load config from sagemaker/.env (never committed) ─────────────────────────
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

def _require(key):
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(
            f"Missing required config: {key}\n"
            f"Copy sagemaker/.env.example → sagemaker/.env and fill in your values."
        )
    return val

# ── Config ────────────────────────────────────────────────────────────────────
REGION        = "us-east-1"
ROLE          = _require("SAGEMAKER_ROLE")
S3_BUCKET     = _require("SAGEMAKER_S3_BUCKET")
S3_PREFIX     = "llamole"
INSTANCE_TYPE = "ml.g5.2xlarge"      # A10G, 24GB VRAM — fits Qwen2-7B in bf16
VPC_SUBNETS   = _require("SAGEMAKER_VPC_SUBNETS").split(",")
VPC_SG        = _require("SAGEMAKER_VPC_SG").split(",")

# PyTorch 2.1 + CUDA 12.1 Deep Learning Container
DL_CONTAINER   = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="qwen_material",
                        help="Config name under config/generate/ (without .yaml)")
    parser.add_argument("--instance", default=INSTANCE_TYPE,
                        help="SageMaker instance type")
    parser.add_argument("--wait", action="store_true",
                        help="Block until job finishes")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("WARNING: No HuggingFace token provided. Model downloads may fail for gated models.")
        print("Set HF_TOKEN env var or pass --hf-token")

    boto_session = boto3.Session(profile_name="genai", region_name=REGION)
    sm_session   = Session(boto_session=boto_session, default_bucket=S3_BUCKET)

    estimator = PyTorch(
        entry_point="sagemaker/entry.py",
        source_dir=".",                  # uploads entire Llamole repo as source
        role=ROLE,
        instance_type=args.instance,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        image_uri=DL_CONTAINER,
        sagemaker_session=sm_session,
        base_job_name="llamole-eval",
        output_path=f"s3://{S3_BUCKET}/{S3_PREFIX}/output",
        code_location=f"s3://{S3_BUCKET}/{S3_PREFIX}/code",
        subnets=VPC_SUBNETS,
        security_group_ids=VPC_SG,
        environment={
            "HF_TOKEN": hf_token,
            "CONFIG_FILE": f"config/generate/{args.config}.yaml",
            "HF_HOME": "/tmp/hf_cache",
        },
        hyperparameters={},
        keep_alive_period_in_seconds=0,   # terminate immediately after job
        volume_size=100,                  # GB — enough for 7B model weights
        max_run=3600 * 2,                 # 2h max runtime
    )

    print(f"\nSubmitting Llamole eval job:")
    print(f"  Config  : config/generate/{args.config}.yaml")
    print(f"  Instance: {args.instance}")
    print(f"  Output  : s3://{S3_BUCKET}/{S3_PREFIX}/output")
    print(f"  Logs    : CloudWatch → /aws/sagemaker/TrainingJobs")

    estimator.fit(wait=args.wait, logs="All" if args.wait else None)

    print(f"\nJob submitted: {estimator.latest_training_job.name}")
    print(f"\nTrack it:")
    print(f"  AWS Console → SageMaker → Training Jobs → {estimator.latest_training_job.name}")
    print(f"  Or run: aws --profile genai sagemaker describe-training-job \\")
    print(f"    --region us-east-1 --training-job-name {estimator.latest_training_job.name}")


if __name__ == "__main__":
    main()
