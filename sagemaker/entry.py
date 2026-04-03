"""SageMaker entry point for Llamole eval or training.

SageMaker extracts the source tarball to /opt/ml/code/ and calls this script.
Model outputs are written to /opt/ml/output/data/ which SageMaker uploads to S3.

Environment variables:
  MODE        : "eval" (default) or "train"
  CONFIG_FILE : path to YAML config under config/generate/ or config/train/
  HF_TOKEN    : HuggingFace token for model downloads
"""
import os
import sys
import subprocess

# ── Install extra dependencies not in the base DL container ──────────────────
EXTRA_PACKAGES = [
    # Core ML
    "numpy==1.26.4",
    "accelerate==0.33.0",
    "datasets==2.21.0",
    "safetensors==0.4.5",
    "transformers==4.44.0",
    "trl==0.9.6",
    "peft==0.12.0",
    # Graph / Chemistry
    "rdkit==2023.9.6",
    "torch_geometric",
    "rdchiral==1.1.0",
    "fcd_torch==1.0.7",
    "git+https://github.com/igor-krawczuk/mini-moses",
    # Utilities
    "omegaconf==2.3.0",
    "pandarallel",
    "einops",
    "sentencepiece",
    "tiktoken",
    "nltk",
    "pandas==1.5.3",
]

# Remove transformer_engine — compiled against the container's PyTorch 2.1.0,
# breaks when we upgrade to torch 2.4.0 (ABI mismatch in .so)
print("Removing transformer_engine to avoid ABI conflicts...")
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y",
                 "transformer_engine", "transformer-engine",
                 "transformer_engine_extensions"])

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + EXTRA_PACKAGES)
print("Dependencies installed.")

# ── Configure HuggingFace ─────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    subprocess.check_call([
        sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
        "login", "--token", hf_token
    ])

# Cache models on the instance's local NVMe (faster than EBS)
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

# ── Run eval or train ─────────────────────────────────────────────────────────
mode = os.environ.get("MODE", "eval")
import glob, shutil

if mode == "train":
    config_file = os.environ.get("CONFIG_FILE", "config/train/qwen_lora_smoketest.yaml")
    # Detect number of GPUs available — torchrun requires nproc_per_node
    import torch
    n_gpus = torch.cuda.device_count()
    n_gpus = max(n_gpus, 1)
    print(f"Running training with config: {config_file} on {n_gpus} GPU(s)")
    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={n_gpus}",
            "--master_port=29500",
            "main.py", "train", config_file,
        ],
        capture_output=False,
    )
    # Copy adapter checkpoints to SageMaker output path
    output_dir = "/opt/ml/output/data"
    os.makedirs(output_dir, exist_ok=True)
    saves_dir = "saves"
    if os.path.exists(saves_dir):
        shutil.copytree(saves_dir, os.path.join(output_dir, "saves"))
        print(f"Saved adapter checkpoints → {output_dir}/saves")
else:
    config_file = os.environ.get("CONFIG_FILE", "config/generate/qwen_material.yaml")
    print(f"Running eval with config: {config_file}")
    result = subprocess.run(
        [sys.executable, "main.py", "eval", config_file],
        capture_output=False,
    )
    # Copy result files to SageMaker output path
    output_dir = "/opt/ml/output/data"
    os.makedirs(output_dir, exist_ok=True)
    for pattern in ["*.json", "*.csv", "*.txt"]:
        for f in glob.glob(pattern):
            shutil.copy(f, output_dir)
            print(f"Saved {f} → {output_dir}")

sys.exit(result.returncode)
