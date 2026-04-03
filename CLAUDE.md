# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Llamole is a multimodal LLM for **inverse molecular design with retrosynthetic planning**. It extends base LLMs (Llama-3.1-8B, Qwen2-7B, Mistral-7B) with three graph modules: a GIN-based encoder, a Graph Diffusion Transformer decoder, and a GNN predictor — all coordinated via an A* retrosynthetic planner.

## Environment Setup

```bash
conda create --name llamole python=3.11 -y
conda activate llamole
./install_environment.sh
```

## Common Commands

```bash
# Training (supervised fine-tuning with LoRA)
python main.py train config/train/llama_lora.yaml

# Evaluation
python main.py eval config/generate/llama_material.yaml

# Export/merge LoRA adapters into base model
python main.py export

# Download datasets
python main.py download_data

# Launch Gradio web UI (available at localhost:7860)
python launch.py
```

## Architecture

**Entry points:**
- `main.py` — CLI dispatch to train/eval/export/download_data
- `launch.py` — Gradio web UI
- `src/model/modeling_llamole.py` — `GraphLLMForCausalMLM`, the core model class integrating LLM + graph modules

**Data flow:**
1. User instruction + property constraints → tokenizer adds special task tokens (`<design_start>`, `<molecule>`, `<retro_start>`, etc.)
2. Base LLM (with LoRA) processes text and routes to graph modules based on task type
3. **Graph Encoder** (`src/model/graph_encoder/`) — GIN-based, encodes molecule SMILES → embeddings
4. **Graph Decoder** (`src/model/graph_decoder/`) — Graph Diffusion Transformer (`GraphDiT`), generates new molecular graphs conditioned on properties
5. **Graph Predictor** (`src/model/graph_predictor/`) — GNN that predicts reactants given a target molecule (one-step retrosynthesis)
6. **A\* Planner** (`src/model/planner/molstar.py`) — builds a `MolTree` synthesis tree via `molstar()`, using the LLM for cost estimation

**Key files:**
- `src/model/modeling_llamole.py` — `GraphLLMForCausalMLM` (model integration)
- `src/model/graph_decoder/molecule_utils.py` — SMILES ↔ graph conversion, valency validation
- `src/model/planner/molstar.py` — A* search over synthesis tree
- `src/model/planner/mol_tree.py` / `mol_node.py` / `reaction_node.py` — synthesis tree structures
- `src/data/collator.py` — batch collation for mixed sequence+graph inputs
- `src/hparams/` — all hyperparameter/config argument classes
- `src/extras/constants.py` — special token IDs, bond types

**Configuration:**
- Training configs: `config/train/` (YAML, specify LoRA rank, loss weights, special tokens)
- Generation/inference configs: `config/generate/` (YAML, temperature, paths to saved graph models)
- Multi-task loss weights in training: `loss_weight_lm`, `loss_weight_design`, `loss_weight_retro`

**Special tokens** (defined in YAML configs and `src/extras/constants.py`):
- `<design_start>/<design_end>/<design_body>` — molecular design task
- `<molecule>` — inline molecule placeholder
- `<retro_start>/<retro_end>/<retro_body>` — retrosynthetic planning task
- `<rollback_start>/<rollback_end>` — planner rollback mechanism

**Molecular properties supported:**
- Drug: HIV, BBBP, BACE
- Materials: CO2, N2, O2, FFV, TC (permeability/selectivity)
- Synthesis metrics: SC (synthetic complexity), SA (synthetic accessibility)

## Key Dependencies

- `torch` + `torch_geometric` — base deep learning and GNNs
- `transformers` + `peft` — LLM backbone and LoRA fine-tuning
- `rdkit` — SMILES parsing, molecular validity checks
- `rdchiral` — reaction template application
- `trl` — training recipes (SFT)
- `gradio` — web UI
- `bitsandbytes` — quantization support
