# Molecule Modification & Optimization Feature — Design Spec

**Date:** 2026-04-03  
**Status:** Approved

---

## Problem Statement

Llamole currently supports inverse molecular design (properties → new molecule from scratch) and retrosynthetic planning. It cannot take an *existing* molecule and produce structurally-similar variants with improved properties. Medicinal chemists need this workflow: "here is my lead compound — give me 5 modifications that improve solubility, and let me synthesize whichever I like."

---

## Goals

- Accept a molecule by SMILES or by name (resolved via PubChem)
- Support two modification modes: **autonomous** (model chooses strategy) and **instruction-guided** (chemist specifies transformation type in natural language)
- Generate 5 diverse candidate modifications ranked by predicted property improvement
- Provide LLM-generated chemical rationale for each candidate
- Allow on-demand retrosynthesis for any selected candidate using the existing A* planner
- Support ADMET properties: logS, logP, PAMPA, metabolic stability (new), plus existing Llamole properties

---

## Non-Goals

- Real-time property measurements (wet lab integration)
- Protein-target docking (separate workflow)
- Enumeration of exhaustive combinatorial libraries

---

## Architecture Overview

```
Chemist input (name or SMILES + optional instruction)
         |
  MoleculeNameResolver       [new: PubChem API tool]
         |
  IntentParser               [new: LLM → operation_embedding]
         |
  GraphEditorDiT             [new: extends GraphDiT, partial noise injection]
  (generates 5 candidates)
         |
  ADMETPredictor             [new: GNN for logS, logP, PAMPA, metabolic stability]
         |
  LLM rationale generation   [existing LLM, new prompt template]
         |
  Ranked modification display
         |  (on demand)
  molstar.py A* planner      [existing, unchanged]
         |
  Synthesis route
```

---

## New Components

### 1. MoleculeNameResolver
- **Location:** `src/tools/molecule_resolver.py`
- **Interface:** `resolve(name: str) -> str` returns SMILES
- **Implementation:** PubChem PUG REST API (`/compound/name/{name}/property/CanonicalSMILES/JSON`)
- **Fallback:** ChEMBL REST API if PubChem returns no result
- **Error handling:** Returns `None` with a user-facing message if both fail

### 2. IntentParser
- **Location:** `src/model/intent_parser.py`
- **Interface:** `parse(instruction: str) -> Tensor` returns operation embedding
- **Implementation:** Fine-tune the existing base LLM (LoRA) on a dataset of (natural language instruction → chemical operation label) pairs derived from MMP transformation types (e.g., "make more soluble" → `add_hbd`, "reduce lipophilicity" → `remove_aromatic`)
- **Autonomous mode:** Returns a learned `null_operation` embedding (no instruction given)
- **Operation vocabulary:** Derived from MMP transformation categories in the existing MMP codebase

### 3. GraphEditorDiT
- **Location:** `src/model/graph_editor/editor_model.py`
- **Class:** `GraphEditorDiT` subclasses existing `GraphDiT` (`src/model/graph_decoder/diffusion_model.py`)
- **Key architectural changes:**
  - **Partial noise injection:** Instead of starting from pure noise, inject noise up to timestep `t_edit ∈ [0, T]`. Low `t_edit` = conservative edits; high `t_edit` = aggressive structural changes.
  - **Cross-attention conditioning** (three signals fed at every transformer layer):
    1. **Source graph embedding** — from existing `GraphEncoder` (GIN-based, `src/model/graph_encoder/`). No changes to GraphEncoder needed.
    2. **Property delta embedding** — target improvement as a delta (e.g., `Δ logS = +1.5`), not absolute value. Encoded via a new learned MLP projection head.
    3. **Operation embedding** — output of `IntentParser`. `null_operation` token for autonomous mode.
- **Diversity generation:** 5 candidates sampled by varying random seed + varying `t_edit` across `[t_min, t_max]` to produce a spectrum from subtle to bold edits in one batch.
- **Validity enforcement:** Reuses existing `molecule_utils.py` valency/SMILES validation after each denoising step.

### 4. ADMETPredictor
- **Location:** `src/model/admet_predictor/`
- **Architecture:** GNN (same family as existing `graph_predictor`) with multi-task heads, one per property
- **Properties:** logS (aqueous solubility), logP (lipophilicity), PAMPA (membrane permeability), metabolic stability (half-life)
- **Training data:** Public datasets — AqSolDB (logS), Lipophilicity dataset (logP), PAMPA-BBB (permeability), MetaboLights/ChEMBL (metabolic stability)
- **Output:** `{property: value, confidence: float}` per candidate
- **Integration:** Called after `GraphEditorDiT` generates candidates; scores used to rank and delta-compute against source molecule scores

---

## Training Pipeline

### Stage 1 — ADMETPredictor (independent)
1. Download and preprocess public ADMET datasets
2. Convert SMILES → molecular graphs using existing `molecule_utils.py`
3. Train multi-task GNN with per-property MSE loss
4. Validate on held-out sets; target Pearson r > 0.85 per property

### Stage 2 — IntentParser (independent)
1. Extract transformation type labels from the MMP database (user has existing MMP code)
2. Generate natural language descriptions of each transformation type (template-based + LLM augmentation)
3. Fine-tune base LLM with LoRA on (instruction → operation_label) classification
4. Embed operation labels as learned vectors; these become the operation embedding space

### Stage 3 — GraphEditorDiT
1. Load MMP pairs as `(mol_A_graph, mol_B_graph, property_delta, operation_label)`
2. For each pair: encode `mol_A` with `GraphEncoder`, inject partial noise into `mol_B` graph, train denoising to recover `mol_B` conditioned on `(source_embedding, property_delta_embedding, operation_embedding)`
3. Loss: reconstruction loss on atom types + bond types (same as GraphDiT) + auxiliary property prediction loss
4. Curriculum: start with high-noise (easy, large edits) and anneal toward low-noise (hard, subtle edits)

### Stage 4 — End-to-end fine-tuning
1. Freeze ADMETPredictor; jointly fine-tune IntentParser + GraphEditorDiT with RL signal from ADMETPredictor scores
2. Reward: `Δ property` for target property + SA penalty for poor synthetic accessibility

---

## Inference Workflow

```python
# Autonomous mode
modifications = model.edit(
    molecule="aspirin",          # or SMILES string
    target_property="logS",
    target_delta=+1.5,
    n_candidates=5,
)

# Instruction-guided mode
modifications = model.edit(
    molecule="CC(=O)Oc1ccccc1C(=O)O",
    instruction="add a hydrogen bond donor to improve solubility",
    target_property="logS",
    n_candidates=5,
)

# On-demand synthesis for selected candidate
route = model.retrosynthesize(modifications[2].smiles)
```

Each returned `Modification` object contains:
- `smiles` — SMILES of the modified molecule
- `property_scores` — dict of predicted ADMET values
- `property_deltas` — improvement over source molecule
- `rationale` — LLM-generated chemical explanation
- `t_edit` — how aggressively the structure was changed (interpretability signal)

---

## Integration with Existing Llamole

| Existing component | Change required |
|---|---|
| `GraphLLMForCausalMLM` | Add `edit()` method alongside existing `design()` and `retrosynthesize()` |
| `GraphDiT` | Subclassed by `GraphEditorDiT`; no changes to base class |
| `GraphEncoder` | Reused as-is for source molecule encoding |
| `molstar.py` | Unchanged; called on selected modification SMILES |
| `src/extras/constants.py` | Add new special tokens: `<edit_start>`, `<edit_end>`, `<edit_body>`, `<operation>` |
| `src/data/template.py` | Add new prompt template for the edit + rationale task |
| Gradio web UI (`launch.py`) | New tab: molecule input, property selector, instruction box, modification cards with "Synthesize" button |

---

## New Special Tokens

```
<edit_start>    — begins an editing task
<edit_end>      — ends an editing task
<edit_body>     — body of the editing context
<operation>     — inline operation embedding injection point
```

---

## File Structure (new files only)

```
src/
  tools/
    molecule_resolver.py        # PubChem/ChEMBL name → SMILES
  model/
    intent_parser.py            # NL → operation embedding
    graph_editor/
      editor_model.py           # GraphEditorDiT
      editor_utils.py           # partial noise injection, diversity sampling
    admet_predictor/
      predictor_model.py        # multi-task GNN
      admet_datasets.py         # dataset loaders for AqSolDB, logP, PAMPA, etc.
  data/
    mmp_loader.py               # MMP pair dataset loader (uses user's existing MMP code)
config/
  train/
    editor_lora.yaml            # training config for GraphEditorDiT
  generate/
    editor_material.yaml        # inference config
```

---

## Open Questions

1. **MMP data format:** What format does the existing MMP codebase output? (SMILES pairs + property CSV, or graph objects?) This determines how much glue code `mmp_loader.py` needs.
2. **Property delta range:** Should the model clip `Δ property` targets to a realistic range (based on MMP distribution) to avoid out-of-distribution conditioning?
3. **Rollback for editor:** Should `GraphEditorDiT` inherit the existing rollback mechanism from `GraphDiT` for invalid outputs?
