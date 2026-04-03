# Molecule Modification & Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `GraphEditorDiT` module to Llamole that takes an existing molecule + property targets + optional natural language instruction and returns 5 ranked modifications with ADMET scores, LLM rationales, and on-demand synthesis routes.

**Architecture:** A `GraphEditorDiT` subclasses the existing `GraphDiT` diffusion model, adding partial noise injection (start from t_edit, not full noise) and three conditioning signals: source molecule graph embedding (via existing `GraphCLIP`), property delta targets, and an operation embedding from a new `IntentParser`. A standalone `ADMETPredictor` GNN scores candidates on logS, logP, PAMPA, and metabolic stability. The full workflow is exposed via `GraphLLMForCausalMLM.edit()` and a new Gradio tab.

**Tech Stack:** PyTorch, torch_geometric, transformers + peft (LoRA), rdkit, requests (PubChem API), gradio

---

## File Map

**New files:**
- `src/tools/molecule_resolver.py` — PubChem/ChEMBL name → SMILES
- `src/model/admet_predictor/__init__.py` — package init
- `src/model/admet_predictor/predictor_model.py` — multi-task GNN (logS, logP, PAMPA, metabolic stability)
- `src/model/admet_predictor/admet_datasets.py` — dataset loaders (AqSolDB, MoleculeNet lipophilicity, PAMPA-BBB)
- `src/model/admet_predictor/train_admet.py` — standalone training script
- `src/data/mmp_loader.py` — adapter for user's MMP database → PyTorch Dataset of (mol_A, mol_B, property_delta, operation_label) pairs
- `src/model/intent_parser.py` — maps NL instruction text → operation embedding vector
- `src/model/graph_editor/__init__.py` — package init
- `src/model/graph_editor/editor_utils.py` — `apply_partial_noise()`, `sample_diverse_edits()`
- `src/model/graph_editor/editor_model.py` — `EditorTransformer` + `GraphEditorDiT`
- `src/model/graph_editor/train_editor.py` — MMP-supervised training script
- `config/train/editor_lora.yaml` — training config for GraphEditorDiT Stage 3
- `config/generate/editor_config.yaml` — inference config
- `tests/tools/test_molecule_resolver.py`
- `tests/model/admet_predictor/test_admet_predictor.py`
- `tests/data/test_mmp_loader.py`
- `tests/model/test_intent_parser.py`
- `tests/model/graph_editor/test_editor_utils.py`
- `tests/model/graph_editor/test_editor_model.py`
- `tests/integration/test_edit_workflow.py`

**Modified files:**
- `src/extras/constants.py` — add `EDIT_SPECIAL_TOKENS` dict
- `src/data/template.py` — add edit prompt template
- `src/model/modeling_llamole.py` — add `edit()` method
- `launch.py` — add "Modify Molecule" Gradio tab

---

## Phase 1: Infrastructure

### Task 1: MoleculeNameResolver

**Files:**
- Create: `src/tools/__init__.py`
- Create: `src/tools/molecule_resolver.py`
- Create: `tests/tools/__init__.py`
- Create: `tests/tools/test_molecule_resolver.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/test_molecule_resolver.py
import pytest
from src.tools.molecule_resolver import MoleculeNameResolver

def test_resolve_aspirin_returns_valid_smiles():
    resolver = MoleculeNameResolver()
    smiles = resolver.resolve("aspirin")
    assert smiles is not None
    from rdkit import Chem
    assert Chem.MolFromSmiles(smiles) is not None

def test_resolve_ibuprofen_returns_valid_smiles():
    resolver = MoleculeNameResolver()
    smiles = resolver.resolve("ibuprofen")
    assert smiles is not None
    from rdkit import Chem
    assert Chem.MolFromSmiles(smiles) is not None

def test_resolve_unknown_returns_none():
    resolver = MoleculeNameResolver()
    smiles = resolver.resolve("xyznotamolecule99999abc")
    assert smiles is None

def test_resolve_smiles_passthrough():
    resolver = MoleculeNameResolver()
    input_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    result = resolver.resolve(input_smiles)
    assert result == input_smiles
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/ldodda/Documents/Codes/Llamole
python -m pytest tests/tools/test_molecule_resolver.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.tools.molecule_resolver'`

- [ ] **Step 3: Create package init files**

```python
# src/tools/__init__.py
# (empty)

# tests/__init__.py
# (empty — create if missing)

# tests/tools/__init__.py
# (empty)
```

- [ ] **Step 4: Implement MoleculeNameResolver**

```python
# src/tools/molecule_resolver.py
import re
from typing import Optional
import requests
from rdkit import Chem


class MoleculeNameResolver:
    """Resolves molecule names to SMILES via PubChem with ChEMBL fallback."""

    PUBCHEM_URL = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
        "/{name}/property/CanonicalSMILES/JSON"
    )
    CHEMBL_URL = (
        "https://www.ebi.ac.uk/chembl/api/data/molecule"
        "?pref_name__iexact={name}&format=json"
    )

    def resolve(self, name_or_smiles: str) -> Optional[str]:
        """Return SMILES for a molecule name, or pass through if already SMILES."""
        if self._is_smiles(name_or_smiles):
            return name_or_smiles
        smiles = self._try_pubchem(name_or_smiles)
        if smiles is None:
            smiles = self._try_chembl(name_or_smiles)
        return smiles

    def _is_smiles(self, text: str) -> bool:
        mol = Chem.MolFromSmiles(text)
        return mol is not None

    def _try_pubchem(self, name: str) -> Optional[str]:
        try:
            resp = requests.get(
                self.PUBCHEM_URL.format(name=requests.utils.quote(name)),
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        except Exception:
            pass
        return None

    def _try_chembl(self, name: str) -> Optional[str]:
        try:
            resp = requests.get(
                self.CHEMBL_URL.format(name=requests.utils.quote(name)),
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                mols = data.get("molecules", [])
                if mols:
                    structs = mols[0].get("molecule_structures") or {}
                    return structs.get("canonical_smiles")
        except Exception:
            pass
        return None
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/tools/test_molecule_resolver.py -v
```
Expected: all 4 tests PASS (note: `test_resolve_unknown_returns_none` makes real HTTP calls — skip with `-m "not slow"` if rate-limited)

- [ ] **Step 6: Commit**

```bash
git add src/tools/ tests/tools/ tests/__init__.py
git commit -m "feat: add MoleculeNameResolver with PubChem/ChEMBL fallback"
```

---

### Task 2: ADMET Dataset Loaders

**Files:**
- Create: `src/model/admet_predictor/__init__.py`
- Create: `src/model/admet_predictor/admet_datasets.py`
- Create: `tests/model/__init__.py`
- Create: `tests/model/admet_predictor/__init__.py`
- Create: `tests/model/admet_predictor/test_admet_datasets.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model/admet_predictor/test_admet_datasets.py
import pytest
from torch_geometric.data import Data
from src.model.admet_predictor.admet_datasets import AqSolDataset, LipophilicityDataset

def test_aqsol_dataset_len():
    ds = AqSolDataset(split="train")
    assert len(ds) > 1000

def test_aqsol_dataset_item_shape():
    ds = AqSolDataset(split="train")
    item = ds[0]
    assert isinstance(item, Data)
    assert item.x is not None
    assert item.edge_index is not None
    assert hasattr(item, "y")
    assert item.y.shape == (1,)

def test_lipophilicity_dataset_item_shape():
    ds = LipophilicityDataset(split="train")
    item = ds[0]
    assert isinstance(item, Data)
    assert item.y.shape == (1,)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/model/admet_predictor/test_admet_datasets.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Download datasets**

AqSolDB (logS): `https://raw.githubusercontent.com/PatWalters/solubility/master/aqsoldb.csv`  
MoleculeNet Lipophilicity: installed with `pip install moleculenet` or download from `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv`

```bash
mkdir -p data/admet
wget -O data/admet/aqsoldb.csv "https://raw.githubusercontent.com/PatWalters/solubility/master/aqsoldb.csv"
wget -O data/admet/lipophilicity.csv "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
```

- [ ] **Step 4: Implement dataset loaders**

```python
# src/model/admet_predictor/admet_datasets.py
from typing import List, Literal, Optional
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import AllChem


ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "num_hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
    ],
}


def one_hot(val, lst):
    return [1 if val == v else 0 for v in lst] + [0 if val in lst else 1]


def atom_features(atom) -> List[float]:
    return (
        one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
        + one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])
        + one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
        + one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
        + [float(atom.GetIsAromatic())]
    )


BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


def smiles_to_graph(smiles: str, y: float) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges_src, edges_dst, edge_attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        btype = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
        edges_src += [i, j]
        edges_dst += [j, i]
        edge_attrs += [btype, btype]
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([y], dtype=torch.float))


def _split_indices(n: int, split: str) -> List[int]:
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    if split == "train":
        return list(range(0, train_end))
    elif split == "val":
        return list(range(train_end, val_end))
    else:
        return list(range(val_end, n))


class AqSolDataset(Dataset):
    """Aqueous solubility (logS) from AqSolDB."""

    def __init__(self, split: Literal["train", "val", "test"] = "train",
                 csv_path: str = "data/admet/aqsoldb.csv"):
        super().__init__()
        df = pd.read_csv(csv_path)
        indices = _split_indices(len(df), split)
        rows = df.iloc[indices]
        self._data = []
        for _, row in rows.iterrows():
            g = smiles_to_graph(row["SMILES"], float(row["Solubility"]))
            if g is not None:
                self._data.append(g)

    def len(self):
        return len(self._data)

    def get(self, idx):
        return self._data[idx]


class LipophilicityDataset(Dataset):
    """LogP (lipophilicity) from MoleculeNet."""

    def __init__(self, split: Literal["train", "val", "test"] = "train",
                 csv_path: str = "data/admet/lipophilicity.csv"):
        super().__init__()
        df = pd.read_csv(csv_path)
        indices = _split_indices(len(df), split)
        rows = df.iloc[indices]
        self._data = []
        for _, row in rows.iterrows():
            g = smiles_to_graph(row["smiles"], float(row["exp"]))
            if g is not None:
                self._data.append(g)

    def len(self):
        return len(self._data)

    def get(self, idx):
        return self._data[idx]
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/model/admet_predictor/test_admet_datasets.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/model/admet_predictor/ tests/model/ data/admet/
git commit -m "feat: add ADMET dataset loaders (AqSolDB, Lipophilicity)"
```

---

### Task 3: MMP Data Loader

**Files:**
- Create: `src/data/mmp_loader.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_mmp_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_mmp_loader.py
import pytest
import torch
from src.data.mmp_loader import MMPDataset, MMPPair

def test_mmp_pair_fields():
    pair = MMPPair(
        smiles_a="CC(=O)Oc1ccccc1C(=O)O",
        smiles_b="OC(=O)c1ccccc1O",
        property_name="logS",
        delta=1.2,
        operation_label="add_hbd",
    )
    assert pair.smiles_a is not None
    assert abs(pair.delta - 1.2) < 1e-6

def test_mmp_dataset_from_csv(tmp_path):
    csv = tmp_path / "pairs.csv"
    csv.write_text(
        "smiles_a,smiles_b,property_name,delta,operation_label\n"
        "CC(=O)Oc1ccccc1C(=O)O,OC(=O)c1ccccc1O,logS,1.2,add_hbd\n"
        "c1ccccc1,Oc1ccccc1,logS,0.8,add_hbd\n"
    )
    ds = MMPDataset(csv_path=str(csv))
    assert len(ds) == 2
    item = ds[0]
    assert "graph_a" in item
    assert "graph_b" in item
    assert "delta" in item
    assert "operation_label" in item
    assert isinstance(item["delta"], float)

def test_mmp_dataset_invalid_smiles_skipped(tmp_path):
    csv = tmp_path / "pairs.csv"
    csv.write_text(
        "smiles_a,smiles_b,property_name,delta,operation_label\n"
        "INVALID_SMILES,OC(=O)c1ccccc1O,logS,1.2,add_hbd\n"
        "c1ccccc1,Oc1ccccc1,logS,0.8,add_hbd\n"
    )
    ds = MMPDataset(csv_path=str(csv))
    assert len(ds) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/data/test_mmp_loader.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement MMP loader**

```python
# src/data/mmp_loader.py
"""Adapter between the user's MMP database and PyTorch Dataset.

Expected CSV format (adapt `csv_path` to your MMP export):
    smiles_a,smiles_b,property_name,delta,operation_label

Where:
  - smiles_a: source molecule SMILES
  - smiles_b: modified molecule SMILES (with better property)
  - property_name: e.g. "logS", "logP"
  - delta: property_b - property_a (positive = improvement)
  - operation_label: transformation type string (e.g. "add_hbd", "remove_aromatic")
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.model.admet_predictor.admet_datasets import smiles_to_graph


OPERATION_VOCAB = [
    "add_hbd",       # add hydrogen bond donor
    "add_hba",       # add hydrogen bond acceptor
    "remove_aromatic",
    "add_aliphatic",
    "reduce_mw",
    "increase_mw",
    "add_polar_group",
    "remove_nonpolar_group",
    "bioisostere_ester",
    "bioisostere_amide",
    "other",
]
OPERATION_TO_IDX = {op: i for i, op in enumerate(OPERATION_VOCAB)}
NULL_OPERATION_IDX = len(OPERATION_VOCAB)  # used when no instruction given


@dataclass
class MMPPair:
    smiles_a: str
    smiles_b: str
    property_name: str
    delta: float
    operation_label: str


class MMPDataset(Dataset):
    """Loads MMP pairs from a CSV file.

    To use your existing MMP code instead of a CSV, subclass this and
    override `_load_pairs()` to return a list of MMPPair.
    """

    def __init__(self, csv_path: str):
        pairs = self._load_pairs(csv_path)
        self._items = []
        for pair in pairs:
            g_a = smiles_to_graph(pair.smiles_a, pair.delta)
            g_b = smiles_to_graph(pair.smiles_b, pair.delta)
            if g_a is None or g_b is None:
                continue
            op_idx = OPERATION_TO_IDX.get(pair.operation_label, OPERATION_TO_IDX["other"])
            self._items.append({
                "graph_a": g_a,
                "graph_b": g_b,
                "delta": pair.delta,
                "property_name": pair.property_name,
                "operation_label": pair.operation_label,
                "operation_idx": op_idx,
            })

    def _load_pairs(self, csv_path: str) -> List[MMPPair]:
        df = pd.read_csv(csv_path)
        pairs = []
        for _, row in df.iterrows():
            pairs.append(MMPPair(
                smiles_a=str(row["smiles_a"]),
                smiles_b=str(row["smiles_b"]),
                property_name=str(row["property_name"]),
                delta=float(row["delta"]),
                operation_label=str(row["operation_label"]),
            ))
        return pairs

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/data/test_mmp_loader.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/mmp_loader.py tests/data/
git commit -m "feat: add MMP dataset adapter with operation vocabulary"
```

---

## Phase 2: ADMETPredictor

### Task 4: ADMETPredictor Model

**Files:**
- Create: `src/model/admet_predictor/predictor_model.py`
- Modify: `tests/model/admet_predictor/test_admet_predictor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model/admet_predictor/test_admet_predictor.py
import pytest
import torch
from torch_geometric.data import Data, Batch
from src.model.admet_predictor.predictor_model import ADMETPredictor

PROPERTIES = ["logS", "logP", "PAMPA", "metabolic_stability"]

def _make_batch():
    graphs = []
    for _ in range(4):
        n = 10
        x = torch.randn(n, 128)
        edge_index = torch.randint(0, n, (2, 20))
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)

def test_admet_predictor_output_keys():
    model = ADMETPredictor(in_channels=128, hidden_channels=64, num_layers=3)
    batch = _make_batch()
    out = model(batch.x, batch.edge_index, batch.batch)
    assert set(out.keys()) == set(PROPERTIES)

def test_admet_predictor_output_shape():
    model = ADMETPredictor(in_channels=128, hidden_channels=64, num_layers=3)
    batch = _make_batch()
    out = model(batch.x, batch.edge_index, batch.batch)
    for prop in PROPERTIES:
        assert out[prop].shape == (4,), f"Wrong shape for {prop}"

def test_admet_predictor_predict_returns_dict():
    model = ADMETPredictor(in_channels=128, hidden_channels=64, num_layers=3)
    batch = _make_batch()
    result = model.predict(batch)
    assert set(result.keys()) == set(PROPERTIES)
    for prop in PROPERTIES:
        assert isinstance(result[prop], list)
        assert len(result[prop]) == 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/model/admet_predictor/test_admet_predictor.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement ADMETPredictor**

```python
# src/model/admet_predictor/predictor_model.py
import os
from typing import Dict, List
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Batch

PROPERTIES = ["logS", "logP", "PAMPA", "metabolic_stability"]


class ADMETPredictor(nn.Module):
    """Multi-task GNN predicting ADMET properties from molecular graphs."""

    PROPERTIES = PROPERTIES

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 256,
        num_layers: int = 5,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            mlp = nn.Sequential(
                nn.Linear(in_c, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.heads = nn.ModuleDict({
            prop: nn.Sequential(
                nn.Linear(hidden_channels, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            for prop in self.PROPERTIES
        })

    def forward(self, x, edge_index, batch) -> Dict[str, torch.Tensor]:
        for conv, bn in zip(self.convs, self.bns):
            x = bn(conv(x, edge_index).relu())
        x = global_mean_pool(x, batch)
        return {prop: self.heads[prop](x).squeeze(-1) for prop in self.PROPERTIES}

    def predict(self, graph_batch: Batch) -> Dict[str, List[float]]:
        self.eval()
        with torch.no_grad():
            preds = self(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        return {prop: preds[prop].tolist() for prop in self.PROPERTIES}

    def save_pretrained(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "admet_model.pt"))

    @classmethod
    def load_pretrained(cls, output_dir: str, **kwargs) -> "ADMETPredictor":
        model = cls(**kwargs)
        path = os.path.join(output_dir, "admet_model.pt")
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/model/admet_predictor/test_admet_predictor.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/admet_predictor/predictor_model.py tests/model/admet_predictor/test_admet_predictor.py
git commit -m "feat: add ADMETPredictor multi-task GNN"
```

---

### Task 5: ADMETPredictor Training Script

**Files:**
- Create: `src/model/admet_predictor/train_admet.py`

- [ ] **Step 1: Write training script**

```python
# src/model/admet_predictor/train_admet.py
"""Train ADMETPredictor on public ADMET datasets.

Usage:
    python -m src.model.admet_predictor.train_admet \
        --output_dir saves/admet_predictor \
        --epochs 50 \
        --lr 1e-3
"""
import argparse
import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from src.model.admet_predictor.predictor_model import ADMETPredictor
from src.model.admet_predictor.admet_datasets import AqSolDataset, LipophilicityDataset


def train_property(property_name: str, dataset_cls, output_dir: str,
                   epochs: int = 50, lr: float = 1e-3, batch_size: int = 64):
    """Train a single-property predictor and save its head weights."""
    print(f"\n=== Training {property_name} predictor ===")
    train_ds = dataset_cls(split="train")
    val_ds = dataset_cls(split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # in_channels from dataset item
    sample = train_ds[0]
    in_channels = sample.x.shape[1]

    model = ADMETPredictor(in_channels=in_channels, hidden_channels=256, num_layers=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(preds[property_name], batch.y.squeeze(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch.x, batch.edge_index, batch.batch)
                val_losses.append(criterion(preds[property_name], batch.y.squeeze(-1)).item())
        val_loss = sum(val_losses) / len(val_losses)
        print(f"  Epoch {epoch+1}/{epochs}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(os.path.join(output_dir, property_name))

    print(f"  Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="saves/admet_predictor")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_property("logS", AqSolDataset, args.output_dir, args.epochs, args.lr)
    train_property("logP", LipophilicityDataset, args.output_dir, args.epochs, args.lr)
    # PAMPA and metabolic_stability: add dataset classes in admet_datasets.py when data is available
    print(f"\nAll models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a quick smoke test (1 epoch)**

```bash
python -m src.model.admet_predictor.train_admet --epochs 1 --output_dir /tmp/admet_test
```
Expected: prints epoch loss, saves `admet_model.pt` to `/tmp/admet_test/logS/` and `/tmp/admet_test/logP/`

- [ ] **Step 3: Commit**

```bash
git add src/model/admet_predictor/train_admet.py
git commit -m "feat: add ADMETPredictor training script"
```

---

## Phase 3: IntentParser

### Task 6: IntentParser Model

**Files:**
- Create: `src/model/intent_parser.py`
- Create: `tests/model/test_intent_parser.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model/test_intent_parser.py
import pytest
import torch
from src.model.intent_parser import IntentParser, OPERATION_VOCAB, NULL_OPERATION_IDX

def test_operation_vocab_coverage():
    assert len(OPERATION_VOCAB) >= 10
    assert "add_hbd" in OPERATION_VOCAB
    assert "other" in OPERATION_VOCAB

def test_null_operation_idx():
    assert NULL_OPERATION_IDX == len(OPERATION_VOCAB)

def test_get_operation_embedding_shape():
    parser = IntentParser(embedding_dim=128)
    emb = parser.get_operation_embedding("add hydrogen bond donor")
    assert emb.shape == (128,)

def test_get_null_embedding_shape():
    parser = IntentParser(embedding_dim=128)
    emb = parser.get_null_embedding()
    assert emb.shape == (128,)

def test_batch_embed_shape():
    parser = IntentParser(embedding_dim=128)
    instructions = ["make it more soluble", "reduce lipophilicity", ""]
    embs = parser.batch_embed(instructions)
    assert embs.shape == (3, 128)

def test_empty_instruction_returns_null_embedding():
    parser = IntentParser(embedding_dim=128)
    null_emb = parser.get_null_embedding()
    empty_emb = parser.get_operation_embedding("")
    assert torch.allclose(null_emb, empty_emb)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/model/test_intent_parser.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement IntentParser**

```python
# src/model/intent_parser.py
"""Maps natural language instructions to operation embedding vectors.

Phase 1 (rule-based): keyword matching → operation index → learned embedding.
Phase 2 (fine-tuned): replace `_classify_instruction` with LLM-based classifier.
"""
from typing import List, Optional
import torch
import torch.nn as nn
from src.data.mmp_loader import OPERATION_VOCAB, OPERATION_TO_IDX, NULL_OPERATION_IDX

# Keywords that map to operation labels (extend as needed)
_KEYWORD_MAP = {
    "solub": "add_hbd",
    "water": "add_hbd",
    "hbd": "add_hbd",
    "donor": "add_hbd",
    "acceptor": "add_hba",
    "hba": "add_hba",
    "polar": "add_polar_group",
    "lipophil": "remove_nonpolar_group",
    "logp": "remove_nonpolar_group",
    "aromat": "remove_aromatic",
    "aliphat": "add_aliphatic",
    "smaller": "reduce_mw",
    "weight": "reduce_mw",
    "larger": "increase_mw",
    "ester": "bioisostere_ester",
    "amide": "bioisostere_amide",
}


class IntentParser(nn.Module):
    """Converts a natural language instruction to an operation embedding vector.

    In Phase 1, classification is rule-based (keyword matching).
    In Phase 2, replace `_classify_instruction` with a fine-tuned LLM classifier.
    """

    def __init__(self, embedding_dim: int = 256, vocab_size: int = None):
        super().__init__()
        vs = vocab_size if vocab_size is not None else NULL_OPERATION_IDX + 1
        # +1 for the null operation
        self.embeddings = nn.Embedding(vs, embedding_dim)
        self.embedding_dim = embedding_dim

    def _classify_instruction(self, instruction: str) -> int:
        """Returns operation index. Override with LLM classifier in Phase 2."""
        if not instruction.strip():
            return NULL_OPERATION_IDX
        lower = instruction.lower()
        for keyword, op in _KEYWORD_MAP.items():
            if keyword in lower:
                return OPERATION_TO_IDX.get(op, OPERATION_TO_IDX["other"])
        return OPERATION_TO_IDX["other"]

    def get_operation_embedding(self, instruction: str) -> torch.Tensor:
        """Returns embedding vector for a single instruction."""
        idx = self._classify_instruction(instruction)
        return self.embeddings(torch.tensor(idx)).detach()

    def get_null_embedding(self) -> torch.Tensor:
        """Returns the null operation embedding (used in autonomous mode)."""
        return self.embeddings(torch.tensor(NULL_OPERATION_IDX)).detach()

    def batch_embed(self, instructions: List[str]) -> torch.Tensor:
        """Returns (N, embedding_dim) tensor for a list of instructions."""
        indices = [self._classify_instruction(inst) for inst in instructions]
        return self.embeddings(torch.tensor(indices)).detach()

    def save_pretrained(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/intent_parser.pt")

    @classmethod
    def load_pretrained(cls, path: str, **kwargs) -> "IntentParser":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(f"{path}/intent_parser.pt", map_location="cpu"))
        return model
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/model/test_intent_parser.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/intent_parser.py tests/model/test_intent_parser.py
git commit -m "feat: add IntentParser with rule-based keyword classification"
```

---

## Phase 4: GraphEditorDiT

### Task 7: Editor Utilities (Partial Noise + Diversity Sampling)

**Files:**
- Create: `src/model/graph_editor/__init__.py`
- Create: `src/model/graph_editor/editor_utils.py`
- Create: `tests/model/graph_editor/__init__.py`
- Create: `tests/model/graph_editor/test_editor_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model/graph_editor/test_editor_utils.py
import pytest
import torch
from src.model.graph_editor.editor_utils import apply_partial_noise, compute_t_edit_schedule

def _make_dense_graph(batch_size=2, n_nodes=10, xdim=16, edim=5):
    X = torch.zeros(batch_size, n_nodes, xdim)
    X[:, :, 0] = 1  # carbon atom
    E = torch.zeros(batch_size, n_nodes, n_nodes, edim)
    node_mask = torch.ones(batch_size, n_nodes, dtype=torch.bool)
    y = torch.randn(batch_size, 10)
    return X, E, y, node_mask

def test_partial_noise_shape():
    X, E, y, node_mask = _make_dense_graph()
    noisy = apply_partial_noise(X, E, y, node_mask, t_edit_frac=0.5, T=500)
    assert noisy["X_t"].shape == X.shape
    assert noisy["E_t"].shape == E.shape
    assert abs(noisy["t"].mean().item() - 0.5) < 0.01

def test_partial_noise_zero_preserves_structure():
    X, E, y, node_mask = _make_dense_graph()
    noisy = apply_partial_noise(X, E, y, node_mask, t_edit_frac=0.0, T=500)
    # At t=0, no noise should be added — X_t should match X
    assert torch.allclose(noisy["X_t"].float(), X.float())

def test_t_edit_schedule_returns_n_values():
    schedule = compute_t_edit_schedule(n=5, t_min=0.1, t_max=0.8)
    assert len(schedule) == 5
    assert schedule[0] <= schedule[-1]
    assert all(0.1 <= v <= 0.8 for v in schedule)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/model/graph_editor/test_editor_utils.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement editor utilities**

```python
# src/model/graph_editor/editor_utils.py
"""Partial noise injection and diversity sampling utilities for GraphEditorDiT."""
import math
from typing import Dict, List
import torch
import torch.nn.functional as F


def apply_partial_noise(
    X: torch.Tensor,   # (bs, n, xdim) one-hot atom types
    E: torch.Tensor,   # (bs, n, n, edim) one-hot bond types
    y: torch.Tensor,   # (bs, ydim) property conditioning
    node_mask: torch.Tensor,  # (bs, n) bool
    t_edit_frac: float,  # fraction in [0, 1] — 0 = no noise, 1 = full noise
    T: int = 500,
) -> Dict:
    """Apply noise to (X, E) at a fixed fractional timestep t_edit_frac.

    Returns a noisy_data dict compatible with GraphDiT._forward().
    At t=0: returns original graph unchanged.
    At t=1: returns fully-noised graph sampled from marginals.
    """
    bs, n, xdim = X.shape
    edim = E.shape[-1]

    t_int = torch.full((bs, 1), fill_value=round(t_edit_frac * T),
                       dtype=torch.float, device=X.device)
    t_float = t_int / T

    if t_edit_frac == 0.0:
        # No noise — return original
        return {
            "t_int": t_int,
            "t": t_float,
            "beta_t": torch.zeros(bs, 1, device=X.device),
            "alpha_s_bar": torch.ones(bs, 1, device=X.device),
            "alpha_t_bar": torch.ones(bs, 1, device=X.device),
            "X_t": X.clone(),
            "E_t": E.clone(),
            "y_t": y,
            "node_mask": node_mask,
        }

    # Linear noise schedule: alpha_bar(t) = 1 - t_frac
    alpha_t_bar = 1.0 - t_edit_frac
    alpha_s_bar = 1.0 - max(0.0, t_edit_frac - 1.0 / T)
    beta_t = 1.0 - (alpha_t_bar / max(alpha_s_bar, 1e-8))

    alpha_t_bar_t = torch.full((bs, 1), alpha_t_bar, device=X.device)
    beta_t_t = torch.full((bs, 1), beta_t, device=X.device)
    alpha_s_bar_t = torch.full((bs, 1), alpha_s_bar, device=X.device)

    # Corrupt X: interpolate between original and uniform noise
    x_noise = torch.ones_like(X) / xdim
    mix_x = alpha_t_bar * X + (1 - alpha_t_bar) * x_noise
    X_t = torch.zeros_like(X)
    indices = torch.multinomial(mix_x.view(-1, xdim), 1).squeeze(-1)
    X_t.view(-1, xdim).scatter_(1, indices.unsqueeze(-1), 1.0)

    # Corrupt E: same approach
    e_noise = torch.ones_like(E) / edim
    mix_e = alpha_t_bar * E + (1 - alpha_t_bar) * e_noise
    E_t = torch.zeros_like(E)
    e_flat = mix_e.view(-1, edim)
    e_idx = torch.multinomial(e_flat, 1).squeeze(-1)
    E_t.view(-1, edim).scatter_(1, e_idx.unsqueeze(-1), 1.0)
    E_t = E_t.view(bs, n, n, edim)

    # Apply node mask
    X_t = X_t * node_mask.unsqueeze(-1).float()
    E_t = E_t * node_mask.unsqueeze(-1).unsqueeze(-1).float()

    return {
        "t_int": t_int,
        "t": t_float,
        "beta_t": beta_t_t,
        "alpha_s_bar": alpha_s_bar_t,
        "alpha_t_bar": alpha_t_bar_t,
        "X_t": X_t,
        "E_t": E_t,
        "y_t": y,
        "node_mask": node_mask,
    }


def compute_t_edit_schedule(
    n: int, t_min: float = 0.1, t_max: float = 0.7
) -> List[float]:
    """Returns n evenly-spaced t_edit values from t_min to t_max.

    Used to generate diverse candidates: low t = conservative edits,
    high t = aggressive structural changes.
    """
    if n == 1:
        return [(t_min + t_max) / 2]
    step = (t_max - t_min) / (n - 1)
    return [t_min + i * step for i in range(n)]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/model/graph_editor/test_editor_utils.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/graph_editor/ tests/model/graph_editor/
git commit -m "feat: add editor_utils with partial noise injection and diversity schedule"
```

---

### Task 8: GraphEditorDiT Model

**Files:**
- Create: `src/model/graph_editor/editor_model.py`
- Modify: `tests/model/graph_editor/test_editor_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model/graph_editor/test_editor_model.py
import pytest
import torch
from unittest.mock import MagicMock, patch

def test_editor_transformer_accepts_source_and_operation():
    from src.model.graph_editor.editor_model import EditorTransformer
    model = EditorTransformer(
        max_n_nodes=20, hidden_size=64, depth=2, num_heads=4,
        mlp_ratio=2.0, drop_condition=0.0,
        Xdim=16, Edim=5, ydim=10, text_dim=768,
        source_dim=256, operation_dim=128,
    )
    bs, n = 2, 20
    X = torch.randn(bs, n, 16)
    E = torch.randn(bs, n, n, 5)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    y = torch.randn(bs, 10)
    text_emb = torch.randn(bs, 768)
    t = torch.zeros(bs)
    source_emb = torch.randn(bs, 256)
    op_emb = torch.randn(bs, 128)

    out = model(X, E, node_mask, y, text_emb, t,
                source_embedding=source_emb, operation_embedding=op_emb)
    assert hasattr(out, "X")
    assert out.X.shape == (bs, n, 16)

def test_editor_transformer_works_without_source_embedding():
    from src.model.graph_editor.editor_model import EditorTransformer
    model = EditorTransformer(
        max_n_nodes=20, hidden_size=64, depth=2, num_heads=4,
        mlp_ratio=2.0, drop_condition=0.0,
        Xdim=16, Edim=5, ydim=10, text_dim=768,
        source_dim=256, operation_dim=128,
    )
    bs, n = 2, 20
    X = torch.randn(bs, n, 16)
    E = torch.randn(bs, n, n, 5)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    y = torch.randn(bs, 10)
    text_emb = torch.randn(bs, 768)
    t = torch.zeros(bs)
    # No source or operation embedding — should still work
    out = model(X, E, node_mask, y, text_emb, t)
    assert out.X.shape == (bs, n, 16)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/model/graph_editor/test_editor_model.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement EditorTransformer and GraphEditorDiT**

```python
# src/model/graph_editor/editor_model.py
"""GraphEditorDiT: extends GraphDiT for molecule editing via partial noise injection.

Key additions over GraphDiT:
1. EditorTransformer: extends Transformer with source molecule and operation conditioning
2. apply_partial_noise_to_graph(): converts a SMILES source → dense graph → partial noise
3. generate_edits(): produces n_candidates diverse SMILES edits
"""
import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from src.model.graph_decoder.diffusion_model import GraphDiT
from src.model.graph_decoder.transformer import Transformer
from src.model.graph_decoder import diffusion_utils as utils
from src.model.graph_editor.editor_utils import apply_partial_noise, compute_t_edit_schedule


class EditorTransformer(Transformer):
    """Extends Transformer with two additional conditioning signals:
      - source_embedding: (bs, source_dim) from GraphCLIP encoder of source molecule
      - operation_embedding: (bs, operation_dim) from IntentParser
    Both are projected to hidden_size and added to the timestep conditioning vector.
    """

    def __init__(
        self,
        *args,
        source_dim: int = 256,
        operation_dim: int = 256,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        hidden_size = kwargs.get("hidden_size", 1024)
        self.source_proj = nn.Linear(source_dim, hidden_size)
        self.operation_proj = nn.Linear(operation_dim, hidden_size)

    def forward(
        self,
        X, E, node_mask, y, text_embedding, t,
        unconditioned: bool = False,
        source_embedding: Optional[torch.Tensor] = None,
        operation_embedding: Optional[torch.Tensor] = None,
    ):
        # Build base conditioning from parent (timestep + property + text)
        # We hook into the conditioning by overriding the adaLN shift/scale computation.
        # Strategy: project source/op to hidden_size, add to t_embedder output.
        # This requires calling the parent's component embedders explicitly.

        # Embed timestep
        t_emb = self.t_embedder(t)  # (bs, hidden_size)

        # Embed properties
        y_emb = self.y_embedder(y, self.training)  # (bs, hidden_size)

        # Embed text
        txt_emb = self.txt_embedder(text_embedding, self.training)  # (bs, hidden_size)

        # Conditioning vector (matches parent's c computation)
        c = t_emb + y_emb + txt_emb

        # Add source molecule conditioning
        if source_embedding is not None:
            c = c + self.source_proj(source_embedding.to(c.dtype))

        # Add operation conditioning
        if operation_embedding is not None:
            c = c + self.operation_proj(operation_embedding.to(c.dtype))

        # Run through transformer blocks using the combined conditioning
        bs, n, _ = X.shape
        x_flat = torch.cat([X, E.reshape(bs, n, -1)], dim=-1)
        x_emb = self.x_embedder(x_flat)

        for block in self.blocks:
            x_emb = block(x_emb, c)

        out = self.output_layer(x_emb, c)
        return out


class GraphEditorDiT(GraphDiT):
    """Extends GraphDiT for molecule editing via partial noise injection.

    Usage:
        editor = GraphEditorDiT(model_config_path, data_info_path, dtype)
        edits = editor.generate_edits(
            source_smiles="CC(=O)Oc1ccccc1C(=O)O",
            property_deltas={"logS": 1.5},
            source_embedding=graph_encoder_output,
            operation_embedding=intent_parser_output,
            n_candidates=5,
        )
    """

    def __init__(self, model_config_path, data_info_path, model_dtype,
                 source_dim: int = 256, operation_dim: int = 256):
        super().__init__(model_config_path, data_info_path, model_dtype)
        # Replace denoiser with EditorTransformer
        dm_cfg = self.model_config
        self.denoiser = EditorTransformer(
            max_n_nodes=self.max_n_nodes,
            hidden_size=dm_cfg.hidden_size,
            depth=dm_cfg.depth,
            num_heads=dm_cfg.num_heads,
            mlp_ratio=dm_cfg.mlp_ratio,
            drop_condition=dm_cfg.drop_condition,
            Xdim=self.Xdim,
            Edim=self.Edim,
            ydim=self.ydim,
            text_dim=self.text_input_size,
            source_dim=source_dim,
            operation_dim=operation_dim,
        )

    def _forward(self, noisy_data, text_embedding, unconditioned=False,
                 source_embedding=None, operation_embedding=None):
        noisy_x = noisy_data["X_t"].to(self.model_dtype)
        noisy_e = noisy_data["E_t"].to(self.model_dtype)
        properties = noisy_data["y_t"].to(self.model_dtype).clone()
        node_mask = noisy_data["node_mask"]
        timestep = noisy_data["t"]
        text_embedding = text_embedding.to(self.model_dtype)

        pred = self.denoiser(
            noisy_x, noisy_e, node_mask, properties, text_embedding, timestep,
            unconditioned=unconditioned,
            source_embedding=source_embedding,
            operation_embedding=operation_embedding,
        )
        return pred

    def forward_edit(self, x, edge_index, edge_attr, graph_batch,
                     properties, text_embedding, no_label_index,
                     source_embedding=None, operation_embedding=None):
        """Training forward pass on MMP pairs.
        Same as GraphDiT.forward() but passes extra conditioning to _forward.
        """
        import torch.nn.functional as F
        properties = torch.where(properties == no_label_index, float("nan"), properties)
        data_x = F.one_hot(x, num_classes=118).to(self.model_dtype)[:, self.active_index]
        data_edge_attr = F.one_hot(edge_attr, num_classes=5).to(self.model_dtype)
        dense_data, node_mask = utils.to_dense(
            data_x, edge_index, data_edge_attr, graph_batch, self.max_n_nodes
        )
        X, E = dense_data.X, dense_data.E
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(X, E, properties, node_mask)
        pred = self._forward(noisy_data, text_embedding,
                             source_embedding=source_embedding,
                             operation_embedding=operation_embedding)
        loss = self.train_loss(
            masked_pred_X=pred.X, masked_pred_E=pred.E,
            true_X=X, true_E=E, node_mask=node_mask,
        )
        return loss

    @torch.no_grad()
    def generate_edits(
        self,
        properties: torch.Tensor,
        text_embedding: torch.Tensor,
        no_label_index: float,
        source_X: torch.Tensor,       # (1, n, xdim) dense one-hot source graph
        source_E: torch.Tensor,       # (1, n, n, edim) dense source bonds
        source_node_mask: torch.Tensor,  # (1, n) bool
        source_embedding: Optional[torch.Tensor] = None,   # (1, hidden)
        operation_embedding: Optional[torch.Tensor] = None,  # (1, hidden)
        n_candidates: int = 5,
        t_min: float = 0.1,
        t_max: float = 0.7,
    ) -> List[Optional[str]]:
        """Generate n_candidates edited molecules by varying t_edit across [t_min, t_max]."""
        schedule = compute_t_edit_schedule(n_candidates, t_min, t_max)
        properties = torch.where(
            properties == no_label_index, float("nan"), properties
        )
        results = []
        for t_edit_frac in schedule:
            noisy_data = apply_partial_noise(
                source_X, source_E, properties, source_node_mask,
                t_edit_frac=t_edit_frac, T=self.T,
            )
            X, E = noisy_data["X_t"], noisy_data["E_t"]
            y = properties

            # Denoise from t_edit back to 0
            for s_int in reversed(range(0, round(t_edit_frac * self.T))):
                s_array = s_int * torch.ones((1, 1)).type_as(y)
                t_array = s_array + 1
                s_norm = s_array / self.T
                t_norm = t_array / self.T
                sampled_s, _ = self.sample_p_zs_given_zt(
                    s_norm, t_norm, X, E, y, text_embedding, source_node_mask,
                )
                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            sampled_s = sampled_s.mask(source_node_mask, collapse=True)
            X_final, E_final = sampled_s.X, sampled_s.E
            molecule_list = []
            for i in range(1):
                n = source_node_mask[i].sum().item()
                atom_types = X_final[i, :n].argmax(-1).tolist()
                bond_types_matrix = E_final[i, :n, :n].argmax(-1).tolist()
                smiles = self._atoms_bonds_to_smiles(atom_types, bond_types_matrix)
                molecule_list.append(smiles)
            results.append(molecule_list[0] if molecule_list else None)

        return results

    def _atoms_bonds_to_smiles(self, atom_types, bond_types_matrix) -> Optional[str]:
        """Wrapper around existing graph_to_smiles utility."""
        from src.model.graph_decoder.molecule_utils import graph_to_smiles
        try:
            return graph_to_smiles(atom_types, bond_types_matrix, self.atom_decoder)
        except Exception:
            return None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/model/graph_editor/test_editor_model.py -v
```
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/graph_editor/editor_model.py tests/model/graph_editor/test_editor_model.py
git commit -m "feat: add GraphEditorDiT with EditorTransformer and generate_edits()"
```

---

### Task 9: GraphEditorDiT Training Script

**Files:**
- Create: `src/model/graph_editor/train_editor.py`
- Create: `config/train/editor_lora.yaml`

- [ ] **Step 1: Write training script**

```python
# src/model/graph_editor/train_editor.py
"""Train GraphEditorDiT on MMP pairs (Stage 3).

Usage:
    python -m src.model.graph_editor.train_editor \
        --mmp_csv data/mmp_pairs.csv \
        --model_config_path saves/graph_decoder/model_config.yaml \
        --data_info_path saves/graph_decoder/data.meta.json \
        --graph_encoder_path saves/graph_encoder \
        --output_dir saves/graph_editor \
        --epochs 20
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

from src.model.graph_editor.editor_model import GraphEditorDiT
from src.model.graph_encoder.model import GraphCLIP
from src.model.intent_parser import IntentParser
from src.data.mmp_loader import MMPDataset


def collate_mmp(batch):
    from torch_geometric.data import Batch
    graphs_a = Batch.from_data_list([item["graph_a"] for item in batch])
    graphs_b = Batch.from_data_list([item["graph_b"] for item in batch])
    deltas = torch.tensor([item["delta"] for item in batch], dtype=torch.float)
    op_indices = torch.tensor([item["operation_idx"] for item in batch], dtype=torch.long)
    return graphs_a, graphs_b, deltas, op_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmp_csv", required=True)
    parser.add_argument("--model_config_path", required=True)
    parser.add_argument("--data_info_path", required=True)
    parser.add_argument("--graph_encoder_path", required=True)
    parser.add_argument("--output_dir", default="saves/graph_editor")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Load editor model
    editor = GraphEditorDiT(
        model_config_path=args.model_config_path,
        data_info_path=args.data_info_path,
        model_dtype=dtype,
        source_dim=256,
        operation_dim=256,
    ).to(device)

    # Load graph encoder (frozen)
    # Adjust GraphCLIP init args to match your saved config
    encoder = GraphCLIP(
        graph_num_layer=5,
        graph_hidden_size=256,
        dropout=0.0,
        model_config={},
    ).to(device)
    encoder_path = os.path.join(args.graph_encoder_path, "model.pt")
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    for p in encoder.parameters():
        p.requires_grad = False

    intent_parser = IntentParser(embedding_dim=256).to(device)
    dataset = MMPDataset(csv_path=args.mmp_csv)
    loader = TorchDataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mmp
    )
    optimizer = torch.optim.Adam(
        list(editor.parameters()) + list(intent_parser.parameters()), lr=args.lr
    )

    no_label_index = -200.0

    for epoch in range(args.epochs):
        editor.train()
        intent_parser.train()
        epoch_loss = 0.0
        for graphs_a, graphs_b, deltas, op_indices in loader:
            graphs_a = graphs_a.to(device)
            graphs_b = graphs_b.to(device)
            deltas = deltas.to(device)
            op_indices = op_indices.to(device)

            # Encode source molecule
            with torch.no_grad():
                source_emb = encoder(
                    graphs_a.x.float(), graphs_a.edge_index,
                    graphs_a.edge_attr, graphs_a.batch
                )  # (bs, 256)

            # Encode operation
            op_emb = intent_parser.embeddings(op_indices)  # (bs, 256)

            # Build dummy text embedding and properties (zeros = unconditional)
            bs = deltas.shape[0]
            text_emb = torch.zeros(bs, 768, device=device)
            # Use delta as the single property conditioning signal
            properties = torch.full((bs, editor.ydim), no_label_index, device=device)

            loss = editor.forward_edit(
                x=graphs_b.x.long(),
                edge_index=graphs_b.edge_index,
                edge_attr=graphs_b.edge_attr.long(),
                graph_batch=graphs_b.batch,
                properties=properties,
                text_embedding=text_emb,
                no_label_index=no_label_index,
                source_embedding=source_emb,
                operation_embedding=op_emb,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(editor.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}  loss={epoch_loss/len(loader):.4f}")

    editor.save_pretrained(args.output_dir)
    intent_parser.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write training config**

```yaml
# config/train/editor_lora.yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
graph_encoder_path: saves/graph_encoder
graph_decoder_path: saves/graph_decoder  # reused as base for editor
graph_editor_path: saves/graph_editor    # output path

# Training
mmp_csv: data/mmp_pairs.csv
output_dir: saves/graph_editor_trained
epochs: 20
learning_rate: 1.0e-4
per_device_train_batch_size: 16
gradient_accumulation_steps: 2

# Editor settings
t_min: 0.1
t_max: 0.7
n_candidates: 5
source_dim: 256
operation_dim: 256
```

- [ ] **Step 3: Smoke test**

```bash
# Create a tiny dummy MMP CSV first
python -c "
import pandas as pd
df = pd.DataFrame([
    {'smiles_a': 'c1ccccc1', 'smiles_b': 'Oc1ccccc1', 'property_name': 'logS', 'delta': 1.0, 'operation_label': 'add_hbd'},
    {'smiles_a': 'CC(=O)O', 'smiles_b': 'OCC(=O)O', 'property_name': 'logS', 'delta': 0.5, 'operation_label': 'add_hbd'},
])
df.to_csv('/tmp/test_mmp.csv', index=False)
print('Created /tmp/test_mmp.csv')
"
```
Expected: `Created /tmp/test_mmp.csv`

(Full training is run after model checkpoints are available — see Phase 5)

- [ ] **Step 4: Commit**

```bash
git add src/model/graph_editor/train_editor.py config/train/editor_lora.yaml
git commit -m "feat: add GraphEditorDiT training script and config"
```

---

## Phase 5: Integration

### Task 10: New Special Tokens and Edit Prompt Template

**Files:**
- Modify: `src/extras/constants.py`
- Modify: `src/data/template.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_constants.py
from src.extras.constants import EDIT_SPECIAL_TOKENS, EDIT_TOKEN_NAMES

def test_edit_special_tokens_defined():
    assert "<edit_start>" in EDIT_TOKEN_NAMES
    assert "<edit_end>" in EDIT_TOKEN_NAMES
    assert "<edit_body>" in EDIT_TOKEN_NAMES
    assert "<operation>" in EDIT_TOKEN_NAMES

def test_edit_special_tokens_unique():
    assert len(set(EDIT_TOKEN_NAMES)) == len(EDIT_TOKEN_NAMES)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_constants.py -v
```
Expected: `ImportError: cannot import name 'EDIT_SPECIAL_TOKENS'`

- [ ] **Step 3: Add constants**

In `src/extras/constants.py`, add after the existing special token constants:

```python
# Edit task special tokens
EDIT_TOKEN_NAMES = ["<edit_start>", "<edit_end>", "<edit_body>", "<operation>"]
EDIT_SPECIAL_TOKENS = {name: None for name in EDIT_TOKEN_NAMES}  # IDs filled at model load
```

- [ ] **Step 4: Add edit prompt template**

In `src/data/template.py`, add an edit template alongside existing design/retro templates. Read the file first to find the right insertion point, then add:

```python
EDIT_TEMPLATE = (
    "{instruction}"
    "Given the molecule <molecule>, propose a modification that improves {property_name} "
    "by approximately {delta:+.1f}. "
    "<edit_start>{analysis}<edit_end>"
)

EDIT_RATIONALE_TEMPLATE = (
    "The modification from {smiles_a} to {smiles_b} improves {property_name} "
    "because {rationale}. "
    "Predicted change: {delta:+.2f}. "
    "Synthetic accessibility score: {sa:.2f}."
)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_constants.py -v
```
Expected: both tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/extras/constants.py src/data/template.py tests/test_constants.py
git commit -m "feat: add edit special tokens and prompt templates"
```

---

### Task 11: GraphLLMForCausalMLM.edit() Method

**Files:**
- Modify: `src/model/modeling_llamole.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/integration/test_edit_workflow.py
import pytest
import torch
from unittest.mock import MagicMock, patch

def test_edit_method_exists():
    """edit() is callable on GraphLLMForCausalMLM."""
    import inspect
    from src.model.modeling_llamole import GraphLLMForCausalMLM
    assert hasattr(GraphLLMForCausalMLM, "edit")
    sig = inspect.signature(GraphLLMForCausalMLM.edit)
    params = list(sig.parameters.keys())
    assert "molecule" in params
    assert "target_property" in params
    assert "target_delta" in params
    assert "n_candidates" in params

def test_modification_dataclass_fields():
    from src.model.modeling_llamole import Modification
    mod = Modification(
        smiles="Oc1ccccc1",
        property_scores={"logS": -1.2, "logP": 1.8},
        property_deltas={"logS": 0.8, "logP": -0.3},
        rationale="Added hydroxyl group increases H-bond donors.",
        t_edit=0.3,
    )
    assert mod.smiles == "Oc1ccccc1"
    assert mod.property_deltas["logS"] == pytest.approx(0.8)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/integration/test_edit_workflow.py::test_edit_method_exists -v
python -m pytest tests/integration/test_edit_workflow.py::test_modification_dataclass_fields -v
```
Expected: `AttributeError` or `ImportError`

- [ ] **Step 3: Add Modification dataclass and edit() method to modeling_llamole.py**

Read `src/model/modeling_llamole.py` first to find the class definition, then add:

After the imports block, add:
```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional as Opt

@dataclass
class Modification:
    smiles: str
    property_scores: Dict[str, float]
    property_deltas: Dict[str, float]
    rationale: str
    t_edit: float
```

Inside `GraphLLMForCausalMLM`, add the `edit()` method (after the `retrosynthesize()` method):

```python
def edit(
    self,
    molecule: str,
    target_property: str,
    target_delta: float = 1.0,
    instruction: str = "",
    n_candidates: int = 5,
    t_min: float = 0.1,
    t_max: float = 0.7,
    **kwargs,
) -> List["Modification"]:
    """Generate n_candidates modifications of molecule improving target_property.

    Args:
        molecule: SMILES string or molecule name (resolved via MoleculeNameResolver)
        target_property: one of "logS", "logP", "PAMPA", "metabolic_stability"
        target_delta: desired improvement magnitude (positive = improve)
        instruction: optional NL instruction (e.g. "add a hydrogen bond donor")
        n_candidates: number of diverse modifications to generate
        t_min / t_max: range of noise levels for diversity
    """
    from src.tools.molecule_resolver import MoleculeNameResolver
    from src.model.graph_editor.editor_model import GraphEditorDiT
    from src.model.admet_predictor.predictor_model import ADMETPredictor
    from src.model.intent_parser import IntentParser
    from src.model.graph_decoder.molecule_utils import check_valid

    # Step 1: resolve name → SMILES
    resolver = MoleculeNameResolver()
    source_smiles = resolver.resolve(molecule)
    if source_smiles is None:
        raise ValueError(f"Could not resolve molecule: {molecule!r}")

    # Step 2: encode source molecule
    source_graph = self.smiles_to_graph(source_smiles)
    if source_graph is None:
        raise ValueError(f"Invalid SMILES: {source_smiles!r}")

    from torch_geometric.data import Batch
    source_batch = Batch.from_data_list([source_graph]).to(self.device)
    source_emb = self.graph_encoder(
        source_batch.x.float(), source_batch.edge_index,
        source_batch.edge_attr, source_batch.batch
    )  # (1, encoder_hidden)

    # Step 3: encode instruction
    operation_emb = self.intent_parser.get_operation_embedding(instruction)
    operation_emb = operation_emb.unsqueeze(0).to(self.device)

    # Step 4: build text embedding from LLM
    prompt = (
        f"Modify the molecule to improve {target_property} by {target_delta:+.1f}. "
        f"Instruction: {instruction}" if instruction else
        f"Modify the molecule to improve {target_property} by {target_delta:+.1f}."
    )
    input_ids = self.tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(self.device)
    with torch.no_grad():
        text_emb = self.language_model.model.embed_tokens(input_ids).mean(dim=1)  # (1, lm_hidden)
    # Project to graph decoder text_input_size (768)
    if hasattr(self, "text_proj_edit"):
        text_emb = self.text_proj_edit(text_emb)
    else:
        text_emb = torch.zeros(1, 768, device=self.device)

    # Step 5: prepare source dense graph for editor
    import torch.nn.functional as F
    from src.model.graph_decoder import diffusion_utils as utils
    data_x = F.one_hot(source_batch.x.long().squeeze(-1), num_classes=118).float()
    data_x = data_x[:, self.graph_decoder.active_index]
    data_e = F.one_hot(source_batch.edge_attr.long(), num_classes=5).float()
    dense, node_mask = utils.to_dense(
        data_x, source_batch.edge_index, data_e,
        source_batch.batch, self.graph_decoder.max_n_nodes
    )

    # Step 6: build property tensor with target delta
    no_label_idx = self.token_id_dict.get("no_label_index", -200)
    properties = torch.full(
        (1, self.graph_decoder.ydim), no_label_idx, device=self.device
    )

    # Step 7: generate edits
    candidate_smiles = self.graph_editor.generate_edits(
        properties=properties,
        text_embedding=text_emb,
        no_label_index=no_label_idx,
        source_X=dense.X,
        source_E=dense.E,
        source_node_mask=node_mask,
        source_embedding=source_emb,
        operation_embedding=operation_emb,
        n_candidates=n_candidates,
        t_min=t_min,
        t_max=t_max,
    )

    # Step 8: score candidates
    source_scores = self._score_admet(source_smiles)
    modifications = []
    for smiles in candidate_smiles:
        if smiles is None or not check_valid(smiles):
            continue
        candidate_scores = self._score_admet(smiles)
        deltas = {p: candidate_scores[p] - source_scores[p] for p in candidate_scores}
        rationale = self._generate_edit_rationale(
            source_smiles, smiles, target_property, deltas[target_property]
        )
        modifications.append(Modification(
            smiles=smiles,
            property_scores=candidate_scores,
            property_deltas=deltas,
            rationale=rationale,
            t_edit=0.0,  # filled in by generate_edits in future
        ))

    # Step 9: rank by target property delta (descending)
    modifications.sort(key=lambda m: m.property_deltas.get(target_property, 0), reverse=True)
    return modifications

def _score_admet(self, smiles: str) -> Dict[str, float]:
    """Score a single SMILES with ADMETPredictor."""
    from torch_geometric.data import Batch
    graph = self.smiles_to_graph(smiles)
    if graph is None:
        return {p: 0.0 for p in ["logS", "logP", "PAMPA", "metabolic_stability"]}
    batch = Batch.from_data_list([graph]).to(self.device)
    preds = self.admet_predictor.predict(batch)
    return {p: v[0] for p, v in preds.items()}

def _generate_edit_rationale(
    self, smiles_a: str, smiles_b: str, property_name: str, delta: float
) -> str:
    """Ask LLM to explain why the modification improves property_name."""
    prompt = (
        f"Original molecule: {smiles_a}\n"
        f"Modified molecule: {smiles_b}\n"
        f"Explain in one sentence why this structural change improves {property_name} "
        f"(predicted change: {delta:+.2f})."
    )
    input_ids = self.tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(self.device)
    output_ids = self.language_model.generate(
        inputs=input_ids,
        max_new_tokens=80,
        do_sample=False,
    )
    decoded = self.tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    )
    return decoded.strip()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/integration/test_edit_workflow.py -v
```
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/modeling_llamole.py tests/integration/
git commit -m "feat: add edit() method and Modification dataclass to GraphLLMForCausalMLM"
```

---

## Phase 6: Web UI

### Task 12: Gradio "Modify Molecule" Tab

**Files:**
- Modify: `launch.py`

- [ ] **Step 1: Read launch.py to understand existing UI structure**

```bash
# Read the existing Gradio tab structure before editing
# launch.py uses gr.Tabs() with gr.TabItem() — find the pattern and add a new tab
```

- [ ] **Step 2: Add the modify tab**

Read `launch.py` fully first. Then locate the `with gr.Tabs():` block and add:

```python
with gr.TabItem("Modify Molecule"):
    gr.Markdown("## Molecule Modification")
    gr.Markdown("Enter a molecule by name or SMILES and specify what to improve.")

    with gr.Row():
        with gr.Column(scale=1):
            mol_input = gr.Textbox(
                label="Molecule (name or SMILES)",
                placeholder="aspirin  OR  CC(=O)Oc1ccccc1C(=O)O",
            )
            property_dropdown = gr.Dropdown(
                choices=["logS", "logP", "PAMPA", "metabolic_stability"],
                value="logS",
                label="Property to improve",
            )
            delta_slider = gr.Slider(
                minimum=0.1, maximum=3.0, step=0.1, value=1.0,
                label="Target improvement (Δ)",
            )
            instruction_box = gr.Textbox(
                label="Instruction (optional)",
                placeholder="e.g. add a hydrogen bond donor",
            )
            n_candidates_slider = gr.Slider(
                minimum=1, maximum=10, step=1, value=5,
                label="Number of candidates",
            )
            modify_btn = gr.Button("Generate Modifications", variant="primary")

        with gr.Column(scale=2):
            modifications_output = gr.JSON(label="Ranked Modifications")
            mol_images = gr.Gallery(label="Molecule Structures", columns=5)

    with gr.Row():
        selected_smiles = gr.Textbox(label="Selected SMILES for synthesis")
        synthesize_btn = gr.Button("Synthesize Selected")
        synthesis_output = gr.JSON(label="Synthesis Route")

    def run_modify(mol, prop, delta, instruction, n):
        try:
            mods = model.edit(
                molecule=mol,
                target_property=prop,
                target_delta=float(delta),
                instruction=instruction,
                n_candidates=int(n),
            )
            results = [
                {
                    "smiles": m.smiles,
                    "property_scores": m.property_scores,
                    "property_deltas": m.property_deltas,
                    "rationale": m.rationale,
                }
                for m in mods
            ]
            # Generate RDKit images
            from rdkit import Chem
            from rdkit.Chem import Draw
            import io
            from PIL import Image
            images = []
            for m in mods:
                mol_obj = Chem.MolFromSmiles(m.smiles)
                if mol_obj:
                    img = Draw.MolToImage(mol_obj, size=(300, 300))
                    images.append(img)
            return results, images
        except Exception as e:
            return [{"error": str(e)}], []

    def run_synthesize(smiles):
        if not smiles:
            return {"error": "No SMILES provided"}
        try:
            route = model.retrosynthesize(smiles)
            return {"route": str(route)}
        except Exception as e:
            return {"error": str(e)}

    modify_btn.click(
        fn=run_modify,
        inputs=[mol_input, property_dropdown, delta_slider, instruction_box, n_candidates_slider],
        outputs=[modifications_output, mol_images],
    )
    synthesize_btn.click(
        fn=run_synthesize,
        inputs=[selected_smiles],
        outputs=[synthesis_output],
    )
```

- [ ] **Step 3: Smoke test the UI (no model loaded)**

```bash
python launch.py --help
```
Expected: no import errors

- [ ] **Step 4: Commit**

```bash
git add launch.py
git commit -m "feat: add Modify Molecule tab to Gradio UI"
```

---

## Phase 7: Generate Config & Final Wiring

### Task 13: Inference Config and Module Registration

**Files:**
- Create: `config/generate/editor_config.yaml`
- Modify: `src/model/loader.py` (register graph_editor + admet_predictor)

- [ ] **Step 1: Write inference config**

```yaml
# config/generate/editor_config.yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: saves/llamole_lora
graph_encoder_path: saves/graph_encoder
graph_decoder_path: saves/graph_decoder
graph_editor_path: saves/graph_editor_trained
admet_predictor_path: saves/admet_predictor
intent_parser_path: saves/graph_editor_trained

# Generation settings
n_candidates: 5
t_min: 0.1
t_max: 0.7
temperature: 0.6
top_p: 0.9
max_new_tokens: 80
```

- [ ] **Step 2: Read src/model/loader.py and add editor/ADMET loading**

Read `src/model/loader.py` to understand how `graph_encoder`, `graph_decoder`, `graph_predictor` are loaded. Then add analogous loading for `graph_editor`, `admet_predictor`, and `intent_parser` in the same pattern, gated on whether `graph_editor_path` is set in the model args.

- [ ] **Step 3: Add graph_editor_path and admet_predictor_path to model args**

Read `src/hparams/model_args.py`. Add:

```python
graph_editor_path: Optional[str] = field(
    default=None,
    metadata={"help": "Path to saved GraphEditorDiT checkpoint directory."},
)
admet_predictor_path: Optional[str] = field(
    default=None,
    metadata={"help": "Path to saved ADMETPredictor checkpoint directory."},
)
intent_parser_path: Optional[str] = field(
    default=None,
    metadata={"help": "Path to saved IntentParser checkpoint directory."},
)
```

- [ ] **Step 4: Commit**

```bash
git add config/generate/editor_config.yaml src/hparams/model_args.py src/model/loader.py
git commit -m "feat: add inference config and model arg registration for editor/ADMET"
```

---

## Training Execution Order

After all code is in place, run training in this order:

```bash
# 1. Train ADMETPredictor (independent, ~2–4h on GPU)
python -m src.model.admet_predictor.train_admet \
    --output_dir saves/admet_predictor \
    --epochs 50

# 2. Export your MMP database to CSV (using your existing MMP code):
#    columns: smiles_a, smiles_b, property_name, delta, operation_label
#    Save as: data/mmp_pairs.csv

# 3. Train GraphEditorDiT on MMP pairs (requires graph_decoder checkpoint)
python -m src.model.graph_editor.train_editor \
    --mmp_csv data/mmp_pairs.csv \
    --model_config_path saves/graph_decoder/model_config.yaml \
    --data_info_path saves/graph_decoder/data.meta.json \
    --graph_encoder_path saves/graph_encoder \
    --output_dir saves/graph_editor_trained \
    --epochs 20

# 4. Launch web UI with editor enabled
python launch.py --config config/generate/editor_config.yaml
```

---

## Open Questions (resolve before training)

1. **MMP CSV format:** Confirm column names from your MMP codebase match `smiles_a`, `smiles_b`, `property_name`, `delta`, `operation_label` — or update `mmp_loader.py:_load_pairs()` accordingly.
2. **GraphEncoder hidden size:** Confirm `graph_hidden_size` in your saved `graph_encoder/model_config.json` — this must match `source_dim` in `GraphEditorDiT` and `editor_lora.yaml`.
3. **PAMPA + metabolic stability datasets:** Add dataset classes to `admet_datasets.py` when data is available — training script already has placeholder calls.
4. **IntentParser Phase 2:** Once MMP operation labels are confirmed, generate NL description → label pairs and fine-tune the base LLM with LoRA to replace the rule-based `_classify_instruction`.
