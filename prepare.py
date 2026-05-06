"""
prepare.py  — LOCKED, do not modify.

Loads embeddings and split CSVs from the main repo, builds DataLoaders,
and exposes evaluate() for use in train.py.

The test split is loaded internally for evaluate() but is NEVER returned
to callers — get_loaders() returns only (train_loader, val_loader).
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Paths — embeddings live in the original data directory
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent / "drug_protein_affinity"
_DATA = _REPO / "data"

DRUG_CLS_PATH = _DATA / "drug_emb_cls.pt"
PROT_CLS_PATH = _DATA / "protein_emb_cls.pt"

SPLITS_DIR = _DATA / "splits"
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV   = SPLITS_DIR / "val.csv"
TEST_CSV  = SPLITS_DIR / "test.csv"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AffinityDataset(Dataset):
    def __init__(self, df, drug_cache, prot_cache):
        self.rows = df.reset_index(drop=True)
        self.drug_cache = drug_cache
        self.prot_cache = prot_cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        drug_emb = self.drug_cache[row["Drug"]]
        prot_emb = self.prot_cache[row["Target"]]
        protein_family = row["Protein_Family"] if "Protein_Family" in row else "other"
        y = torch.tensor(row["pKd"], dtype=torch.float)
        return drug_emb, prot_emb, y, protein_family


# ---------------------------------------------------------------------------
# Weighted sampler (used in training to oversample rare proteins)
# ---------------------------------------------------------------------------
def _make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    protein_counts = df["Target"].value_counts()
    weights = df["Target"].map(lambda p: 1.0 / np.sqrt(protein_counts[p])).values
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(df),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------
def compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_fn) -> torch.Tensor:
    """Handles both MSE (output_dim=1) and GaussianNLL (output_dim=2)."""
    if pred.dim() == 1:
        return loss_fn(pred, target)
    elif pred.dim() == 2 and pred.shape[-1] == 2:
        mu = pred[:, 0]
        log_var = pred[:, 1]
        var = torch.exp(log_var).clamp(min=1e-6)
        return loss_fn(mu, target, var)
    else:
        raise ValueError(f"Unexpected pred shape: {pred.shape}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_loaders(batch_size: int = 256):
    """Returns (train_loader, val_loader). Test loader is never exposed."""
    drug_cache = torch.load(DRUG_CLS_PATH, weights_only=True)
    prot_cache = torch.load(PROT_CLS_PATH, weights_only=True)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    train_ds = AffinityDataset(train_df, drug_cache, prot_cache)
    val_ds   = AffinityDataset(val_df,   drug_cache, prot_cache)

    train_sampler = _make_weighted_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def evaluate(model, loader, device) -> dict:
    """Evaluate model on loader; returns dict with at least 'RMSE'."""
    loss_fn = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    all_pred, all_true, all_fam = [], [], []

    with torch.no_grad():
        for drug_emb, prot_emb, pkd, family in loader:
            drug_emb = drug_emb.to(device)
            prot_emb = prot_emb.to(device)
            pkd = pkd.to(device)

            pred = model(drug_emb, prot_emb)
            loss = compute_loss(pred, pkd, loss_fn)
            total_loss += loss.item()

            if pred.dim() == 1:
                all_pred.extend(pred.cpu().tolist())
            elif pred.dim() == 2 and pred.shape[-1] == 2:
                all_pred.extend(pred[:, 0].cpu().tolist())

            all_true.extend(pkd.cpu().tolist())
            all_fam.extend(family)

    results = {
        "loss": total_loss / len(loader),
        "RMSE": float(np.sqrt(mean_squared_error(all_true, all_pred))),
    }
    df_eval = pd.DataFrame({"true": all_true, "pred": all_pred, "family": all_fam})
    for fam, grp in df_eval.groupby("family"):
        if len(grp) > 1:
            results[f"{fam}_RMSE"] = float(
                np.sqrt(mean_squared_error(grp["true"], grp["pred"]))
            )
    return results
