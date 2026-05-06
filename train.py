"""
train.py  — mutable experiment file.

All hyperparameters are at the top as named constants.
main() trains, evaluates on val, then prints:
    VAL_RMSE=<float>
    TRAIN_RMSE=<float>
"""

import random

import numpy as np
import torch
import torch.nn as nn

from prepare import get_loaders, evaluate, compute_loss

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED            = 42
LR              = 1e-4
HIDDEN_DIMS     = [512, 256]
DROPOUT         = 0.2
EPOCHS          = 100
PATIENCE        = 20
BATCH_SIZE      = 256
MODALITY_DROP_P = 0.05   # probability of zeroing an entire drug or protein vector

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# Model  (verbatim from notebook cell 83)
# ---------------------------------------------------------------------------
class MLPPredictor(nn.Module):
    def __init__(
        self,
        drug_dim: int = 768,
        protein_dim: int = 1024,
        hidden_dim: list = None,
        output_dim: int = 1,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = HIDDEN_DIMS

        self.drug_norm = nn.LayerNorm(drug_dim)
        self.prot_norm = nn.LayerNorm(protein_dim)
        self.output_dim = output_dim

        input_dim = drug_dim + protein_dim
        layers = []
        in_dim = input_dim
        for h in hidden_dim:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, drug_emb, prot_emb):
        drug_emb = self.drug_norm(drug_emb)
        prot_emb = self.prot_norm(prot_emb)
        x = torch.cat([drug_emb, prot_emb], dim=1)
        out = self.mlp(x)
        if self.output_dim == 1:
            return out.squeeze(dim=-1)
        else:
            mu = out[:, 0]
            log_var = out[:, 1]
            return torch.stack([mu, log_var], dim=1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _apply_modality_dropout(emb: torch.Tensor, p: float) -> torch.Tensor:
    """Zero entire rows of emb independently with probability p (training only)."""
    mask = (torch.rand(emb.size(0), device=emb.device) > p).float().unsqueeze(1)
    return emb * mask


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for drug_emb, prot_emb, pkd, _ in loader:
        drug_emb = drug_emb.to(device)
        prot_emb = prot_emb.to(device)
        pkd = pkd.to(device)
        if MODALITY_DROP_P > 0.0:
            drug_emb = _apply_modality_dropout(drug_emb, MODALITY_DROP_P)
            prot_emb = _apply_modality_dropout(prot_emb, MODALITY_DROP_P)
        optimizer.zero_grad()
        pred = model(drug_emb, prot_emb)
        loss = compute_loss(pred, pkd, loss_fn)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    train_loader, val_loader = get_loaders(BATCH_SIZE)

    model = MLPPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_results = evaluate(model, val_loader, device)
        val_rmse = val_results["RMSE"]

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    # Load best weights for final eval
    if best_state is not None:
        model.load_state_dict(best_state)

    val_results   = evaluate(model, val_loader,   device)
    train_results = evaluate(model, train_loader, device)

    print(f"VAL_RMSE={val_results['RMSE']:.4f}")
    print(f"TRAIN_RMSE={train_results['RMSE']:.4f}")


if __name__ == "__main__":
    main()
