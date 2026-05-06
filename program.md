# Research agenda — affinity MLP

## Goal
Minimize VAL RMSE on the held-out val set. The test set is sacred — never read, never referenced during the loop.

## Baseline (locked)
Current MLPPredictor from the notebook. Initial val RMSE ≈ 1.3577, train RMSE ≈ 0.7029. The gap is the problem to solve.

## Priority hypotheses (try in this order, ONE knob per experiment)

1. **Architectural fix**: change layer order from Linear → ReLU → BN → Dropout to canonical Linear → BN → ReLU → Dropout.
2. **LayerNorm in place of BatchNorm** in the hidden block. Linear → LayerNorm → ReLU → Dropout. LN avoids train/eval running-stats mismatch.
3. **Dropout sweep**: 0.1, 0.3, 0.4, 0.5.
4. **Modality dropout**: with probability p ∈ {0.05, 0.1, 0.15}, zero the entire drug or entire protein input vector during training (independent decisions per sample).
5. **Input vector dropout**: nn.Dropout on the concatenated input before the first Linear, p ∈ {0.1, 0.2}.
6. **Label noise**: add Gaussian noise N(0, σ) to pKd targets during training only, σ ∈ {0.05, 0.1, 0.2}.
7. **Weight decay**: AdamW with wd ∈ {1e-5, 1e-4, 1e-3}.
8. **Activation**: GELU, SiLU, then SwiGLU. SwiGLU is gated (two projections + element-wise mul) — match parameter budget when comparing.
9. **Width/depth**: hidden dims [256], [768, 384], [1024, 512, 256], [512, 512, 256].
10. **Residual connections** between consecutive same-width hidden layers.
11. **LR sweep**: 5e-3, 1e-3, 5e-4, 5e-5. Also try cosine annealing schedule.
12. **Loss head**: switch to GaussianNLL (output_dim=2) — lets the model express uncertainty on noisy inter-lab labels.

## Hard rules
- Edit `train.py` only. NEVER touch `prepare.py`. NEVER read the test set.
- ONE knob per experiment. If you need to change two things to test a hypothesis (e.g., SwiGLU requires changing the hidden block), make that explicit in the description but keep the change minimal.
- Fixed seed=42 always.
- Commit if val RMSE improves by ≥ 0.01. Otherwise `git reset --hard HEAD`.
- If a run crashes or returns NaN/inf, log the failure and reset.
- Append rows to `results.csv` using Python's `csv` module (or `pandas.to_csv` with `mode='a'`), never with raw f-string concatenation. This ensures fields containing commas are properly quoted.
- When all 12 priorities have been tried, propose new hypotheses (combinations of the best individual wins, or new directions) and append them to this file before continuing.
