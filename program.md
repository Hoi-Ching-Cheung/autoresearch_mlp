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

## Phase 3 hypotheses (runs 039+, proposed after 38 experiments; best so far: 1.2769)

Key observations: modality dropout p=0.10 is the only reliable win (+0.042 RMSE); AdamW wd=1e-4 came close (1.2816); deeper arch [1024,512,256] was also close (1.2903). Everything else regressed.

13. **SwiGLU activation**: gated linear unit with two projections (gate + value), sigmoid-gate product per layer. Halve projection width to match param budget.
14. **LR 1e-3**: LR sweep value never tried (tried 5e-4 and 5e-5, not 1e-3).
15. **LR 5e-3**: Highest LR sweep value never tried.
16. **Modality dropout p=0.08**: Fine-tune between p=0.05 (1.2910) and p=0.10 (1.2769).
17. **Embedding noise**: Add N(0, 0.01) Gaussian noise to drug and protein embeddings during training — softer than dropout.
18. **AdamW wd=5e-5**: Between Adam (0 wd) and tried 1e-5; the 1e-4 result was near best at 1.2816.
19. **AdamW wd=2e-4**: Finer sweep between wd=1e-4 (1.2816) and wd=1e-3 (1.3004).
20. **Gradient clipping = 0.5**: Tighter than current 1.0, additional implicit regularization.
21. **modality_drop=0.10 + hidden dims [1024,512,256]**: Best modality dropout with best near-miss architecture.
22. **Target clamping**: Clamp pKd training targets to [3, 12] to remove outlier labels.
23. **ELU activation**: Smooth negative saturation, not yet tried.
24. **modality_drop=0.10 + AdamW wd=2e-4**: Combine best modality regularization with the closest weight-decay candidate.

## Hard rules
- Edit `train.py` only. NEVER touch `prepare.py`. NEVER read the test set.
- ONE knob per experiment. If you need to change two things to test a hypothesis (e.g., SwiGLU requires changing the hidden block), make that explicit in the description but keep the change minimal.
- Fixed seed=42 always.
- Commit if val RMSE improves by ≥ 0.01. Otherwise `git reset --hard HEAD`.
- If a run crashes or returns NaN/inf, log the failure and reset.
- Append rows to `results.csv` using Python's `csv` module (or `pandas.to_csv` with `mode='a'`), never with raw f-string concatenation. This ensures fields containing commas are properly quoted.
- When all 12 priorities have been tried, propose new hypotheses (combinations of the best individual wins, or new directions) and append them to this file before continuing.
