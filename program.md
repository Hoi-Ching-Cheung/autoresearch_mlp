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

## Phase 4 hypotheses (runs 051+, after 50 experiments; best 1.2769, near-miss 1.2712)

Key insight from Phase 3: only architecture scale (deeper/wider) with modality_drop=0.10 gets close. No other single knob beats it. Need to explore around run 047's near-miss config (modality_drop=0.10 + [1024,512,256]).

25. **hidden dims [2048, 1024, 512, 256]**: Even wider — does scaling up further help?
26. **hidden dims [1024, 512, 256, 128]**: Deeper narrowing pyramid with modality_drop=0.10.
27. **hidden dims [1024, 512]**: Wider two-layer vs current [512,256].
28. **modality_drop=0.10 + [512,512,256]**: Second-best arch (run 022: 1.2858) with modality dropout.
29. **modality_drop=0.10 + [1024,512,256] + GELU**: Combine near-miss arch with GELU that was close (run 017: 1.289).
30. **Drug-only modality dropout p=0.10**: Zero only drug vector; protein always present — check which modality matters more.
31. **Protein-only modality dropout p=0.10**: Zero only protein vector.
32. **Input concat LayerNorm**: Add nn.LayerNorm on the (drug||prot) concatenated vector before MLP, keeping modality_drop.
33. **Mixup augmentation**: Mix two training samples λ*sample_a + (1-λ)*sample_b with λ~Beta(0.4,0.4), same for labels.
34. **dropout=0.15**: Fine-tune between 0.1 (run 002: 1.3253) and 0.2 (best: 1.2769).
35. **modality_drop=0.10 + [1024,512,256] + dropout=0.15**: Best arch + tuned dropout.
36. **hidden dims [2048, 512, 256]**: Wide first layer compressing to standard depth.

## Phase 5 hypotheses (runs 075+, after 74 experiments; best 1.2322; near-miss 1.2314 at alpha=0.5)

Key observations: Mixup α=0.4 + [1024,512,256] + modality_drop=0.10 is the dominant config. Near-miss at α=0.5 (1.2314, delta=+0.0008). All single-knob variations on this config tried. Manifold Mixup hurt. Train-val gap (~0.59 vs 1.23) indicates regularization is still the lever.

37. **Mixup alpha=0.45**: Fine-tune between best (0.4→1.2322) and near-miss (0.5→1.2314).
38. **Embedding noise std=0.01 + Mixup best config**: Run 043 tried noise standalone (1.2872); never tried with full Mixup config.
39. **Cosine annealing LR + Mixup best config**: Run 025 tried cosine standalone (1.3109); never combined with Mixup.
40. **batch_size=128 + Mixup best config**: Smaller batches = more gradient steps, more diverse Mixup pairs per epoch.
41. **Mixup alpha=0.5 + modality_drop=0.15**: Combine alpha near-miss with modality_drop near-miss (0.15→1.2376).
42. **LR=2e-4 + Mixup best config**: Between current 1e-4 and tried 5e-4 (1.3074); untried with Mixup.
43. **Mixup alpha=0.5 + embedding noise std=0.005**: Very light noise added to the near-miss alpha.
44. **AdamW wd=5e-5 + Mixup best config**: Lighter than wd=1e-4 (1.2477); between Adam and tried wd values.
45. **Modality dropout p=0.12 + Mixup best config**: Between p=0.10 (best) and p=0.15 (1.2376).
46. **Mixup alpha=0.4 + [1024,512,256] + warmup 5 + cosine LR**: Add warm-up + cosine decay to best config.
47. **Mixup alpha=0.5 + batch_size=128**: Combine alpha near-miss with smaller batches.
48. **Label noise sigma=0.03 + Mixup best config**: Very light label noise on top of Mixup soft targets.

## Phase 6 hypotheses (runs 099+, after 98 experiments; best 1.2322; near-miss 1.2283 at alpha=0.45)

Key observations: 36 consecutive failed experiments since run 062. All Mixup variants (3-way, manifold, label-only, cross-modal, clamped lambda), activation changes, training regime changes, structural changes all regressed. Only alpha=0.45 gives a near-miss (1.2283, delta=0.0039). Model near its ceiling.

49. **Per-sample lambda Mixup**: Sample a different lambda for each sample in the batch (shape [B,1]) instead of one lambda per batch. Creates more diverse virtual samples.
50. **Drug-only Mixup**: Mix drug embeddings only (protein stays unchanged), with label mixing using drug's lambda. Isolates which modality benefits most from Mixup.
51. **Residual connections + Mixup best config**: Run 031 tried residuals without Mixup (1.3152). With Mixup, skip connections may prevent gradient degradation.
52. **Bilinear elementwise product fusion**: Add `drug_proj * prot_proj` (both projected to 512 dims) as extra input feature alongside drug||prot concatenation. Captures cross-modal interactions.
53. **InstanceNorm instead of BatchNorm in hidden layers**: IN normalizes each sample independently; avoids train/eval discrepancy in BN.
54. **No LayerNorm on inputs (raw embeddings)**: Remove drug_norm and prot_norm, feed raw embeddings. Tests whether LayerNorm is destroying useful scale information.
55. **Mixup alpha=0.45 + hidden dims [1024, 512, 512]**: Widen last hidden layer with best alpha.
56. **Separate modality Mixup lambdas**: Sample independent lam_drug and lam_prot from Beta(alpha,alpha), use same idx permutation but different mixing weights per modality.
57. **Dropout p=0.3 on last hidden layer only** (other layers keep p=0.2): Target regularization at the final representation.
58. **Drug-protein interaction dot product feature**: Compute drug_proj (→128) ⊙ prot_proj (→128) dot product scalar and append to MLP input.

## Phase 7 hypotheses (runs 109+, after 108 experiments; best 1.2322; near-miss 1.2283 at alpha=0.45)

Key observations: All 10 Phase 6 ideas failed (structural changes, cross-modal interaction features, InstanceNorm crash). 47 consecutive regressions since run 062. The alpha=0.45 near-miss (1.2283, delta=0.0039) remains the closest result ever. Drug-only Mixup (1.2699) confirms both modalities must be mixed. Residuals, bilinear fusion, dot-product features, separate lambdas all regressed. Model near ceiling; need weight-averaging and fine-grained alpha search.

59. **Mixup alpha=0.46**: Fine-tune between near-miss (0.45→1.2283) and 0.50 (1.2314). Interpolation between two near-misses.
60. **Mixup alpha=0.44**: Fine-tune between best (0.40→1.2322) and near-miss (0.45→1.2283). Closer to 0.45.
61. **Protein-only Mixup**: Mix protein embeddings only (drug unchanged), labels mixed with protein's lambda. Complement to drug-only (run 100: 1.2699).
62. **SWA (Stochastic Weight Averaging)**: After epoch 50, average model weights every 5 epochs using torch.optim.swa_utils. Use SWA model for final eval.
63. **R-Drop regularization**: Two forward passes per batch with different dropout masks; add KL divergence between two output distributions as auxiliary loss (lambda=1.0).
64. **FGM adversarial embedding training**: After normal backward, compute gradient of loss w.r.t. input embeddings, add perturbation ε=0.05*||emb||, re-compute loss, add to total loss.
65. **Protein-only modality dropout p=0.10 + Mixup best config**: Run 031 (protein-only modality drop without Mixup: 1.2858); never tried with Mixup config.
66. **Modality projection before concat: drug(768→512) + prot(1024→512)**: Project each modality to equal 512-dim before concat, MLP input becomes 1024 instead of 1792.
67. **Stochastic depth: skip each hidden layer with p=0.1**: Each hidden block independently skipped (identity) during training.
68. **alpha=0.45 + AdamW wd=1e-4**: Best alpha near-miss combined with slight weight decay (wd=1e-4 alone gave 1.2816, best wd result).

## Hard rules
- Edit `train.py` only. NEVER touch `prepare.py`. NEVER read the test set.
- ONE knob per experiment. If you need to change two things to test a hypothesis (e.g., SwiGLU requires changing the hidden block), make that explicit in the description but keep the change minimal.
- Fixed seed=42 always.
- Commit if val RMSE improves by ≥ 0.01. Otherwise `git reset --hard HEAD`.
- If a run crashes or returns NaN/inf, log the failure and reset.
- Append rows to `results.csv` using Python's `csv` module (or `pandas.to_csv` with `mode='a'`), never with raw f-string concatenation. This ensures fields containing commas are properly quoted.
- When all 12 priorities have been tried, propose new hypotheses (combinations of the best individual wins, or new directions) and append them to this file before continuing.
