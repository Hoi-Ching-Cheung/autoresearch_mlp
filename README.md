# AutoResearch MLP — Drug-Target Affinity Prediction

Structured experiment loop for tuning an MLP that predicts drug-target
binding affinity (pKd) from pre-computed drug and protein embeddings.

## Task

Regression on a drug-target interaction dataset. Input: concatenated drug
embedding (768-d) + protein embedding (1024-d). Output: pKd binding affinity.
Metric: validation RMSE.

embedding (768-d) + protein embedding (1024-d). Output: pKd binding
affinity. Metric: validation RMSE.

## Approach
Experiments are run one hyperparameter change at a time against a locked
baseline, with results logged to `results.csv`. Hypotheses are generated
iteratively based on what worked, organized into research phases:

- **Baseline**: val RMSE ≈ 1.358
- **Best so far**: val RMSE ≈ 1.232 (Mixup α=0.4 + hidden [1024, 512, 256] +
 modality dropout p=0.10)

Key wins: modality dropout, Mixup augmentation, and architectural depth.
100+ experiments across 7 phases.

## Files

| File | Description |
|------|-------------|
| `train.py` | Mutable experiment file — hyperparameters live at the top |
| `prepare.py` | Data loading, preprocessing, evaluation |
| `results.csv` | Log of all runs (val RMSE, train RMSE, notes) |
| `program.md` | Research agenda and hypothesis backlog |

## Usage

```bash
pip install torch numpy
python train.py

Prints VAL_RMSE=<float> and TRAIN_RMSE=<float> on completion.

Course

Harvard ACOMP 209b, Spring 2026
