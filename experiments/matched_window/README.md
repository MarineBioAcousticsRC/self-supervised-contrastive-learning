# Matched-window clustering (NIPS4Bplus)

Compares self-supervised contrastive animal2vec embeddings with BirdNET (3 s)
and Perch (5 s) on the same analysis windows.

Preferred protocol (Sec. III D): cluster animal2vec **vocal frames** in native
dimension (768-d, L2-normalized), then **majority-vote** frame cluster IDs to
one label per window. BirdNET (1024-d) and Perch (1536-d) remain one embedding
per window. MiniBatch k-means, `k` from 10 to 200 step 10, batch size 128.

## Experiments

| Exp | Windows | animal2vec | Baseline |
|-----|---------|------------|----------|
| 2 | `(0–3)`, `(2–5)` | cluster frames → majority vote | BirdNET |
| 3 | `(0–5)` | cluster frames → majority vote | Perch |

Windows with more than one class, or with no vocalization, are discarded.

## Run (from repository root)

Use separate environments for animal2vec, BirdNET, and Perch.

```bash
python -m experiments.matched_window.embed \
  --experiment all --model a2v \
  --wav-dir /path/to/nips4bplus \
  --out-dir /path/to/matched_window \
  --a2v-checkpoint /path/to/ssl_contrastive/checkpoint_last.pt

python -m experiments.matched_window.embed \
  --experiment 2 --model birdnet \
  --wav-dir /path/to/nips4bplus \
  --out-dir /path/to/matched_window

python -m experiments.matched_window.embed \
  --experiment 3 --model perch \
  --wav-dir /path/to/nips4bplus \
  --out-dir /path/to/matched_window

python -m experiments.matched_window.evaluate_vote \
  --out-dir /path/to/matched_window \
  --wav-dir /path/to/nips4bplus \
  --cluster-space native --vocal-frames-only --results-subdir vote_native
```

Paper curves in this repository: `results/vote_native/clustering_metrics.csv`.
Figures: `python scripts/plot_matched_window_baselines_paper.py`.
