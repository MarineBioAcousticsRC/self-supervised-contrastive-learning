# Self-supervised contrastive learning for bioacoustics

Implementation of **self-supervised contrastive learning** for adapting a
pretrained animal2vec encoder using only vocalization presence/absence timing
(no species or call-type labels during training). Representations are evaluated
by MiniBatch *k*-means clustering, with *k* chosen by the inertia elbow.

Checkpoints (`.pt`) and frame embeddings (`.h5`) are not included.

## Method

A pretrained animal2vec checkpoint is loaded. Frame embeddings from the same
vocalization interval are treated as positives; embeddings from a non-vocal
region of the same recording or from another recording in the minibatch are
treated as negatives. Training uses a triplet-margin loss on cosine distance
(Table I).

| Piece | Path |
|---|---|
| Sampling and triplet loss | `nn/contrastive.py` |
| Training | `scripts/ssl_contrastive_finetune.py` |
| Hyperparameters (Table I) | `configs/ssl_contrastive.yaml` |
| animal2vec encoder (load checkpoint) | `nn/data2vec2.py` |
| Embed recordings | `animal2vec_inference.py` |
| MiniBatch *k*-means sweep | `scripts/kmeans_sweep.py` |
| Inertia elbow | `scripts/analyze_k_selection_metrics.py` |
| BirdNET / Perch comparison | `experiments/matched_window/` |

**Self-supervised contrastive learning:** run `ssl_contrastive_finetune.py` with no extra flag.  
**Supervised contrastive learning** (labels used only to reject same-class negatives): add `--supervised-contrastive`.

```
pretrained animal2vec .pt
        │
        ▼
scripts/ssl_contrastive_finetune.py
        │
        ▼
checkpoint_last.pt
        │
        ▼
animal2vec_inference.py
        │
        ▼
scripts/kmeans_sweep.py  →  inertia elbow selects k
```

Run Python from this repository root so `import nn` resolves.

## Requirements

- NIPS4Bplus wavs and temporal CSVs
- Fairseq manifest (`train_0.tsv`; first line is the data root)
- Pretrained animal2vec `.pt` (xeno-canto)

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/fairseq.git@920a548ca770fb1a951f7f4289b4d3a0c1bc226f
python scripts/test_ssl_contrastive.py
```

## Self-supervised contrastive finetuning

```bash
python scripts/ssl_contrastive_finetune.py \
  --pretrain-ckpt /path/to/pretrained_animal2vec.pt \
  --manifest-dir /path/to/nips4bplus/manifest \
  --train-subset train_0 \
  --label-csv-dir /path/to/nips4bplus \
  --label-csv-format auto \
  --label-index-rate-hz 44100 \
  --save-dir /path/to/self_supervised_contrastive \
  --batch-size 8 \
  --max-updates 5000 \
  --lr 1e-5 \
  --device cuda
```

Supervised contrastive control: add `--supervised-contrastive`.

## Embeddings

```bash
python animal2vec_inference.py \
  --path /path/to/nips4bplus \
  --model-path /path/to/self_supervised_contrastive/checkpoint_last.pt \
  --out-path /path/to/embeddings/self_supervised_contrastive \
  --write-embeddings True \
  --unique-values "[]" \
  --sample-rate 24000 \
  --average_top_k_layers 12
```

Self-supervised pretraining baseline: set `--model-path` to the pretrained animal2vec checkpoint.

## Clustering and inertia elbow

```bash
python scripts/kmeans_sweep.py \
  --emb-dir /path/to/embeddings/self_supervised_contrastive \
  --wav-csv-dir /path/to/nips4bplus \
  --use-csv-labels \
  --k-min 10 --k-max 200 --k-step 10 \
  --out-csv /path/to/kmeans_sweep.csv

python scripts/analyze_k_selection_metrics.py \
  --csv /path/to/kmeans_sweep.csv \
  --out-dir /path/to/k_selection
```

Operating *k* is the inertia elbow (largest perpendicular distance of the
normalized WCSS curve to the chord joining the first and last sweep points).

Four representations (Sec. III A) at their elbow *k*:

| Representation | Elbow *k* |
|---|---|
| Self-supervised pretraining | 120 |
| Supervised finetuning | 40 |
| Self-supervised contrastive learning | 60 |
| Supervised contrastive learning | 70 |

Sweep used for Table II: `results/kmeans_sweep/four_representations/kmeans_sweep.csv` (184,603 vocal frames, 88 classes).  
`results/kmeans_sweep/four_representations_validation/` is a validation split and is not Table II.

## BirdNET and Perch

See `experiments/matched_window/README.md`. Native-dimension vote metrics:
`results/vote_native/clustering_metrics.csv`.

## Figures from CSVs (no GPU)

```bash
python scripts/plot_inertia_elbow_paper.py
python scripts/plot_matched_window_baselines_paper.py
```

UMAP (needs embeddings on disk):

```bash
python scripts/plot_umap_pretrain_vs_ssl_contrastive_subset.py \
  --pretrain-dir /path/to/embeddings/self_supervised_pretrain \
  --contrastive-dir /path/to/embeddings/self_supervised_contrastive \
  --wav-csv-dir /path/to/nips4bplus \
  --species-list data/nips4b_birdchallenge_espece_list.csv \
  --out-dir /path/to/umap_out \
  --n-classes 8 --ignore-test-files
```

## Layout

```
nn/contrastive.py
scripts/ssl_contrastive_finetune.py
animal2vec_inference.py
experiments/matched_window/
configs/ssl_contrastive.yaml
results/self_supervised_contrastive/     # training logs
results/supervised_contrastive/
results/kmeans_sweep/
results/vote_native/
paper/figures/
data/nips4b_birdchallenge_espece_list.csv
```

## Data

NIPS4Bplus transcriptions: https://figshare.com/articles/dataset/Transcriptions_of_NIPS4B_2013_Bird_Challenge_Training_Dataset/6798548

## License

MIT (see `LICENSE`). The `nn/` encoder is from animal2vec.
