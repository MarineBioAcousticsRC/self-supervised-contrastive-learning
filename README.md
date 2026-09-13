# Self-supervised contrastive learning for bioacoustics

Finetune a pretrained animal2vec encoder using only vocalization onset/offset
(no species or call-type labels). Frame embeddings from the same vocalization
are positives; a non-vocal region of the same recording or a vocal region from
another recording in the minibatch are negatives (50-50). Training uses a
triplet-margin loss on cosine distance (Table I).

Checkpoints (`.pt`) and frame embeddings (`.h5`) are not included. Paper
figures are in `paper/figures/`.

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

NIPS4Bplus transcriptions:
https://figshare.com/articles/dataset/Transcriptions_of_NIPS4B_2013_Bird_Challenge_Training_Dataset/6798548

## Run contrastive finetuning

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

**Self-supervised contrastive learning** (the proposed method):

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

Hyperparameters match Table I (`configs/ssl_contrastive.yaml`).

**Supervised contrastive control** (labels used only to reject same-class
negatives from other recordings): add `--supervised-contrastive`.

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

Self-supervised pretraining baseline: set `--model-path` to the pretrained
animal2vec checkpoint.

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

Operating *k* is the inertia elbow: largest perpendicular distance of the
normalized WCSS curve to the chord joining the first and last sweep points.

## N independent trials (Table II)

Repeats the pipeline over N seeds. Pretrain and supervised finetune reuse
frozen checkpoints/embeddings and only re-run clustering + elbow. The two
contrastive methods retrain, embed, then elbow-select *k* each trial.

```bash
python scripts/run_n_trials.py \
  --out-root /path/to/multi_trial_out \
  --n-trials 31 \
  --base-seed 42 \
  --pretrain-ckpt /path/to/pretrained_animal2vec.pt \
  --manifest-dir /path/to/nips4bplus/manifest \
  --label-csv-dir /path/to/nips4bplus \
  --wav-dir /path/to/nips4bplus \
  --wav-csv-dir /path/to/nips4bplus \
  --emb-ssl-pretrain /path/to/embeddings/self_supervised_pretrain \
  --emb-supervised-finetune /path/to/embeddings/supervised_finetune
```

Mean ± sample std tables are written under `--out-root/aggregate/`.

Paper Table II numbers (31 trials, 184,603 vocal frames, 88 classes):

| Embedding | Elbow *k* | ARI | AMI |
|---|---|---|---|
| Self-supervised pretraining | 71.3 ± 43.6 | 0.098 ± 0.030 | 0.244 ± 0.043 |
| Self-supervised contrastive | 61.3 ± 8.1 | 0.231 ± 0.023 | 0.474 ± 0.012 |
| Supervised finetuning | 55.2 ± 7.2 | 0.484 ± 0.018 | 0.715 ± 0.007 |
| Supervised contrastive | 61.0 ± 7.5 | 0.239 ± 0.026 | 0.485 ± 0.013 |

## Paper figures

| Paper | File |
|---|---|
| Fig. 2 (inertia elbow, four embeddings) | `paper/figures/fig_elbow_four_models.pdf` |
| Fig. 3 (mean ARI / AMI vs *k*, N=31) | `paper/figures/fig_ari_ami_vs_k_n31.pdf` |
| Fig. 4 (AMI vs BirdNET / Perch) | `paper/figures/fig_ami_matched_windows.pdf` |
| Fig. 5 (ARI vs BirdNET / Perch) | `paper/figures/fig_ari_matched_windows.pdf` |
| Fig. 6 (UMAP, 8 random call classes) | `paper/figures/fig_umap_ssl_pretrain_vs_contrastive_random_calls.pdf` |

PNG and SVG copies sit next to the PDFs. Fig. 1 is a method schematic, not
produced by these scripts.

Rebuild Figs. 2–5 from CSVs already in this repository (no GPU):

```bash
python scripts/plot_inertia_elbow_paper.py
python scripts/plot_ari_ami_vs_k_n31.py
python scripts/plot_matched_window_baselines_paper.py
```

Fig. 2 source: `results/kmeans_sweep/four_representations/kmeans_sweep.csv`  
Fig. 3 source: `results/kmeans_sweep/n31_vs_k/`  
Figs. 4–5 source: `paper/figures/matched_window_vote_metrics.csv`

BirdNET / Perch re-run (needs those models): see
`experiments/matched_window/README.md`.

UMAP (Fig. 6) needs embeddings on disk:

```bash
python scripts/plot_umap_pretrain_vs_ssl_contrastive_subset.py \
  --pretrain-dir /path/to/embeddings/self_supervised_pretrain \
  --contrastive-dir /path/to/embeddings/self_supervised_contrastive \
  --wav-csv-dir /path/to/nips4bplus \
  --species-list data/nips4b_birdchallenge_espece_list.csv \
  --out-dir /path/to/umap_out \
  --n-classes 8 \
  --class-selection random \
  --vocal-type call \
  --legend-names scientific \
  --ignore-test-files
```

## Layout

```
nn/contrastive.py                         # sampling + triplet loss
scripts/ssl_contrastive_finetune.py       # training
scripts/run_n_trials.py                   # N-seed Table II pipeline
animal2vec_inference.py
experiments/matched_window/               # BirdNET / Perch
configs/ssl_contrastive.yaml              # Table I
results/kmeans_sweep/                     # elbow + N=31 CSVs
results/vote_native/                      # matched-window metrics
paper/figures/                            # Figs. 2–6
data/nips4b_birdchallenge_espece_list.csv
```

## License

MIT (see `LICENSE`).
