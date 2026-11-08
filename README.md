# NLP course project: Business Entity Resolution

Team: Akshat, Keshav, Rishi, Shivam, Ayush. The work split is in [TEAM.md](TEAM.md); the tasks are tracked as GitHub issues.

The official test set has no labels, so every result in this project is measured on splits
cut from the **labelled training data**.

## 1. Data splits (`src/make_splits.py`)

```bash
cd nlp_project
python3 src/make_splits.py --data-dir ../dataset/train --out data/full     # ~7 min, ~5 GB RAM peak
```

How the split is made:

* **Unit = Source-1 entity**, 70 / 10 / 20 % (train / val / test), seed 42.
* **Stratified** by country × number of true matches (0 / 1–3 / 4+). Every split has the same
  country mix and the same share of singletons (5.6 %).
* **Linked S2/S3 records follow their S1.** In the data each S2/S3 record is linked to at most
  one S1 (7,638,365 links, 0 shared). So no true match crosses a split boundary.
* **Unlinked S2/S3 records** (distractors, 26 % of the pool) are spread over the splits with the
  same 70/10/20 proportions, using a hash of their id. Every split keeps the original distractor density.
* Files are copied **line for line**, in the competition format. Any pipeline that reads
  `<dir>/train/train_*.tsv` and `<dir>/test/test_*.tsv` can run on `data/full` unchanged.
* The script re-reads its output and checks it: no id in two splits, GT rows = S1 rows, every GT
  id present in its split's pool.

```
data/full/
├── train/        train_source{1,2,3}.tsv  train_ground_truth.tsv
├── val/          val_source{1,2,3}.tsv    val_ground_truth.tsv      <- tuning / early stopping
├── test/         test_source{1,2,3}.tsv                             <- no labels, like the real test
├── test_labels/  test_ground_truth.tsv                              <- only for the final evaluation
├── s1_splits.tsv     entity_id, country, n_true, split
└── manifest.json     settings, counts, verification
```

| split | S1 | singletons | true links | S2 | S3 | unlinked pool |
|---|---|---|---|---|---|---|
| train | 1,544,775 | 86,273 | 5,346,174 | 3,524,824 | 3,699,027 | 1,877,677 (26.0 %) |
| val | 220,683 | 12,325 | 763,903 | 503,137 | 528,612 | 267,846 (26.0 %) |
| test | 441,363 | 24,649 | 1,528,288 | 1,006,655 | 1,057,964 | 536,331 (26.0 %) |

### Small regional subset (`data/small`, 74 MB) for the LSTM, transformer and MoE experiments

```bash
python3 src/make_splits.py --data-dir ../dataset/train --out data/small \
    --states "US:OR,US:UT,India:Kerala,India:Madhya Pradesh"            # ~11 min
python3 src/make_splits.py --data-dir ../dataset/train --list-states   # available states
```

This subset keeps **whole states** instead of a random sample. The near-duplicate businesses
of one region stay together, so the task stays as hard as on the full data.

* States are spelled differently in each source (TX / Texas, MH / Maharashtra). The script learns
  the mapping from raw address component to state from the ground-truth links
  (81 canonical states, 15,095 spellings).
* Records with no detectable state are kept at the country's subset rate, so the share of
  addresses with a missing state stays realistic.
* Same 70/10/20 stratified split and the same checks as the full split.

| split | S1 (US / India) | singletons | true links | S2 | S3 | unlinked pool |
|---|---|---|---|---|---|---|
| train | 75,353 (36,238 / 39,115) | 4,169 | 261,183 | 173,057 | 182,320 | 94,194 (26.5 %) |
| val | 10,764 (5,176 / 5,588) | 595 | 37,236 | 24,896 | 25,856 | 13,516 (26.6 %) |
| test | 21,529 (10,353 / 11,176) | 1,191 | 74,619 | 49,479 | 52,095 | 26,955 (26.5 %) |

Rules for the experiments:

* Fit **everything** on `train/`: models, TF-IDF vocabularies, transliteration dictionaries,
  normalisation rules, thresholds learned from data.
* Tune hyper-parameters and the decision threshold on `val/`.
* Score `test/` **once per final model**, with `test_labels/`.

## 2. Evaluation (`src/evaluate.py`)

This is the competition metric: macro F0.5 per Source-1 entity, with singletons included.

```bash
python3 src/evaluate.py --gt data/full/test_labels/test_ground_truth.tsv \
    --pred <run>/matching_results.tsv --candidates <run>/candidate_pairs.tsv \
    --s1 data/full/test/test_source1.tsv --json <run>/test_metrics.json
```

It prints macro F0.5, macro and micro precision/recall, exact-set accuracy and singleton
accuracy, with a per-country breakdown. With `--candidates` it also prints blocking pair recall,
candidates per S1, and the blocking ceiling (the best F0.5 the matcher could reach on those candidates).

Sanity checks on the test split:

| prediction | macro F0.5 |
|---|---|
| ground truth itself | 1.00000 |
| everything empty | 0.05585 (= singleton share) |
| last true id of every S1 dropped | 0.87114 (P 1.0, R 0.66) |

## 3. Kaggle notebooks (one per method)

See [kaggle/README.md](kaggle/README.md). Upload `data/small.zip` as a Kaggle Dataset, then run
`kaggle/notebooks/01 … 07` in order. `07_moe_fusion_and_results` builds the final results table.
`08_demo` is the live demo: one record followed through every step (see the Kaggle README).

Splits use BLAKE2b hashing for every random choice. (CRC32 was used at first, but CRC hashes of the
same id with different seeds are correlated, so two "independent" choices were not independent.)
