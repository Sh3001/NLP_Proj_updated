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

### Limitations to state in the report

* **Distractor density.** Our splits keep the training data's density of unlinked records (≈ 1.2 per S1).
  During the challenge we measured ≈ 2.3 per S1 on the official test set, so F0.5 on these splits is
  optimistic compared with the leaderboard. Precision suffers most at higher density.
* **Challenge vs course numbers.** The challenge pipeline (public LB 0.9895) used the full data and a
  different validation scheme. Do not put it in the same table as the course results; present it as the
  starting point, and the course splits as the controlled comparison.
* **Subset.** `data/small` is about 3.5 % of the S1 entities (4 states). The ranking of methods should
  carry over, but the absolute numbers are not those of a full-data model.

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

See [kaggle/README.md](kaggle/README.md). The data and the candidate pairs are already on Kaggle, so
there is nothing to build or upload:

| Dataset | Link |
|---|---|
| `er-small`: the `data/small` splits | https://www.kaggle.com/datasets/akshatiit/er-small |
| `er-03-output`: output of `03_candidates` (candidate pairs, blocking numbers, fine-tuned e5) | https://www.kaggle.com/datasets/akshatiit/er-03-output |

Attach them with **Add Input → Datasets**, then run `kaggle/notebooks/01 … 07` in order. `07_moe_fusion_and_results` builds the final results table.
`06g_ditto` reproduces Ditto (Li et al., VLDB 2021) as a DK / SU / DA ablation ([docs/DITTO_PLAN.md](docs/DITTO_PLAN.md)).
`08_demo` is the live demo: one record followed through every step (see the Kaggle README).

### Executed Kaggle Notebooks & Outputs

| Notebook | Topic & Method | Kaggle Notebook Link | Test $F_{0.5}$ | Owner |
|---|---|---|---|---|
| `06d_ce_bge_moe_head` | MoE head inside BGE cross-encoder (top-2 of 4) | [ayush122222/06d](https://www.kaggle.com/code/ayush122222/notebook7ebbf31256) | **0.98888** | Ayush |
| `07_moe_fusion_and_results` | Gated Mixture-of-Experts fusion & final results | [ayush122222/07_moe](https://www.kaggle.com/code/ayush122222/notebook2a4366b080) | **0.99481** | Ayush |

### Final Results Summary (`data/small`, Held-out Test Split)

| Section | Method | Val $F_{0.5}$ | Test $F_{0.5}$ | Test $P$ | Test $R$ | Test $F_{0.5}$ (US) | Test $F_{0.5}$ (India) | Singleton Acc |
|---|---|---|---|---|---|---|---|---|
| **5 Classical** | LightGBM baseline | 0.99286 | 0.99081 | 0.9938 | 0.9873 | 0.98946 | 0.99205 | 0.9857 |
| **6 RNN / LSTM** | Siamese char-BiLSTM | 0.98920 | 0.98335 | 0.9863 | 0.9846 | 0.98390 | 0.98284 | 0.9454 |
| **6 RNN / LSTM** | Siamese char-BiGRU | 0.98814 | 0.98197 | 0.9850 | 0.9835 | 0.98142 | 0.98247 | 0.9404 |
| **6 RNN / LSTM** | Siamese char-BiRNN (vanilla) | 0.98374 | 0.97475 | 0.9782 | 0.9789 | 0.97646 | 0.97316 | 0.9085 |
| **7 Transformers** | CE mDeBERTa-v3 full fine-tune (fp32) | 0.99561 | 0.99402 | 0.9953 | 0.9931 | 0.99261 | 0.99533 | 0.9866 |
| **7 Transformers** | CE bge-reranker-base full fine-tune | 0.99338 | 0.99124 | 0.9941 | 0.9878 | 0.98935 | 0.99299 | 0.9840 |
| **7 Transformers** | CE bge-reranker-base + MoE head | 0.99226 | **0.98888** | 0.9914 | 0.9871 | 0.98719 | 0.99044 | 0.9706 |
| **7 Transformers** | CE bge-reranker-base LoRA ($r=16$) | 0.99142 | 0.98708 | 0.9900 | 0.9854 | 0.98566 | 0.98839 | 0.9681 |
| **7 Transformers** | CE IndicBERTv2 full fine-tune | 0.98249 | 0.97611 | 0.9841 | 0.9666 | 0.97268 | 0.97929 | 0.9446 |
| **7 Transformers** | CE bge-reranker-base zero-shot | 0.94192 | 0.91674 | 0.9161 | 0.9595 | 0.92713 | 0.90712 | 0.6868 |
| **7b Ditto** | Ditto bge DK+SU+DA-all 40k | 0.98713 | 0.98214 | 0.9856 | 0.9813 | 0.98158 | 0.98266 | 0.9379 |
| **8 Fusion** | **Gated Mixture of Experts** | **0.99648** | **0.99481** | **0.9961** | **0.9935** | **0.99304** | **0.99645** | **0.9933** |
| **8 Fusion** | Logistic-regression stacking | 0.99648 | 0.99477 | 0.9967 | 0.9920 | 0.99322 | 0.99620 | 0.9933 |
| **8 Fusion** | MoE, constant gate (ablation) | 0.99642 | 0.99474 | 0.9963 | 0.9928 | 0.99299 | 0.99636 | 0.9933 |
| **8 Fusion** | LightGBM stacking | 0.99632 | 0.99470 | 0.9968 | 0.9918 | 0.99305 | 0.99622 | 0.9924 |
| **8 Fusion** | Mean of 9 calibrated experts | 0.99525 | 0.99286 | 0.9941 | 0.9927 | 0.99204 | 0.99363 | 0.9857 |

Splits use BLAKE2b hashing for every random choice. (CRC32 was used at first, but CRC hashes of the
same id with different seeds are correlated, so two "independent" choices were not independent.)
