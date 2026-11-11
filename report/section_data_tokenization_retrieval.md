# Report sections: Data and splits, Tokenization, Candidate generation

**Author:** Akshat (issues #1, #2, #3) · **Notebooks:** `01_tokenization`, `03_candidates` · **Scripts:** `src/make_splits.py`, `src/evaluate.py`
**Course topics:** tokenization, noisy channel, n-grams, word vectors (count-based vs predictive), retriever–reader QA, coreference (framing)

Tables: `results/01/tokenizer_stats.csv`, `results/01/tokenization_blocking.csv`; blocking numbers from the `er-03-output` Kaggle dataset (`blocking.json`).

---

## D. Data and splits

### D.1 Task

The data come from the Amazon ML Challenge 2026 entity-resolution task. Source 1 (S1) is a reference list of
businesses. Sources 2 and 3 (S2, S3) describe businesses from the same regions with noise: abbreviated legal
forms (`private limited` / `pvt ltd`), romanised Indian spellings (`praivtt`, `enjiniyring`), names written in
Devanagari or Malayalam, missing addresses, and near-duplicate branches of one chain that differ only in a house
number. Each S2/S3 record is linked to at most one S1 entity; 26 % of S2/S3 records are linked to none.

Every model in the project uses the same **decision rule**: each S2/S3 record is assigned to its best-scoring S1
candidate if the score is at least a threshold, and to no S1 otherwise. This is the mention-ranking formulation of
coreference resolution with a dummy antecedent: each record either picks its best antecedent or starts a new
cluster.

### D.2 Metric

The competition metric is **macro F0.5 per S1 entity**. For every S1 entity we compare the set of S2/S3 records
predicted for it with its true set, compute F0.5, and average over all S1 entities. β = 0.5 weighs precision more than
recall, so a false link costs more than a missed one. Singletons (S1 entities with no true match) are included: they
score 1 only if nothing is linked to them. `src/evaluate.py` implements the metric and was checked on the test split:

| prediction | macro F0.5 |
|---|---|
| ground truth itself | 1.00000 |
| everything empty | 0.05585 (= singleton share) |
| last true id of every S1 dropped | 0.87114 (P 1.0, R 0.66) |

### D.3 Splits

The official test set has no labels, so all results come from splits of the labelled training data
(`src/make_splits.py`, seed 42).

* **Unit = S1 entity**, 70 / 10 / 20 % (train / val / test). Linked S2/S3 records follow their S1, so no true match
  crosses a split boundary and no business is seen in both training and test.
* **Stratified** by country × number of true matches (0 / 1–3 / 4+): every split has the same country mix and the
  same singleton share (5.6 %).
* **Unlinked S2/S3 records** (distractors) are spread with the same proportions using a hash of their id, so every
  split keeps the original distractor density.
* All random choices use BLAKE2b hashes. (CRC32 was tried first, but CRC hashes of one id under different seeds are
  correlated, so two "independent" choices were not independent.)
* The script re-reads its output and checks it: no id in two splits, ground-truth rows = S1 rows, every ground-truth
  id present in its split's pool.

**Regional subset `data/small`.** Training on the full data (2.2M S1 entities) is beyond our compute (free Kaggle
T4 GPUs), so all experiments use four whole states: Oregon, Utah, Kerala and Madhya Pradesh. Keeping whole states
instead of a random sample keeps a region's near-duplicate businesses together, so the matching task stays as hard as
on the full data. States are spelled differently in each source (TX / Texas, MH / Maharashtra); the script learns the
mapping from address components to states from the ground-truth links (81 canonical states, 15,095 spellings).

| split | S1 (US / India) | singletons | true links | S2 | S3 | unlinked pool |
|---|---|---|---|---|---|---|
| train | 75,353 (36,238 / 39,115) | 4,169 | 261,183 | 173,057 | 182,320 | 94,194 (26.5 %) |
| val | 10,764 (5,176 / 5,588) | 595 | 37,236 | 24,896 | 25,856 | 13,516 (26.6 %) |
| test | 21,529 (10,353 / 11,176) | 1,191 | 74,619 | 49,479 | 52,095 | 26,955 (26.5 %) |

**Protocol.** Everything (models, TF-IDF vocabularies, the transliteration lexicon, normalisation rules) is fitted on
train. Hyper-parameters and decision thresholds are tuned on val. Test is scored once per final model; its labels are
read only inside the final scoring function.

### D.4 Limitations of the splits

* **Distractor density.** Our splits keep the training data's density of unlinked records, about 1.2 per S1. On the
  official test set we measured about 2.3 per S1 during the challenge. F0.5 on our splits is therefore optimistic,
  mostly in precision.
* **Subset.** `data/small` holds about 3.5 % of the S1 entities. The ranking of methods should carry over; the
  absolute numbers are not those of a full-data model.
* **Challenge score.** Our challenge pipeline (public leaderboard 0.9895) used the full data and a different
  validation scheme. It is the project's starting point and is not comparable with the numbers in this report.

---

## T. Tokenization (`01_tokenization`)

**Question.** How do different tokenizers split business names and addresses in each script, and which tokenization
and normalisation give the best retrieval?

**Groups.** Val records split by country and script: US Latin, India Latin, Devanagari, Malayalam, and the
Devanagari and Malayalam records again after our romaniser. Fertility = subword tokens per whitespace word.

### T.1 How tokenizers split each script

| tokens per word | US Latin | India Latin | Devanagari | Devanagari, romanised | Malayalam | Malayalam, romanised |
|---|---|---|---|---|---|---|
| whitespace | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| char 3-gram | 6.34 | 6.68 | 6.82 | 6.52 | 7.63 | 6.74 |
| mBERT (WordPiece) | 2.06 | 2.06 | **2.99** | 2.08 | **3.90** | 2.47 |
| XLM-R (SentencePiece) | 2.05 | 1.94 | 2.17 | 2.06 | 2.82 | 2.36 |
| mDeBERTa-v3 (SentencePiece) | 1.95 | 1.87 | 2.45 | 2.15 | 2.81 | 2.41 |
| IndicBERTv2 (WordPiece) | 2.04 | 1.77 | **1.54** | 1.92 | **1.71** | 2.32 |

| share of words split into > 1 piece | US Latin | India Latin | Devanagari | Devanagari, romanised | Malayalam | Malayalam, romanised |
|---|---|---|---|---|---|---|
| mBERT | 0.50 | 0.50 | 0.71 | 0.60 | 0.77 | 0.74 |
| XLM-R | 0.57 | 0.53 | 0.59 | 0.60 | 0.74 | 0.73 |
| mDeBERTa-v3 | 0.57 | 0.52 | 0.72 | 0.68 | 0.76 | 0.72 |
| IndicBERTv2 | 0.50 | 0.36 | 0.32 | 0.52 | 0.37 | 0.66 |

UNK rate is 0 for every tokenizer and group, except mDeBERTa-v3 on Malayalam (0.002).

**Example** (`आदित्य प्रॉपर्टीज एलएलपी`, normalised `aditya properties llp`):

| tokenizer | pieces | tokens |
|---|---|---|
| IndicBERTv2 | 5 | `आदित्य` `प्रॉपर्टी` `##ज` `एल` `##एलपी` |
| XLM-R | 10 | `▁आदि` `त्य` `▁प्र` `ॉ` `पर्` `टी` `ज` `▁एल` `एल` `पी` |
| mDeBERTa-v3 | 10 | `▁आद` `ित्य` `▁प्र` `ॉ` `प` `र्ट` `ीज` `▁एल` `एल` `पी` |
| mBERT | 13 | `आदि` `##त्य` `प` `##्र` `##ॉ` `##पर` `##्टी` `##ज` `ए` `##ल` `##ए` `##ल` `##पी` |

A Latin example for contrast, `Shree Infracon Private Ltd`: IndicBERTv2 `Shree` `Infra` `##con` `Private` `Ltd`
(5); mDeBERTa-v3 `▁Shre` `e` `▁Infra` `con` `▁Private` `▁Ltd` (6); XLM-R 7; mBERT `S` `##hre` `##e` `Inf` `##rac` `##on`
`Private` `Ltd` (8).

### T.2 Tokenization and normalisation for retrieval

TF-IDF retrieval (cosine, same country), recall of the true S1 among the top k on val. "Non-Latin" = S2/S3 records
whose text is not in Latin script.

| analyzer | text | R@1 | R@5 | R@10 | R@1 non-Latin | vocabulary | seconds |
|---|---|---|---|---|---|---|---|
| words | raw | 0.9729 | 0.9895 | 0.9927 | 0.9536 | 59,722 | 10.6 |
| words | romanised | 0.9752 | 0.9907 | 0.9937 | 0.9592 | 58,939 | 11.0 |
| words | romanised + abbrev. | 0.9749 | 0.9906 | 0.9936 | 0.9581 | 58,920 | 10.8 |
| words | romanised + abbrev. + lexicon | 0.9806 | 0.9930 | 0.9951 | 0.9913 | 58,707 | 10.7 |
| char 3-grams | not romanised (+ abbrev.) | 0.9844 | 0.9945 | 0.9963 | 0.9583 | 22,042 | 26.5 |
| char 3-grams | romanised + abbrev. | 0.9878 | 0.9960 | 0.9975 | 0.9768 | 20,093 | 27.2 |
| **char 3-grams** | **romanised + abbrev. + lexicon** | **0.9910** | **0.9973** | **0.9983** | **0.9952** | 20,086 | 27.2 |
| char 2–4-grams | romanised + abbrev. + lexicon | 0.9912 | 0.9972 | 0.9983 | 0.9952 | 113,651 | 66.5 |
| XLM-R subwords | raw | 0.9830 | 0.9940 | 0.9962 | 0.9599 | 16,903 | 14.7 |
| XLM-R subwords | romanised + abbrev. + lexicon | 0.9894 | 0.9968 | 0.9979 | 0.9931 | 15,403 | 12.1 |

The **transliteration lexicon** (316 words) is learned from the training links (part A only). For each linked pair
whose non-Latin S2/S3 name, once romanised, has as many words as its S1 name, the words are aligned by position. A
romanised word becomes an entry if it was aligned with the same English word at least 3 times and in at least 60 % of
its occurrences (`praivtt → private`, `limittd → limited`, `mainejmemt → management`, `knstrksns → constructions`).
It is applied only to records that are not pure ASCII.

### T.3 Findings

1. **mBERT fragments Indic scripts the most**: 3.90 pieces per Malayalam word and 2.99 per Devanagari word, against
   2.06 on Latin text. It falls back to single characters and even splits the virama (`##्र`) into its own token.
   IndicBERTv2, trained on Indian languages, has the lowest fertility on both native scripts (1.54 and 1.71), lower
   than any tokenizer on Latin text.
2. **The cost of a foreign script is fragmentation, not unknown tokens.** On this data no tokenizer produces more
   than 0.2 % [UNK] tokens; unfamiliar words are broken into small pieces instead.
3. **Romanisation is a trade-off.** It brings mBERT from 3.90 to 2.47 pieces per Malayalam word, but makes IndicBERTv2
   worse (1.71 → 2.32): romanised Indian words are in no tokenizer's vocabulary. This is why the cross-encoders get
   both the native name and its romanisation (`name / romanised name | address`).
4. **The learned lexicon matters more than the tokenizer.** On non-Latin records the lexicon alone adds +0.033 R@1
   with words (0.958 → 0.991) and +0.018 with character 3-grams (0.977 → 0.995). With it, the three analyzers end within
   0.004 of each other. Abbreviation expansion adds nothing measurable.
5. **Character 3-grams are the best TF-IDF analyzer overall** (R@1 0.9910). Char 2–4-grams tie (0.9912) at 5.7× the
   vocabulary and 2.4× the time, so `03_candidates` uses 3-grams. XLM-R subwords are close (0.9894) with the smallest
   vocabulary.
6. **Good tokenization does not make a good matcher.** IndicBERTv2 tokenizes Indic text best, but as a fine-tuned
   cross-encoder it is the weakest fully trained transformer (test F0.5 0.9761, see the Indic-backbone section), even on
   Devanagari and Malayalam.

**Noisy-channel view.** A romanised record is an S1 name sent through a noisy channel: transliterated to Latin
letters, then spelled phonetically and abbreviated. Romanisation inverts the first step and the lexicon the most
common errors of the second, which is why both are needed (non-Latin R@1 0.958 → 0.977 → 0.995 for character
3-grams).

---

## C. Candidate generation (`03_candidates`)

**Why blocking.** Scoring every test S2/S3 record (about 100k) against every test S1 entity of its country
(about 11k) is about 10⁹ pairs. Blocking keeps the 5 most likely S1 entities per record, so every matcher scores the same
507,870 test pairs.

**Retrievers.** All search is restricted to S1 entities of the same country.

* **TF-IDF over character 3-grams** of the normalised text (romanised, abbreviations expanded, lexicon applied):
  the count-based representation chosen in the tokenization study.
* **multilingual-e5-small** (118M parameters), zero-shot: mean-pooled, L2-normalised embeddings.
* **e5-small, fine-tuned**: 100k linked (S2/S3, S1) pairs from train part A, symmetric in-batch contrastive loss
  (cross-entropy over the similarity matrix in both directions), 3.5 min on a T4. This is the recipe of DPR's
  retriever.
* **Fusion**: reciprocal rank fusion, score = Σ 1 / (60 + rank) over the TF-IDF and fine-tuned e5 lists; keep the
  top 5.

**Results (val).**

| retriever | R@1 | R@5 | R@10 | training |
|---|---|---|---|---|
| TF-IDF char 3-gram | 0.9912 | 0.9972 | 0.9983 | – |
| e5-small zero-shot | 0.9883 | 0.9954 | 0.9968 | – |
| e5-small fine-tuned | **0.9940** | **0.9988** | **0.9994** | 3.5 min |
| fused (RRF), top 5 | | **0.9989** | | |

* **Blocking ceiling**, the best macro F0.5 a perfect matcher could reach on these candidates: val 0.99974,
  **test 0.99957**.
* 5.0 candidates per record. Pairs: train 1,776,885 (part A 1,509,720 / part B 267,165), val 253,760,
  test 507,870. About 14.7 % of pairs are positive.

**Findings.**

1. **Zero-shot dense retrieval loses to character n-grams** (R@1 0.9883 vs 0.9912). A general-purpose embedding
   model has not learned that two branches with similar names are different businesses, and character n-grams
   match spelling variants directly.
2. **A 3.5-minute contrastive fine-tune puts the dense retriever ahead at every k** (R@1 0.9940). Count-based vs
   predictive representations: the predictive one wins only once it is trained on the task.
3. **Fusion adds little on top** (R@5 0.9988 → 0.9989), but makes the candidate set robust to either retriever's
   misses.
4. **Recall is not the bottleneck.** With a ceiling of 0.9996, almost all of the remaining error of the matchers
   (best test F0.5 0.9948) comes from scoring, not from blocking.
5. **The retriever carries the classical model.** In the LightGBM baseline, the fine-tuned e5 cosine is the most
   important feature (51 % of gain) and the retrieval features together give 83 % (see the classical baseline section).

**QA view.** The pipeline has the shape of open-domain QA: a cheap bi-encoder retriever narrows the ~11,000 S1 entities
of a record's country to 5, and an expensive cross-encoder reader scores only those 5.
