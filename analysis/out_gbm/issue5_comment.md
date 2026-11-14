04 is done: two full runs (`QUICK_RUN = False`, CPU), one with `er-02-output` attached and one without it (rule parser).

**Output (public Kaggle dataset)**
- `er-04-output` (the run with 02): https://www.kaggle.com/datasets/xenoncipher/er-04-output

The zip has `results/`, `scores/` (B/val/test logits for 07), `preds/`, `models/gbm.txt` + `gbm_meta.json` (for 08) and `gbm_feature_importance.csv`.

**Setup.** LightGBM on 48 features per candidate pair from 03: string similarities of name and address (rapidfuzz ratios, Jaro-Winkler, token Jaccard, numbers), parsed-address field comparisons, script / source flags, and 03's retrieval scores (TF-IDF and fine-tuned e5 cosine and ranks, RRF rank). Trained on part A (1.51M pairs), threshold tuned on val, test scored once.

### Results (held-out test split, threshold tuned on val)

| parsed fields from | threshold | val F0.5 | **test F0.5** | test P | test R | F0.5 India | F0.5 US | singleton acc | trees | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| BiLSTM-CRF (02) | 0.36 | 0.99286 | **0.99081** | 0.9938 | 0.9873 | 0.99205 | 0.98946 | 0.9857 | 1609 | 4.6 |
| rules (no 02) | 0.33 | 0.99275 | **0.99051** | 0.9931 | 0.9879 | 0.99191 | 0.98901 | 0.9832 | 1722 | 4.7 |

### Is the BiLSTM-CRF worth it? No, not for matching on this data.

- **The difference is within noise.** The BiLSTM-CRF fields give +0.0003 test F0.5. A paired bootstrap over the 21,529 test S1 entities (`analysis/paired_bootstrap.py`, 10,000 resamples) gives a 95% interval of **[−0.00006, +0.00064]**, which includes 0. The two runs differ on only 273 entities, 137 better with 02 and 136 better without.
- **Why:** every non-empty address in er-small has its commas, and with commas the BiLSTM-CRF reproduces the rule parse almost exactly (token accuracy 0.9994, exact address 0.9969, see #4). The two runs therefore get nearly the same fields. The BiLSTM-CRF only helps on comma-free addresses, which this data doesn't have.
- **And the parsed fields matter little anyway:** the 7 parsed-field features (`house_ratio`, `locality_tset`, `street_tset`, `city_ratio`, `house_eq`, `post_eq`, `q_landmark`) together get **0.64%** of the total gain in both runs. The best of them, `house_ratio`, is rank 21. `post_eq` has no gain at all (er-small has no postcodes), and `q_landmark` has almost none.

The parser is still useful as a robustness measure and for the 08 demo, which shows the fields. Its value would show up on data where addresses come without punctuation.

### Top 20 features by gain (run with 02; the rules run has the same order)

| # | feature | share of gain | | # | feature | share of gain |
|---|---|---|---|---|---|---|
| 1 | `de_cos` | 51.2% | | 11 | `num_jac` | 0.8% |
| 2 | `de_rank` | 13.6% | | 12 | `num_onlys` | 0.8% |
| 3 | `de_cos_gap` | 9.9% | | 13 | `n_partial` | 0.7% |
| 4 | `rrf_rank` | 5.3% | | 14 | `q_nlen` | 0.6% |
| 5 | `num_onlyq` | 4.6% | | 15 | `n_jac` | 0.6% |
| 6 | `n_extra_q` | 2.2% | | 16 | `core_ratio` | 0.4% |
| 7 | `tf_cos_gap` | 1.3% | | 17 | `n_ratio_gap` | 0.4% |
| 8 | `tf_rank` | 1.1% | | 18 | `n_jw` | 0.4% |
| 9 | `n_tsort` | 1.0% | | 19 | `a_partial` | 0.4% |
| 10 | `n_ratio` | 0.9% | | 20 | `n_tset` | 0.3% |

(`q` = the S2/S3 record being matched, `s` = the S1 candidate; `n_` = name, `a_` = address, `num_` = the numbers in the record.)

**The top 5:**
1. **`de_cos`** (51%): cosine similarity of the two records under 03's fine-tuned multilingual-e5 bi-encoder. It was trained on exactly this matching task, so it already captures most of the signal, including across scripts (a Devanagari or Malayalam name against its Latin spelling), where string ratios fail.
2. **`de_rank`** (14%): the rank of this S1 among the dense neighbours of the query. The rank is relative: a cosine of 0.85 means a match if it is the best candidate and a non-match if three others are higher.
3. **`de_cos_gap`** (10%): `de_cos` minus the best `de_cos` among this query's candidates (0 for the best one). It tells the model how far a candidate is behind the leader, which separates a clear winner from a near-tie between branches of the same chain.
4. **`rrf_rank`** (5%): the rank after 03's reciprocal-rank fusion of TF-IDF and dense retrieval. It agrees with `de_rank` most of the time, so it adds the cases where TF-IDF (character n-grams) and the bi-encoder disagree.
5. **`num_onlyq`** (5%): how many numbers in the query record (house number, plot, door, shop number) do **not** appear in the candidate. This is the strongest hard *negative* signal: two branches of the same business have near-identical names but different street numbers, and the embeddings blur exactly that difference.

**Caveat for the report:** 80% of the gain comes from 03's fine-tuned e5 retrieval features (`de_*`, `rrf_rank`); 83% with the TF-IDF ones. So this is a "classical model on top of a neural retriever", not a purely edit-distance baseline. By feature family, the gain splits as follows:
- retrieval scores and ranks: 82.8%
- string / edit-distance / number features (`n_*`, `a_*`, `num_*`, `core_*`, `first_num_eq`, `legal_conflict`): 15.4%
- record metadata (lengths, script, source): 1.2%
- parsed-address fields: 0.6%
