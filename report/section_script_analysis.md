# Report section: Script-specific analysis

**Author:** Rishi (issues #8, #9) · **Notebook run:** `06e_ce_indicbert` · **Analysis:** `analysis/per_script.py`
**Course topics:** fine-tuning, multilingual / Indic pre-training, error analysis

Figure: `analysis/out_script/per_script_f05.png`.
Tables: `analysis/out_script/per_script_f05.md`, `per_script_errors.md`, `ranking_per_script.md`,
`indic_vs_bge.md`. Reproduce with `python3 analysis/per_script.py <folder with prep/, test_labels/ and the er-*-output folders>`.

> Covers 9 matchers: 04, 05a–c, 06a, 06b, 06c, 06e, 06f. 06d (MoE head) and the 07 fusion rows will be
> added when their outputs are shared; the method does not change.

---

## S.1 Setup

All S1 records (the reference registry) are in Latin script. The S2/S3 records to be linked come in
several scripts. We group every test S2/S3 record by **country × script** of its name and address:

| group | test records | linked | unlinked |
|---|---|---|---|
| US Latin | 48,710 | 35,788 | 12,922 |
| India Latin | 34,342 | 25,488 | 8,854 |
| India Devanagari | 7,868 | 5,703 | 2,165 |
| India Malayalam | 10,371 | 7,374 | 2,997 |
| India other Indic (Gujarati, Telugu, Tamil, Kannada, Bengali, Odia, Gurmukhi) | 283 | 266 | 17 |

In a "Devanagari" or "Malayalam" record, the native script is not only the state name: about half have
the **business name** itself in that script (Devanagari 48 %, Malayalam 54 %), and ≈60 % have the
address in it.

**Metric.** Macro F0.5 per S1 entity, as in the competition, restricted to the links of that group's
records. Each model's decisions are taken unchanged from its `preds/` file (threshold tuned on val).
As a check, the overall F0.5 recomputed this way matches every model's `results/*.json`. We also
report threshold-free ranking quality (pair AUC and top-1 accuracy), and count each error as one of:
*wrong match* (linked to the wrong S1), *false match* (linked, but the record has no S1),
*missed* (true S1 not linked), or *blocking miss* (true S1 never among the candidates).

**What each model sees.** The char-RNNs (05a–c) and the GBM (04) read the **romanised** text only.
The cross-encoders (06a–f) read the original name next to its romanisation (`ce_text`), so the
non-Latin characters reach their subword tokenisers.

## S.2 Per-script F0.5 for every model

| model | test F0.5 | US Latin | India Latin | Devanagari | Malayalam | other Indic (283) |
|---|---|---|---|---|---|---|
| GBM (04) | 0.9908 | 0.9889 | 0.9860 | 0.9924 | 0.9909 | 1.0000 |
| Siamese BiLSTM (05a) | 0.9834 | 0.9830 | 0.9754 | 0.9831 | 0.9781 | 0.9048 |
| Siamese BiGRU (05b) | 0.9820 | 0.9804 | 0.9752 | 0.9812 | 0.9779 | 0.8961 |
| Siamese BiRNN (05c) | 0.9748 | 0.9752 | 0.9641 | 0.9694 | 0.9700 | 0.9090 |
| bge zero-shot (06a) | 0.9167 | 0.9242 | 0.8944 | 0.8826 | 0.8857 | 0.9615 |
| bge full fine-tune (06b) | 0.9912 | 0.9887 | 0.9878 | 0.9935 | 0.9915 | 0.9441 |
| bge LoRA r=16 (06c) | 0.9871 | 0.9849 | 0.9824 | 0.9889 | 0.9862 | 0.9578 |
| IndicBERTv2 full fine-tune (06e) | 0.9761 | 0.9712 | 0.9684 | 0.9728 | 0.9778 | 0.9715 |
| **mDeBERTa-v3 full fine-tune (06f)** | **0.9940** | **0.9922** | **0.9901** | **0.9969** | **0.9970** | 0.9815 |

**1. Script is not what makes a record hard.** For every trained model, **India Latin is the hardest
of the four main groups**, and Devanagari and Malayalam score *higher* than both Latin groups for every
fine-tuned cross-encoder. Once romanisation and fine-tuning are in place, a record written in
Devanagari is no harder to link than one in Latin. The difficulty lies in the records themselves:
near-duplicate branches, missing addresses and generic names (S.4). India Latin also has the most
blocking misses (57).

**2. Fine-tuning is what closes the script gap.** Zero-shot bge (06a) is the one model that is
weakest on native scripts (Devanagari 0.883, Malayalam 0.886, against 0.924 for US Latin). Its error
rate there is 10–11 % of records, mostly false matches. After full fine-tuning (06b), the same backbone
reaches 0.994 / 0.992, a gain of +0.11 F0.5 on Devanagari.

**3. The romanised-only models fail on rare scripts.** On the 283 other-Indic records, the char-RNNs
drop to 0.90–0.91, and every error is a missed link. The generic romanisation of Telugu, Gujarati or
Bengali gives spellings far from the English S1 name (`iunibharsal sphtoy yar`,
`emtrpraijes`), and the learned transliteration lexicon only knows words seen in ≥ 3 aligned training
pairs. The cross-encoders see the original script and stay at 0.94–0.98. The GBM is the exception
(1.000) thanks to its address and number features, but 283 records (17 unlinked) are too few for
strong claims.

**4. Errors are matching errors, not retrieval errors.** Blocking misses are identical for every model
(35 US, 57 India Latin, 2 Devanagari, 2 Malayalam, 0 other Indic) and are a small share of each model's
errors. Ranking quality is high everywhere (pair AUC ≥ 0.998 for every fine-tuned model on every main
group), so most remaining errors come from the decision threshold on near-duplicates.

## S.3 IndicBERTv2 vs bge-reranker-base

The question for 06e: does an encoder pre-trained specifically on Indian languages
(`ai4bharat/IndicBERTv2-MLM-only`, 278M, Indian languages + English, its own WordPiece vocabulary)
beat a general multilingual cross-encoder (`BAAI/bge-reranker-base`, 278M) on the Indic records? Both
use **the same recipe**: full fine-tuning, 1 epoch over the same 160k part-A candidate pairs, batch 32,
BCE loss, same decision rule. Only the backbone differs, plus the learning rate (3e-5 for IndicBERTv2,
2e-5 for bge).

| India records | n | F0.5 IndicBERTv2 | F0.5 bge | only Indic right | only bge right | AUC Indic | AUC bge | McNemar p |
|---|---|---|---|---|---|---|---|---|
| Latin | 34,342 | 0.9684 | **0.9878** | 173 | 775 | 0.99897 | 0.99983 | 2e-91 |
| Devanagari | 7,868 | 0.9728 | **0.9935** | 27 | 139 | 0.99937 | 0.99996 | 2e-19 |
| Malayalam | 10,371 | 0.9778 | **0.9915** | 35 | 149 | 0.99963 | 0.99996 | 6e-18 |
| other Indic | 283 | **0.9715** | 0.9441 | 7 | 6 | 0.99878 | 0.99607 | 1.0 |

"Only X right" counts records that one model gets right (correct S1, or correctly left unlinked) and the
other gets wrong. McNemar's exact test is run on those discordant pairs.

**The Indic-specific backbone does not help, even on Indic scripts.** bge is significantly better on
Latin, Devanagari and Malayalam, both in F0.5 and in threshold-free AUC. IndicBERTv2 is ahead only on
the 283 other-Indic records, and there the difference is 7 vs 6 records, not significant. IndicBERTv2
also loses most on US Latin (1,192 missed links vs 436 for bge), so its deficit is not about script.
It has the lowest recall of any fine-tuned model (R 0.967 at P 0.984).

**Likely reasons.**
- *Pre-training objective.* bge-reranker-base was already trained as a **cross-encoder for relevance**
  on query/passage pairs, so fine-tuning only adapts an existing pairwise head. IndicBERTv2 is an
  MLM-only encoder; its `[CLS]` representation and the classification head start from scratch for a
  pairwise task.
- *What the task needs.* Matching is decided by names, house numbers and locality tokens that are
  mostly Latin or romanised on at least one side (S1 is always Latin). Deep knowledge of Indic
  morphology adds little; robust pairwise comparison adds a lot.
- *A typical IndicBERTv2 mistake* (example 4 below): it links a record to a different company at the
  **same address**, while every other model follows the identical company name.

The ranking across all cross-encoders (mDeBERTa-v3 > bge full > bge LoRA > IndicBERTv2 > bge zero-shot)
is the same in every main script group. The backbone's general pairwise ability matters more than its
language coverage.

## S.4 Five typical errors

| # | S2/S3 record | prediction | truth | models wrong | why |
|---|---|---|---|---|---|
| 1 | `Eye Care Associates of Hermiston Inc`, ##1139 Hartley Ave, Hermiston OR | *Eye Care Associates of Hermiston Inc.*, 1135 Hartley Avenue | no match | 9 / 9 | **Hard negative**: same name, house number off by 4. Every model reads it as a typo. |
| 2 | `The Dent Seafood Corporation`, no address | *The Dent Seafood*, Hubbard OR | *The Dent Seafood*, Portland OR | 9 / 9 | **Sibling records without an address**: two S1 records share the name, so nothing can choose. |
| 3 | `Tech അഗ്രോ പ്രൈവറ്റ് ലിമിറ്റഡ്` (Latin + Malayalam), no address | 7 models link a wrong S1; 06e and 06f abstain | *Tech Agro Pvt Ltd*, Calicut | 9 / 9 | **Blocking miss + mixed script**: retrieval never offered the true S1. |
| 4 | `SREE SYSTEMS PRIVATE LIMITED`, 436/A Machamthode, Mannarkad, കേരളം | 06e: *Star Logistics Limited*, same address | *Sree Systems Private Limited*, same address | 1 / 9 (06e) | **IndicBERTv2 over-weights the address**: same building, different company. |
| 5 | `ಸೂರ್ಯ ಇನ್‌ಫ್ರಾಸ್ಟ್ರಕ್ಚರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` (Kannada), Bangalore | 06e: no match | *Surya Infrastructure Private Limited*, Bangalore | 1 / 9 (06e) | **Transliteration**: the name matches only after Kannada → Latin. 8 models link it; IndicBERTv2's score stays below its threshold. |

The errors that **every** model makes (1–3) come from the data: deliberate near-duplicates, sibling
branches with no distinguishing field, and records retrieval never paired with their S1. None of them
is script-specific. The script-specific errors (4–5) belong to a single model.

## S.5 Summary

- **Script is not the bottleneck.** After romanisation and fine-tuning, Devanagari and Malayalam records
  are linked *more* accurately than Latin ones by every fine-tuned cross-encoder. India Latin is the
  hardest group.
- **The Indic-specific backbone loses.** IndicBERTv2 is significantly worse than bge-reranker-base on
  Latin, Devanagari and Malayalam (McNemar p < 1e-17), and only ties on rare scripts. A
  relevance-pre-trained cross-encoder transfers better than Indic MLM pre-training.
- **Romanised-only models break on rare scripts.** The char-RNNs lose ≈0.08 F0.5 on the other-Indic
  records, where romanisation and the learned lexicon fall short; models that also see the original
  script do not.
- **Remaining errors are shared, data-driven cases** (hard negatives, address-less siblings, blocking
  misses) that no single matcher resolves. That motivates fusing the matchers (section *Mixture of
  Experts*).
