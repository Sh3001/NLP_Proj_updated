# Plan: Ditto (Deep Entity Matching with Pre-Trained Language Models) in this project

**Paper:** Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, Wang-Chiew Tan.
*Deep Entity Matching with Pre-Trained Language Models.* PVLDB 14(1), 2020 (VLDB 2021).
Reference code: `megagonlabs/ditto` (Apache-2.0).

**Goal:** reproduce Ditto on our data (the `data/small` split, same candidates, same metric), measure
what each of its three optimisations adds, and use it as one more expert in the MoE fusion and the
demo. This gives the project a published method to compare against, which addresses the "is this
really NLP?" question directly.

---

## 1. What Ditto is

Ditto treats entity matching as **sequence-pair classification** with a fine-tuned pre-trained
language model:

1. **Serialization.** Each record becomes text with attribute markers:
   `[COL] name [VAL] Balaji Care Pvt Ltd [COL] address [VAL] H.No 110, Narela Shankari, Bhopal`.
   A pair is `serialize(a) [SEP] serialize(b)`. The `[CLS]` vector goes to a linear layer that outputs match / no match.
2. **Three optimisations** on top:
   - **DK, domain knowledge injection:** *span typing* (mark important spans with their type, e.g.
     numbers, product IDs) and *span normalisation* (rewrite equivalent spans the same way).
   - **SU, summarisation:** long records are cut to the max sequence length by keeping the tokens with
     the highest TF-IDF, instead of truncating the end.
   - **DA, data augmentation with MixDA:** make an augmented copy of each training pair
     (`span_del`, `span_shuffle`, `attr_del`, `attr_shuffle`, `entry_swap`). Encode both, and train on
     an interpolation of the two `[CLS]` vectors, λ·E(x) + (1−λ)·E(x_aug), with λ ~ Beta(α, α).
     This keeps the label correct even when the augmentation breaks it.
3. **Blocking** happens before Ditto; Ditto only classifies candidate pairs.

Paper results: up to +29 % F1 over the previous state of the art (DeepMatcher). With half the labels
it matches DeepMatcher trained on all of them.

## 2. How it maps onto our project

| Ditto component | What we already have | What we add |
|---|---|---|
| Blocking | `03_candidates` (TF-IDF + fine-tuned e5, RRF, top 5) | nothing: Ditto scores the same pairs as every other matcher |
| Serialization | `ce_text` = `name / romanised name \| address` | `[COL]/[VAL]` serializer with attributes `name`, `address` (+ DK attributes) |
| DK: span typing | **BiLSTM-CRF parser (02)** tags HOUSE, STREET, LOCALITY, LANDMARK, CITY, STATE, POSTCODE | typed attributes `[COL] house [VAL] 110 [COL] city [VAL] bhopal …`; numbers marked |
| DK: span normalisation | NFKC, romanisation, **learned transliteration lexicon**, abbreviation expansion (03) | apply these to the serialized values; normalise house numbers (`h.no. 110` → `110`) |
| SU: summarisation | IDF over the serialized words, fitted on train | per-attribute TF-IDF summariser down to `MAX_LEN` tokens |
| DA + MixDA | nothing | the 5 augmentation operators + MixDA in the training loop |
| Language model | bge-reranker-base (XLM-R), IndicBERTv2, mDeBERTa (06b–f) | **xlm-roberta-base** (the multilingual RoBERTa, the closest match to Ditto's default) and bge-reranker-base |
| Evaluation | macro F0.5 per S1, threshold tuned on val, test scored once | the same, plus **pair-level F1** on the candidate pairs (Ditto's metric, so we can compare with the paper) |

Our domain knowledge is stronger than the paper's generic regexes: the BiLSTM-CRF parser **is** a span
typer, and the transliteration lexicon **is** span normalisation for Indic scripts. That's the
project's own contribution on top of the reproduction.

## 3. Experiments

All runs use the same candidate pairs from 03, train on part A, tune the threshold on val, and score
test once.

### 3a. Ablation (what each optimisation adds), 40k training pairs each

| Run | Serialization | DK | SU | DA (MixDA) | Backbone | Question it answers |
|---|---|---|---|---|---|---|
| R0 | plain `ce_text` | – | – | – | bge-reranker-base | = 06b, already run (the reference) |
| R1 | `[COL]/[VAL]` | – | – | – | xlm-roberta-base | Ditto baseline: does serialization alone help? |
| R2 | `[COL]/[VAL]` | ✓ | – | – | xlm-roberta-base | + parser-typed spans and normalisation |
| R3 | `[COL]/[VAL]` | – | ✓ | – | xlm-roberta-base | + summarisation (matters most for long Indic records) |
| R4 | `[COL]/[VAL]` | – | – | ✓ | xlm-roberta-base | + MixDA |
| R5 | `[COL]/[VAL]` | ✓ | ✓ | ✓ | xlm-roberta-base | **full Ditto** |
| R6 | `[COL]/[VAL]` | ✓ | ✓ | ✓ | bge-reranker-base | full Ditto from a reranker checkpoint |

Augmentation-operator study (R4 variants, 20k pairs): each operator alone, plus a mix of all five.

### 3b. Label efficiency (the paper's second claim)

Train R1 (no DA) and R5 (full) on **2k / 10k / 40k** pairs. Plot F0.5 against training pairs.
Does DA close the gap at low resource, as the paper claims?

### 3c. Final model

The best configuration from 3a, trained on **160k pairs** (the same budget as 06b). It saves scores for
B / val / test and its weights, so it joins **07** (MoE fusion) as an expert and **08** (demo) as a
live model.

### 3d. Analysis
* Per-script F0.5 (Latin / Devanagari / Malayalam): does DK with transliteration help most on Indic records?
* Error categories (transliteration, abbreviation, word order, missing tokens, sibling branch), compared with 06b.
* Token-length distribution before and after summarisation; how often `MAX_LEN` cut the text.
* MoE gate weight given to Ditto by script (from 07).

### GPU budget (T4, estimates)

| Block | Runs | Time |
|---|---|---|
| Ablation R1–R6 @ 40k | 6 | ~1.5–2 h (MixDA runs ~1.6× slower) |
| Operator study @ 20k | 6 | ~1 h |
| Label efficiency | 6 | ~1 h |
| Final @ 160k | 1 | ~50–70 min |
| **Total** | | **~6 h** including the val / test scoring of every run, inside the 30 h weekly Kaggle quota |

## 4. Implementation (done: `kaggle/build_kaggle.py` → `kaggle/notebooks/06g_ditto.ipynb`)

The code is in `DITTO_SRC` (shared with the demo) and in the `06g_ditto` cells.

| Piece | Where | Notes |
|---|---|---|
| `attrs_of(ix, dk, summ)`, `serialize(attrs, mark_num)` | `DITTO_SRC` | plain = name + address as in 06b; DK = normalised name + the parser's typed spans, house numbers normalised, numbers marked `[NUM]` |
| `Summarizer` | `DITTO_SRC` | IDF from 100k **train** records; keeps the highest-IDF words up to `DITTO_SU_WORDS` per record; every attribute keeps its best word |
| `augment(A, B, op, rng)` | `DITTO_SRC` | `span_del`, `span_shuffle`, `attr_del`, `attr_shuffle`, `entry_swap`, `all` |
| `DittoModel` | `DITTO_SRC` | LM + linear on `[CLS]`; `[COL] [VAL] [NUM]` added as special tokens, initialised at the mean embedding; MixDA mixes the `[CLS]` vectors of the original and the augmented pair (lam ~ Beta(0.8, 0.8), made >= 0.5) in one forward pass |
| `run_ditto(cfg)` | `06g_ditto` | trains on whole candidate groups of part A, tunes the threshold on val, scores test once; writes `results/` with pair F1 and per-script F0.5 |
| run sets | `06g_ditto` | `ablation`, `operators`, `label_eff`, `final`, `all`; `final` picks the best ablation run by val F0.5 |
| fusion, demo | 07, 08 | the `final` run saves scores and weights, so 07 treats it as one more expert; the demo scores it live and shows what Ditto reads |

Choices that differ from the paper (state them in the report):
* Binary cross-entropy on one logit instead of a 2-class softmax (equivalent), so Ditto's scores are logits like every other matcher's.
* `DITTO_LEN = 96`, the same as 06b, so R0 is a fair reference. Summarisation is a per-record word budget (`DITTO_SU_WORDS = 16`), not a per-pair token budget. `ditto_lengths.csv` shows how often truncation happens with and without SU.
* Small training sets are repeated until there are at least `DITTO_MIN_STEPS = 600` steps, as the paper trains small sets for more epochs.
* Backbone: `xlm-roberta-base` (R1–R5) and `bge-reranker-base` (R6); `DITTO_FINAL` picks among them by val F0.5.

Tested locally with a tiny stand-in model: all 17 runs, the summary cells, the saved final model, and the demo scoring it live (matches the stored scores to 0.0012 logits). No real result exists yet.

## 5. Owner

This part is implemented and run by Shivam alone (the earlier five-way split is dropped). The other four
members' issues are unchanged.

## 6. Order and timeline

| Phase | Work | Depends on |
|---|---|---|
| 1. Code | `DITTO_SRC`, `06g_ditto`, 08 changes; local smoke test | **done** |
| 2. Inputs | `er-02-output` and `er-03-output` from the current notebooks (issues #1, #4) | already planned |
| 3. Ablation | R1–R6, operator study, label efficiency (in parallel over 5 accounts) | 1, 2 |
| 4. Final | best configuration @ 160k → scores + weights | 3 |
| 5. Fusion + demo | rerun 07 with Ditto attached; 08 shows Ditto live | 4, and the other model runs |
| 6. Write-up | report section "Ditto on multilingual business records", slides | 3–5 |

## 7. Deliverables and "done when"

- [ ] `06g_ditto` notebook, generated and smoke-tested; `DITTO_SRC` shared with 08
- [ ] Ablation table R0–R6 with macro F0.5, pair F1, P, R, per-script F0.5, training time
- [ ] Operator-study table and label-efficiency plot
- [ ] Final Ditto model in 07's results table and as a live model in the 08 demo
- [ ] Report section (method, adaptation to our data, ablation, label efficiency, errors) and 3–4 slides
- [ ] Related work: DeepMatcher (Mudgal et al., 2018), Ditto (Li et al., 2020), Sentence-BERT (Reimers & Gurevych, 2019)

## 8. Risks

| Risk | Mitigation |
|---|---|
| Ditto ≈ 06b (both fine-tune an XLM-R cross-encoder), so gains may be small | That is still a result. The ablation shows which optimisation helps, and per-script analysis shows *where* |
| DK attributes make inputs longer → slower, more truncation | SU exists exactly for this; `DITTO_LEN` = 96 for all runs, truncation measured in `ditto_lengths.csv` |
| MixDA needs two forward passes (≈1.6× training time) | Ablation at 40k pairs; only the final run uses 160k |
| New `[COL]`/`[VAL]` tokens start untrained | Initialise from the mean embedding; compare with plain-text markers in R1 |
| Leakage | Summariser IDF, lexicon and parser are all fitted on train only; threshold on val; test scored once per run |
| fp16 instability (as with mDeBERTa) | xlm-roberta-base and bge are stable in fp16 on T4 (06b already uses it) |
