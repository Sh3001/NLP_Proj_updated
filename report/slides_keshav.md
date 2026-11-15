# Slides: Address parsing and Classical baseline
### Keshav · issues #4, #5, #6 → hand to Akshat for the deck

Five slides: two for parsing, three for the classical baseline. Each slide has its content, the figure or table to
use, and a line of speaker notes. Full text: `report/section_parsing.md`, `report/section_classical_baseline.md`.

---

## SLIDE P1: Address parsing with a BiLSTM-CRF

**Title:** Parsing addresses into fields, without gold labels

```
003rd floor | platinum plaza plot no.24 pu-4 | behind c-21 mall | a.b. road | indore | mp
   HOUSE    |          LOCALITY              |     LANDMARK     |   STREET  |  CITY  | STATE

token ──► word emb (100) ⊕ char BiLSTM (2×32) ──► 2-layer BiLSTM (2×128) ──► CRF ──► Viterbi
```

- **No gold parses**, so silver labels come from rules on the commas: state at the end (42 spellings learned on
  train), city before it, house numbers first, landmark / street words.
- **Trick:** delete the commas from half of the training addresses. The model must find the field boundaries from the
  words; the rules can't.
- 2.72M parameters, 120k addresses, 1.4 min on a T4.

*Notes:* The CRF scores whole tag sequences with learned transitions (CITY → STATE likely, STATE → HOUSE rare), so it
doesn't produce impossible sequences.

---

## SLIDE P2: Without commas, the rules fail and the BiLSTM-CRF doesn't

**Title:** 85 % of comma-free addresses parsed exactly; rules: 0 %

| input | parser | token acc. | macro F1 | exact address |
|---|---|---|---|---|
| with commas | rules | 1.000 | 1.000 | 1.000 |
| with commas | BiLSTM-CRF | 0.999 | 0.999 | 0.997 |
| **commas removed** | rules | 0.576 | 0.369 | **0.000** |
| **commas removed** | **BiLSTM-CRF** | **0.950** | **0.906** | **0.855** |

`1915 Claybourne Avenue Salt Lake City Utah` → HOUSE `1915` · STREET `claybourne avenue` · CITY `salt lake city` · STATE `utah`

- Easy: STATE 0.999, HOUSE 0.979, CITY 0.972 F1. Hard: STREET vs LOCALITY, LANDMARK (0.64).
- Caveat: evaluated against the silver labels; every real address in our data has commas, so this is a stress test.

*Notes:* With commas the rules score 1.0 by construction (they make the labels). LANDMARK precision is low partly
because the model finds landmarks in mid-component (`… road infront of marketing society`) that the rules miss.

---

## SLIDE G1: Classical baseline: LightGBM on 48 pair features

**Title:** A CPU model within 0.0004 of a fine-tuned cross-encoder

| model | test F0.5 | params | train |
|---|---|---|---|
| mDeBERTa-v3 full FT (06f) | 0.9940 | 279M | 60 min GPU |
| bge full FT (06b) | 0.9912 | 278M | 19 min GPU |
| **LightGBM (04)** | **0.9908** | 1,609 trees | **4.6 min CPU** |
| bge LoRA (06c) | 0.9871 | 1.5M trained | 10 min GPU |
| Siamese BiLSTM (05a) | 0.9834 | 0.73M | 13 min GPU |

- Features per candidate pair: name and address edit-distance ratios (Indel, token-set, Jaro-Winkler), word Jaccard,
  **numbers only in one record**, legal-form conflict, parsed-field comparisons, 03's retrieval scores and ranks, and
  *gaps* to the record's best candidate.
- Third-best single model; 1.000 on the 283 "other Indic" records.

*Notes:* The `_gap` features matter because the decision is relative: does this S1 beat the other four candidates?

---

## SLIDE G2: What the GBM actually uses

**Title:** 83 % of the gain comes from the neural retriever

Figure: `analysis/out_gbm/gbm_feature_gain.png` (top 15 features and the share per family).

- `de_cos` alone, the fine-tuned e5 cosine from 03: **51 %**. All retrieval features: **83 %**.
- String / number features: 15 %, led by `num_onlyq` (house or shop numbers in the record but not in the candidate):
  the signal that separates two branches of one chain.
- Parsed-address fields: **0.6 %**.

*Notes:* So this is a classical learner on top of a neural retriever, not a pure edit-distance model. We didn't run it
without the retrieval features.

---

## SLIDE G3: Does the parser help matching? And the GBM in the fusion

**Title:** The parser doesn't change matching; the GBM anchors the fusion

| parsed fields | test F0.5 | 95 % bootstrap interval of the difference |
|---|---|---|
| BiLSTM-CRF (02) | 0.99081 | |
| rules | 0.99051 | **[−0.00006, +0.00064]**: not significant |

- Paired bootstrap over 21,529 test S1 entities; the runs differ on 273 entities (137 vs 136).
- Why: real addresses have commas, so both parsers give the same fields 99.7 % of the time.

| gate weight (07) | Latin | Devanagari | Malayalam | other Indic |
|---|---|---|---|---|
| GBM | 0.60 | 0.74 | 0.75 | 0.91 |
| mDeBERTa-v3 | 0.36 | 0.23 | 0.24 | 0.09 |

*Notes:* The parser's value is robustness on comma-free input, not this data. In the fusion, the GBM is the expert
most different from the cross-encoders, so the gate pairs it with the best one.
