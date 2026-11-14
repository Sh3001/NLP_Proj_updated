# Slides — Mixture of Experts and Results
## NLP Course Project: Business Entity Resolution
### Ayush · Issue #15 → hand to Akshat & Keshav for final deck

---

## SLIDE 1 — Mixture of Experts Architecture

### What problem does MoE solve?

No single matcher is best across all languages and data qualities:
- **mDeBERTa-v3** is the best single matcher on every script
- **LightGBM** (string + parsed-address features) is close behind and makes different errors
- **IndicBERTv2** loses to the multilingual bge cross-encoder even on Indian scripts (McNemar p < 1e-17)

**MoE learns how much to trust each expert, per pair.**

---

### Two MoE components in this project

```
┌─────────────────────────────────────────────────────────────┐
│  Component 1 · MoE HEAD on the cross-encoder (notebook 06d) │
│                                                             │
│  bge-reranker-base → [CLS] → Gate (Linear) → softmax       │
│                               ↓                             │
│                        top-2 of 4 experts selected          │
│                        (Switch-style load-balancing loss)   │
│                               ↓                             │
│                        Weighted sum of expert outputs        │
│                               ↓   logit                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Component 2 · GATED FUSION over all matchers (notebook 07) │
│                                                             │
│  Expert logits: z₁…z_N (from 04–06, N = #matchers)         │
│                                                             │
│  Gate inputs (11 features, never sees country label):       │
│    script one-hot (4), is-S3, addr-empty, addr-has-num,     │
│    name-len/40, addr-len/80, TF-IDF cos, e5 cos,           │
│    RRF rank/K, #candidates/K                                │
│                                                             │
│  Gate:  x → Linear(13,32) → ReLU → Linear(32,N) → softmax  │
│         → weights w(x) ∈ Δ^N                               │
│                                                             │
│  Score: p = σ(Σᵢ wᵢ(x)·(aᵢ zᵢ + bᵢ) + c)                 │
│         aᵢ, bᵢ per-expert affine calibration               │
└─────────────────────────────────────────────────────────────┘
```

---

### Training protocol (Gated MoE, notebook 07)

| Setting | Value |
|---------|-------|
| Training data | Part B (15% of train pairs, unseen by matchers) |
| Optimiser | Adam, lr=1e-2, weight-decay=1e-5 |
| Schedule | Cosine annealing |
| Batch size | 2048 pairs |
| Loss | Binary cross-entropy |
| Init | aᵢ, bᵢ from individual logistic fits (starts at mean-of-experts) |

---

### Why gating works without seeing the country label

The gate sees **script** (a proxy for language) and **retrieval quality**.
This lets it generalise to unseen countries:

```
Learned gate weight (mean over test pairs, moe_gate_by_script.csv)

              LightGBM   mDeBERTa-v3   other 7 experts
Latin           0.60        0.36          < 0.02 each
Devanagari      0.74        0.23          < 0.02 each
Malayalam       0.75        0.24          < 0.01 each
other Indic     0.91        0.09          < 0.01 each
```

---

## SLIDE 2 — Experimental Results

### Dataset: `data/small` (4 states, 2 countries)

| Split | S1 entities | True links | Distractor density |
|-------|------------|-----------|-------------------|
| train | 75,353 | 261,183 | 26.5% |
| val | 10,764 | 37,236 | 26.6% |
| **test** | **21,529** | **74,619** | **26.5%** |

**Metric:** Macro F0.5 per S1 (precision-weighted; singletons included)  
**Threshold:** Tuned on val; test scored once.

---

### Results table (fill from `final_results.csv` after Kaggle run)

| Method | Test F0.5 | Test P | Test R | US | India |
|--------|-----------|--------|--------|----|-------|
| LightGBM baseline | 0.9908 | 0.9938 | 0.9873 | 0.9895 | 0.9921 |
| Siamese BiLSTM | 0.9834 | 0.9863 | 0.9846 | 0.9839 | 0.9828 |
| Siamese BiGRU | 0.9820 | 0.9850 | 0.9835 | 0.9814 | 0.9825 |
| CE bge zero-shot | 0.9167 | 0.9161 | 0.9595 | 0.9271 | 0.9071 |
| CE bge full FT | 0.9912 | 0.9941 | 0.9878 | 0.9894 | 0.9930 |
| CE bge LoRA r=16 | 0.9871 | 0.9900 | 0.9854 | 0.9857 | 0.9884 |
| CE bge + MoE head | 0.9889 | 0.9914 | 0.9871 | 0.9872 | 0.9904 |
| CE IndicBERTv2 | 0.9761 | 0.9841 | 0.9666 | 0.9727 | 0.9793 |
| CE mDeBERTa-v3 | 0.9940 | 0.9953 | 0.9931 | 0.9926 | 0.9953 |
| Mean of 9 experts | 0.9929 | 0.9941 | 0.9927 | 0.9920 | 0.9936 |
| LR stacking | 0.9948 | 0.9967 | 0.9920 | 0.9932 | 0.9962 |
| **Gated MoE** | **0.9948** | **0.9961** | **0.9935** | **0.9930** | **0.9965** |
| Constant-gate ablation | 0.9947 | 0.9963 | 0.9928 | 0.9930 | 0.9964 |

*Test split of `data/small`, threshold tuned on val. Fusion rows: the final 9-expert run of 07
(04, 05a, 05b, 06a–f; `results/*.json`).*

---

### Key observations

1. **LightGBM is a surprisingly strong baseline** — address feature engineering
   captures structure that raw string similarity misses.

2. **Multilingual beats Indic-only** — mDeBERTa-v3 is the best single model on
   every script; IndicBERTv2 is below the bge cross-encoder even on Devanagari
   and Malayalam (McNemar p < 1e-17).

3. **Gated MoE beats every single matcher** (+0.0008 over mDeBERTa, +0.0020
   over the plain mean). The gate never sees the country, so it can route an
   unseen country by its script.

4. **But the input-conditional gate is not what helps** — the constant-gate
   ablation (0.9947) and LR stacking (0.9948) are within 0.0001 of it. The gain
   comes from learned, calibrated weights over two complementary experts
   (LightGBM + mDeBERTa); the gate only shifts weight toward LightGBM on Indic scripts.

5. **Distractor density caveat:** Our splits have ≈1.2 distractors / S1;
   the leaderboard has ≈2.3. F0.5 values on our test are optimistic.

---

### Four demo records summary

| # | Category | Key observation |
|---|----------|----------------|
| 1 | Easy US Latin | True S1 at TF-IDF rank 1; all models agree. Gate: LightGBM + mDeBERTa dominate. |
| 2 | Indic script (Devanagari/Malayalam) | Romanisation + lexicon aligns tokens. Gate shifts weight to LightGBM (~70%). Correct. |
| 3 | Sibling branch / distractor | Same core name, different address. House/unit numbers and dense embeddings disambiguate. |
| 4 | Failure case | Ambiguous / empty address confuses matchers. Even MoE fails — honest failure analysis. |

---

## SLIDE 3 (optional) — Architecture Comparison Diagram

```
INPUT PAIR  (S2/S3 record q,  S1 candidate s)
     │
     ├── TF-IDF char 3-gram similarity ─────────────────────► z_gbm
     │   + parsed address fields (HOUSE, STREET, …)
     │   + 30+ string features
     │        └── LightGBM ──────────────────────────────────► z_gbm
     │
     ├── Character sequences (ltext = "name | addr")
     │        └── Siamese BiLSTM / BiGRU / BiRNN ──────────► z_rnn_{a,b,c}
     │
     └── ce_text = "name [| romanised] | addr"
              ├── BGE reranker (zero-shot) ─────────────────► z_ce_zs
              ├── BGE full fine-tune ───────────────────────► z_ce_ft
              ├── BGE + LoRA r=16 ──────────────────────────► z_ce_lora
              ├── BGE + MoE head (top-2 of 4) ─────────────► z_ce_moe
              ├── IndicBERTv2 fine-tune ────────────────────► z_indic
              └── mDeBERTa-v3 fine-tune ───────────────────► z_mdeberta

   z₁…z_N (N expert logits, normalised to z-score on part B)
   x (11 gate features from record properties + retrieval scores)
        │
        ▼
   GatedMoE: w(x) = softmax(MLP(x))
             p = σ(Σᵢ wᵢ·(aᵢzᵢ+bᵢ) + c)
        │
        ▼
   Decision: match candidate j* = argmax p  if p[j*] ≥ threshold
             else: no match (new entity)
```

---

## Slide notes for Akshat & Keshav

- **Slide 1** covers the MoE architecture; include the two architecture boxes
  and the training protocol table. The gate diagram is the key visual.
- **Slide 2** covers results; fill the table from `final_results.csv` (attached
  to notebook 07 output / `er-07-output` Kaggle dataset).
- **Slide 3** is optional; use as a backup slide or combine with Slide 1.
- Use the bar chart `results_test_f05.png` from the 07 output as a figure
  in Slide 2 — it already has colour-coded sections (classical / RNN / Transformer / Fusion).
- The gate-weight chart `moe_gates.png` from 07 output is a good figure for Slide 1
  to show which expert handles which script.

---

*Files referenced:*
- `demo/report_section_moe_results.md` — full text for the report
- `demo/demo_script_5min.md` — 5-minute live demo script
- Kaggle 07 output: `final_results.csv`, `final_results.md`, `results_test_f05.png`, `moe_gates.png`
- Kaggle 08 output: `demo_fallback.html`

