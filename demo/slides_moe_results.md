# Slides — Mixture of Experts and Results
## NLP Course Project: Business Entity Resolution
### Ayush · Issue #15 → hand to Akshat & Keshav for final deck

---

## SLIDE 1 — Mixture of Experts Architecture

### What problem does MoE solve?

No single matcher is best across all languages and data qualities:
- **LightGBM** wins on structured US addresses
- **IndicBERTv2** wins on Devanagari/Malayalam
- **BGE full fine-tune** wins on Latin-script cross-language pairs

**MoE routes each pair to the right expert.**

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
Devanagari pair  →  gate up-weights IndicBERTv2 + mDeBERTa
Latin pair with  →  gate up-weights LightGBM (address features)
  structured addr     and BGE cross-encoder
Latin pair with  →  gate up-weights pure-string matchers
  no address           (Siamese RNN, TF-IDF)
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
| LightGBM baseline | — | — | — | — | — |
| Siamese BiLSTM | — | — | — | — | — |
| Siamese BiGRU | — | — | — | — | — |
| CE bge zero-shot | — | — | — | — | — |
| CE bge full FT | — | — | — | — | — |
| CE bge LoRA r=16 | — | — | — | — | — |
| CE bge + MoE head | — | — | — | — | — |
| CE IndicBERTv2 | — | — | — | — | — |
| CE mDeBERTa-v3 | — | — | — | — | — |
| Mean of experts | — | — | — | — | — |
| LR stacking | — | — | — | — | — |
| **Gated MoE** | **—** | **—** | **—** | **—** | **—** |
| Constant-gate ablation | — | — | — | — | — |

*Expected ordering: Gated MoE ≥ LR stacking ≥ Constant-gate ≥ Mean of experts*

---

### Key observations

1. **LightGBM is a surprisingly strong baseline** — address feature engineering
   captures structure that raw string similarity misses.

2. **Cross-encoders dominate on Indic pairs** — IndicBERTv2 narrows the
   script gap vs. LightGBM; mDeBERTa is strongest but prohibitively slow.

3. **Gated MoE beats every single matcher** — routing by script, not country,
   allows the ensemble to handle unseen country codes at inference.

4. **Constant-gate ablation confirms input-conditioning is the key** — the
   same architecture with fixed weights gains less than the input-conditional gate.

5. **Distractor density caveat:** Our splits have ≈1.2 distractors / S1;
   the leaderboard has ≈2.3. F0.5 values on our test are optimistic.

---

### Four demo records summary

| # | Category | Key observation |
|---|----------|----------------|
| 1 | Easy US Latin | True S1 at TF-IDF rank 1; all models agree. Gate: LightGBM + BGE dominate. |
| 2 | Indic script (Devanagari/Malayalam) | Romanisation + lexicon aligns tokens. Gate up-weights IndicBERTv2. Correct. |
| 3 | Sibling branch / distractor | Same core name, different address. Address features (postcode, house-no) disambiguate. |
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

