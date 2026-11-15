# Report Section — Mixture of Experts and Experimental Results

**Course: NLP | Project: Business Entity Resolution**  
**Author: Ayush (Issue #15) · Models: 06d, 07, 08**

---

## A. Mixture of Experts

### A.1 Motivation

No single matcher is universally best across the two countries and three scripts
in our dataset. LightGBM is fast and decides from 03's retrieval scores plus explicit
string and number mismatches (Section G.3); the cross-encoders read both records jointly.
While IndicBERTv2 provides dedicated Indic
subword tokenization, multilingual cross-encoders like BGE and mDeBERTa achieve higher
F0.5 across both Latin and Indic scripts ($p < 10^{-17}$, see Issue #8). Rather
than picking one winner, we use a **Mixture of Experts (MoE)** to weight each
matcher adaptively, based on the properties of each individual candidate pair.

### A.2 Two MoE Components in This Project

The project implements MoE at two levels:

| Component | Notebook | Purpose |
|-----------|---------|---------|
| **MoE head on the cross-encoder** | `06d` | Replaces the single classification head of `bge-reranker-base` with 4 expert MLPs; a gate selects the top 2 per pair (Switch-style). Trained end-to-end. |
| **Gated MoE fusion over all matchers** | `07` | Treats every matcher (LightGBM, Siamese RNNs, all cross-encoders) as an expert; a gate network routes per pair at inference. Trained on held-out part-B pairs. |

### A.3 MoE Head on the Cross-encoder (Notebook 06d)

```
Input pair  →  bge-reranker-base encoder  →  [CLS] vector  h ∈ ℝ^d
                                              │
                                    Gate: Linear(d, 4)  →  softmax  →  probs ∈ ℝ^4
                                              │
                                    top-2 indices selected  →  weights renormalised
                                              │
                              Expert_0  Expert_1  Expert_2  Expert_3
                              (each: Dropout → Linear(d,256) → GELU → Linear(256,1))
                                              │
                              weighted sum of top-2 expert outputs  →  logit
```

**Load-balancing loss (Switch-style):** To prevent expert collapse, a load-
balancing auxiliary loss is added:

> L_lb = N · Σᵢ fᵢ · pᵢ

where fᵢ is the fraction of tokens (pairs) routed to expert i as top-1, and pᵢ
is the mean gate probability for expert i. The total loss is BCE + λ·L_lb
(λ = 0.01). Without such a term, a learned top-k gate tends to collapse onto a few
experts (Shazeer et al., 2017).

**Gate analysis output (notebook 06d):** After training, the notebook reports
the mean gate weight per expert, broken down by script and country. In our run the
experts do **not** specialise by script: the four mean gate weights are almost the
same in every script (≈0.21 / 0.22 / 0.23 / 0.33, Section S.5).

### A.4 Gated MoE Fusion over All Matchers (Notebook 07)

#### Architecture

```
Matcher outputs (logits):  z_1, z_2, ..., z_N   (one per expert/matcher)
Gate inputs (13 features): x  =  [script_latin, script_devanagari,
                                   script_malayalam, script_other,
                                   is_S3, addr_empty, addr_has_number,
                                   name_len/40, addr_len/80,
                                   tf_cos, e5_cos, rrf_rank/K, n_cand/K]
                                   (z-scored on part B)

Gate network:  x  →  Linear(13, 32)  →  ReLU  →  Linear(32, N)  →  softmax  →  w(x)

Fusion:  score = σ(Σᵢ wᵢ(x) · (aᵢ zᵢ + bᵢ) + c)
```

- **aᵢ, bᵢ** (per-expert affine calibration) are learned parameters, initialised
  from individual logistic fits so the model starts at the "mean of calibrated
  experts" before gradient steps.
- **c** is a global bias.
- **w(x)** is the input-conditional gate: softmax weights that depend on the
  record properties and retrieval scores, never the country label.

#### Training Protocol

- Trained on **part B** of the training pairs — the 15% of training pairs that
  no matcher was trained on. This prevents the gate from fitting to in-sample
  idiosyncrasies of any individual matcher.
- Optimiser: Adam, lr=1e-2, weight-decay=1e-5.
- Schedule: cosine annealing over max(30 epochs × steps_per_epoch, 2000) steps.
- Batch size: 2048 pairs.
- Loss: binary cross-entropy on the match label.

#### Why Input-Conditional Gating Helps

The gate sees the record's script, its address quality and the retrieval scores. This lets it:

- Route between the cross-encoders (BGE / mDeBERTa) and the feature-based LightGBM depending on script and address completeness.
- Shift weight toward LightGBM where the cross-encoders are less reliable. In our run this happens on the Indic scripts (Section B.4).
- Down-weight models that are overconfident or underperform in a specific
  sub-domain, without manual per-country tuning.

#### Baselines Compared in Notebook 07

| Method | Description |
|--------|-------------|
| Mean of calibrated experts | Average of individually calibrated matcher probabilities |
| Logistic-regression stacking | LR on the N normalised logits (global weights, no input) |
| MoE, constant gate (ablation) | Same architecture but gate weights do not depend on input |
| **Gated MoE** (ours) | Full input-conditional gate — our final system |
| LightGBM stacking | LightGBM on the N logits + 13 gate features |

#### Input / Output Flow Diagram

```
                    ┌──────────────────────────────────────────────────┐
                    │  Candidate pair (q, s)                           │
                    │  q: S2/S3 record   s: S1 candidate               │
                    └──────────────────────────────────────────────────┘
                                        │
              ┌─────────────────────────┼──────────────────────────┐
              ▼                         ▼                          ▼
       Matcher logits            Gate features (13)         [already computed]
       z₁…z_N (from             script, addr quality,       during retrieval step
       04-06 models)            retrieval scores
              │                         │
              └─────────────┬───────────┘
                            ▼
                   GatedMoE.forward(z, g)
                     w = softmax(gate(g))       ← input-conditional weights
                     out = Σᵢ wᵢ·(aᵢzᵢ+bᵢ)+c  ← affine calibration per expert
                            ▼
                   p = σ(out)                   ← match probability
                            ▼
              Decision: best candidate if p ≥ threshold
              (threshold tuned on val split, test scored once)
```

---

## B. Experimental Results

### B.1 Dataset Details

| Split | S1 entities | True links | S2+S3 pool | Unlinked (distractors) |
|-------|------------|-----------|-----------|----------------------|
| train | 75,353 | 261,183 | 355,377 | 94,194 (26.5%) |
| val | 10,764 | 37,236 | 50,752 | 13,516 (26.6%) |
| test | 21,529 | 74,619 | 101,574 | 26,955 (26.5%) |

**Countries and scripts in `data/small`:**

| Country | States | Script |
|---------|--------|--------|
| US | Oregon, Utah | Latin |
| India | Kerala, Madhya Pradesh | Malayalam (Kerala), Devanagari (MP) |

**Blocking ceiling** (from notebook 03): with the candidate retrieval (TF-IDF + e5,
top-5 by RRF), a perfect matcher would reach test macro F0.5 **0.99957**. This is
the upper bound for every matcher on the retrieved candidates.

### B.2 Evaluation Methodology

- **Metric:** Macro F0.5 per S1 entity (singletons — entities with no true link
  — count as correct if they receive no prediction).
- **Decision rule:** Each S2/S3 record is assigned to its highest-scoring S1
  candidate if that score ≥ threshold; otherwise it is treated as an unlinked
  entity.
- **Threshold tuning:** On the **val** split only. Test is scored exactly once
  per model.
- **Part A / B split:** Training pairs are divided 85/15 (A/B) by hashed S1 id.
  Matchers (04–06) are trained on part A. The gated MoE (07) is trained on
  part B, so no matcher ever contributes training signal to the data the gate
  learns from.

> **Limitation:** Our splits come from the labelled training data, so the
> distractor density (≈1.2 unlinked records per S1) is lower than the
> official leaderboard (≈2.3). F0.5 on our test split is therefore optimistic.
> Do not compare directly with the public LB score of 0.9895 (full data,
> different validation scheme).

### B.3 Individual Model Performance

The table comes from notebook 07 (`final_results.md`, the final 9-expert run). Values are
from the final run of each notebook on Kaggle (GPU T4; CPU for 04 and 07).

| Section | Method | Val F0.5 | Test F0.5 | Test P | Test R | US F0.5 | India F0.5 |
|---------|--------|---------|---------|--------|--------|---------|-----------|
| 5 Classical | LightGBM (retrieval + string features) | 0.99286 | 0.99081 | 0.9938 | 0.9873 | 0.98946 | 0.99205 |
| 6 RNN / LSTM | Siamese char-BiLSTM | 0.98920 | 0.98335 | 0.9863 | 0.9846 | 0.98390 | 0.98284 |
| 6 RNN / LSTM | Siamese char-BiGRU | 0.98814 | 0.98197 | 0.9850 | 0.9835 | 0.98142 | 0.98247 |
| 6 RNN / LSTM | Siamese char-BiRNN (vanilla) | 0.98374 | 0.97475 | 0.9782 | 0.9789 | 0.97646 | 0.97316 |
| 7 Transformers | CE mDeBERTa-v3 full fine-tune (fp32) | 0.99561 | 0.99402 | 0.9953 | 0.9931 | 0.99261 | 0.99533 |
| 7 Transformers | CE bge-reranker-base full fine-tune | 0.99338 | 0.99124 | 0.9941 | 0.9878 | 0.98935 | 0.99299 |
| 7 Transformers | CE bge-reranker-base + MoE head (top-2 of 4) | 0.99226 | **0.98888** | 0.9914 | 0.9871 | 0.98719 | 0.99044 |
| 7 Transformers | CE bge-reranker-base LoRA r=16 | 0.99142 | 0.98708 | 0.9900 | 0.9854 | 0.98566 | 0.98839 |
| 7 Transformers | CE IndicBERTv2 full fine-tune | 0.98249 | 0.97611 | 0.9841 | 0.9666 | 0.97268 | 0.97929 |
| 7 Transformers | CE bge-reranker-base zero-shot | 0.94192 | 0.91674 | 0.9161 | 0.9595 | 0.92713 | 0.90712 |
| **8 Fusion** | **Gated Mixture of Experts** | **0.99648** | **0.99481** | **0.9961** | **0.9935** | **0.99304** | **0.99645** |
| 8 Fusion | Logistic-regression stacking | 0.99648 | 0.99477 | 0.9967 | 0.9920 | 0.99322 | 0.99620 |
| 8 Fusion | MoE, constant gate (ablation) | 0.99642 | 0.99474 | 0.9963 | 0.9928 | 0.99299 | 0.99636 |
| 8 Fusion | LightGBM stacking (same inputs) | 0.99632 | 0.99470 | 0.9968 | 0.9918 | 0.99305 | 0.99622 |
| 8 Fusion | Mean of 9 calibrated experts | 0.99525 | 0.99286 | 0.9941 | 0.9927 | 0.99204 | 0.99363 |

### B.4 Key Observations

1. **LightGBM is a strong baseline** (0.9908, third-best single model, 4.6 CPU
   minutes). 83 % of its gain comes from 03's retrieval scores (above all the
   fine-tuned e5 cosine); the hand-built features add what the embeddings miss,
   mainly numbers that appear in only one of the two addresses. The parsed address
   fields carry 0.6 % of the gain, and the data has no postcodes (Section G.3).

2. **Cross-encoders substantially outperform character-level RNNs on Indic
   pairs.** The BiLSTM/GRU encoders see only the romanised text. On the 283
   other-Indic records they drop to 0.90–0.91, while the cross-encoders, which also
   read the original-script name, stay at 0.94–0.98 (Section S.2).

3. **Cross-encoders (BGE, mDeBERTa) outperform IndicBERTv2 even on Indian scripts**
   ($p < 10^{-17}$, see Issue #8), though IndicBERTv2 provides native Indic vocabulary coverage. The gate gives IndicBERTv2 almost no weight (under 0.4%); it blends LightGBM (60-91%) and mDeBERTa-v3 (9-36%), moving weight toward LightGBM on Indic scripts.

4. **Input-conditional gating does not measurably beat constant gating.** The
   ablation (MoE with a constant gate = learned fixed weights, a weighted
   ensemble) scores 0.99474 on test against 0.99481 for the gated MoE (val
   0.99642 vs 0.99648); LR stacking ties the gated MoE on val (0.99648) and
   scores 0.99477 on test. In a paired bootstrap over test entities, the gain of
   the gated MoE over mDeBERTa-v3 (+0.0008) is significant, but its gain over the
   constant gate (+0.00007) is not, and both gaps are below the run-to-run noise
   of 0.003–0.006 seen in the 06g reruns. The
   gain over single models comes from learned, calibrated weights over two
   complementary experts, not from routing by record properties.

5. **Distractor density caveat.** At training density (1.2 distractors / S1),
   precision is easier to maintain than at leaderboard density (2.3 / S1).
   The ranking of methods should carry over; the absolute F0.5 values are
   optimistic.

### B.5 Analysis of the Four Demo Records

| # | Category | Retrieval outcome | Gating observation | Final verdict |
|---|----------|------------------|-------------------|---------------|
| 1 | Easy US Latin | True S1 at TF-IDF rank 1 | High weight on LightGBM + mDeBERTa-v3; all models agree | Correct |
| 2 | Indic script | True S1 retrieved (romanisation needed) | Gate shifts weight to LightGBM (~70%); lexicon-corrected tokens help | Correct |
| 3 | Sibling branch | True S1 and sibling both in top-5 | Experts separate them by house / shop numbers present in only one address (LightGBM) and by reading both addresses (cross-encoders) | Usually correct |
| 4 | Failure case | True S1 may be in candidates | Overconfident matcher gets down-weighted; MoE may still fail | Wrong |

### B.6 Strengths and Limitations

**Strengths:**
- Self-contained pipeline from raw TSV to match predictions.
- Handles three scripts without hard-coding language-specific rules.
- The gate never sees the country label; it routes on script and retrieval scores (we did not test it on an unseen country).
- Threshold tuned on val; test scored once — no leakage.

**Limitations:**
- `data/small` is 3.5% of the full dataset; absolute numbers do not transfer.
- Romanisation loses some information (e.g. vowel length, geminate consonants).
- The BiLSTM-CRF parser was trained on silver labels from the rule parser, so rule errors propagate to the GBM's parsed-field features (which carry only 0.6 % of its gain).
- mDeBERTa-v3 must run in fp32 (fp16 gives NaN losses), which makes it the slowest matcher: 60.4 min of training on a T4, 9.3 GB of GPU memory.
- Gradio share links expire when the Kaggle session ends; the fallback HTML is the durable artifact.

