# Final Evaluation Results & Mixture-of-Experts Fusion Analysis

**Project:** Cross-Lingual Business Entity Resolution  
**Course Topic:** Mixture of Experts (MoE), Ensemble Fusion, and System Evaluation  
**Evaluation Dataset:** `data/small` (Held-out Test Split, 507,870 candidate pairs)  
**Execution Environment:** Kaggle Tesla T4 (GPU) / Kaggle CPU  
**Blocking Ceiling:** **0.99957 Macro $F_{0.5}$** (Val: 0.99974, 5.00 candidates/record via fused RRF)

---

## 1. Master Results Table

The table below summarizes performance across every completed module of the pipeline, from classical baselines to deep cross-encoders and gated ensemble fusion. All thresholds were calibrated exclusively on the validation split and frozen prior to scoring the held-out test split.

| Section | Method | Val $F_{0.5}$ | Test $F_{0.5}$ | Test $P$ | Test $R$ | Test $F_{0.5}$ (US) | Test $F_{0.5}$ (India) | Singleton Acc | Parameters | Train Min | Threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **5 Classical baseline** | GBM (string + parsed-address features) | 0.99286 | **0.99081** | 0.9938 | 0.9873 | 0.98946 | 0.99205 | 0.9857 | 1,609 trees | 4.60 | 0.36 |
| **6 RNN / LSTM** | Siamese char-BiLSTM | 0.98920 | **0.98335** | 0.9863 | 0.9846 | 0.98390 | 0.98284 | 0.9454 | 0.73M | 13.30 | 0.06 |
| **6 RNN / LSTM** | Siamese char-BiGRU | 0.98814 | **0.98197** | 0.9850 | 0.9835 | 0.98142 | 0.98247 | 0.9404 | 0.68M | 13.10 | 0.15 |
| **6 RNN / LSTM** | Siamese char-BiRNN (vanilla) | 0.98374 | **0.97475** | 0.9782 | 0.9789 | 0.97646 | 0.97316 | 0.9085 | 0.58M | 11.90 | 0.26 |
| **7 Transformers** | CE mDeBERTa-v3 full fine-tune (fp32) | 0.99561 | **0.99402** | 0.9953 | 0.9931 | 0.99261 | 0.99533 | 0.9866 | 279M (100% trained) | 60.40 | 0.11 |
| **7 Transformers** | CE bge-reranker-base + MoE head (top-2 of 4) | 0.99226 | **0.98888** | 0.9914 | 0.9871 | 0.98719 | 0.99044 | 0.9706 | 279M (100% trained) | 20.00 | 0.08 |
| **7 Transformers** | CE bge-reranker-base LoRA ($r=16$) | 0.99142 | **0.98708** | 0.9900 | 0.9854 | 0.98566 | 0.98839 | 0.9681 | 280M (0.5% trained) | 10.10 | 0.11 |
| **7 Transformers** | CE IndicBERTv2 full fine-tune | 0.98249 | **0.97611** | 0.9841 | 0.9666 | 0.97268 | 0.97929 | 0.9446 | 278M (100% trained) | 19.30 | 0.15 |
| **8 Fusion** | **Gated Mixture of Experts** | **0.99638** | **0.99480** | **0.9963** | **0.9930** | **0.99307** | **0.99641** | **0.9933** | 450 | **0.16** | 0.35 |
| **8 Fusion** | LightGBM stacking (same inputs) | 0.99608 | **0.99478** | 0.9968 | 0.9919 | 0.99346 | 0.99602 | 0.9924 | ~150k | 0.80 | 0.40 |
| **8 Fusion** | Logistic-regression stacking | 0.99632 | **0.99476** | 0.9963 | 0.9927 | 0.99288 | 0.99650 | 0.9933 | 7 | 0.10 | 0.34 |
| **8 Fusion** | MoE, constant gate (ablation) | 0.99642 | **0.99474** | 0.9963 | 0.9928 | 0.99299 | 0.99636 | 0.9933 | 18 | 0.50 | 0.34 |
| **8 Fusion** | Mean of 7 calibrated experts | 0.99566 | **0.99339** | 0.9955 | 0.9908 | 0.99216 | 0.99453 | 0.9874 | — | 0.10 | 0.39 |

---

## 2. Key Findings & Course Topic Analysis

### 2.1 Gated MoE Fusion Achieves Global Best Performance
- **Top Score in Project:** The **Gated Mixture of Experts** achieves the highest Macro $F_{0.5}$ score across the entire project (**0.99480** test $F_{0.5}$, Precision 0.9963, Recall 0.9930, Singleton Accuracy 0.9933).
- **Outperforming All Single Models:** Gated MoE surpasses the single best neural model (`mDeBERTa-v3` at 0.99402) and the best classical model (`LightGBM` at 0.99081).
- **Ultra-Fast Training:** Training the gating network on out-of-fold validation logit representations requires only **0.16 minutes (~10 seconds)** on CPU.

### 2.2 BGE MoE Head Inside Cross-Encoder (Issue #13)
- The BGE Cross-Encoder with an internal 4-expert MLP head and Top-2 routing achieved **0.98888 Test $F_{0.5}$**, outperforming parameter-efficient LoRA fine-tuning (**0.98708**) on the exact same backbone.
- Top-1 expert distribution during test inference:
  - Expert 0: `0.08%`
  - Expert 1: `7.75%`
  - Expert 2: `77.79%`
  - Expert 3: `14.37%`
- The auxiliary Switch-style load balancing loss successfully prevented routing collapse while allowing the experts to specialize.

### 2.3 Post-Hoc Gate Weight Analysis by Script & Country (Issue #14)

The input-conditional gating network learns distinct weight distributions across geographic regions and orthographic scripts:

#### Gate Weight per Expert by Script:
| Expert Matcher | Devanagari | Latin | Malayalam | Other Indic |
|---|---|---|---|---|
| **LightGBM Baseline** | **0.708** | **0.562** | **0.702** | **0.844** |
| **CE mDeBERTa-v3 (fp32)** | **0.260** | **0.400** | **0.286** | **0.152** |
| **Siamese BiLSTM** | 0.012 | 0.015 | 0.003 | 0.001 |
| **Siamese BiGRU** | 0.006 | 0.007 | 0.002 | 0.001 |
| **CE BGE MoE Head** | 0.005 | 0.005 | 0.002 | 0.001 |
| **CE IndicBERTv2** | 0.005 | 0.005 | 0.002 | 0.001 |
| **CE BGE LoRA** | 0.005 | 0.005 | 0.002 | 0.001 |

#### Gate Weight per Expert by Country (Never seen by Gate):
| Expert Matcher | India | United States |
|---|---|---|
| **LightGBM Baseline** | **0.655** | **0.516** |
| **CE mDeBERTa-v3 (fp32)** | **0.316** | **0.444** |
| **Siamese BiLSTM** | 0.011 | 0.016 |
| **Siamese BiGRU** | 0.006 | 0.007 |
| **CE BGE MoE Head** | 0.004 | 0.005 |
| **CE IndicBERTv2** | 0.004 | 0.005 |
| **CE BGE LoRA** | 0.004 | 0.006 |

*Observation:* The ensemble places primary confidence on the complementary pairing of **LightGBM** (structured address string and token overlaps) and **mDeBERTa-v3** (deep contextual semantics). When handling Latin entities (US), mDeBERTa weight increases to 44.4%, whereas on Indic and non-Latin entities, LightGBM's normalized address feature weights increase to 70–84%.

---

## 3. Upstream Component Evaluations

### 3.1 Candidate Blocking Evaluation (`03_candidates`)
- **Char 3-gram TF-IDF:** Recall@1: 0.9912 | Recall@5: 0.9972 | Recall@10: 0.9983
- **Fine-tuned e5-small bi-encoder:** Recall@1: 0.9940 | Recall@5: 0.9988 | Recall@10: 0.9994
- **Reciprocal Rank Fusion (RRF top-5):** Recall@5: **0.9989**
- **Empirical Candidate Ceiling:** Val $F_{0.5} = 0.99974$ | Test $F_{0.5} = \mathbf{0.99957}$

### 3.2 Address Parsing Evaluation (`02_parser_bilstm_crf`)
- **With Commas:** BiLSTM-CRF Macro F1: `0.9989` | Exact Address: `0.9969` (Rules: `1.0000`)
- **Commas Removed (Fuzzy Stress Test):** BiLSTM-CRF Macro F1: `0.9063` | Exact Address: `0.8545` (Rules: `0.3691` Macro F1, `0.0000` Exact Address)
