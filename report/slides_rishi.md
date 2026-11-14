# Slides: Recurrent matchers and Script-specific analysis
### Rishi · issues #7, #8, #9 → hand to Akshat for the deck

Six slides, three per section. Each slide has its content, the figure or table to use, and a line of
speaker notes. Figures are in `analysis/out_rnn/` and `analysis/out_script/`.

---

## SLIDE R1: Siamese character-level recurrent matcher

**Title:** Recurrent matchers: reading records character by character

```
 S2/S3 record                         S1 candidate
 "sree systems pvt ltd | 436/a …"     "sree systems private limited | 436/a …"
        │                                    │
  char embedding (64)                  char embedding (64)        ← shared weights
        │                                    │
  BiLSTM / BiGRU / BiRNN (2×128)       same encoder
        │                                    │
  max-pool ⊕ mean-pool → u (512)       max-pool ⊕ mean-pool → v (512)
        └──────────────┬─────────────────────┘
              [ u ; v ; |u − v| ; u ⊙ v ]  (2048)
                       │
              MLP 2048 → 256 → 1  →  match logit
```

- Characters, not words: typos, abbreviations and transliterations are small edits.
- Trained on the **hard** candidate pairs from blocking (part A), BCE, 3 × 600k pairs, 13 min on a T4.
- The three notebooks differ **only in the recurrent cell**.

*Notes:* `|u − v|` shows where the records differ, `u ⊙ v` where they agree. The encoder is shared, so
each record is encoded once.

---

## SLIDE R2: LSTM ≈ GRU > vanilla RNN

**Title:** 0.7M parameters reach 0.983, and gating is worth ≈0.01

| | test F0.5 | P | R | singleton acc | params | min |
|---|---|---|---|---|---|---|
| **BiLSTM** | **0.9834** | 0.986 | 0.985 | 0.945 | 0.73M | 13.3 |
| BiGRU | 0.9820 | 0.985 | 0.984 | 0.940 | 0.68M | 13.1 |
| BiRNN (vanilla) | 0.9748 | 0.978 | 0.979 | **0.909** | 0.58M | 11.9 |
| *for scale:* bge full fine-tune | 0.9912 | | | | 278M | |

Figure: `analysis/out_rnn/rnn_val_loss.png` (train / val loss per epoch).

- The vanilla RNN **underfits**: it has the highest *training* loss at every epoch (0.044 vs 0.029), 2.1 %
  clipped steps, and a val-loss spike at epoch 2.
- LSTM vs GRU: 0.0014 apart, within seed noise. We treat them as equal.

*Notes:* 380× fewer parameters than a cross-encoder, ≈0.01 F0.5 behind.

---

## SLIDE R3: Vanishing gradients, measured

**Title:** The vanilla RNN forgets after ~10 characters

Figure: `analysis/out_rnn/rnn_grad_by_distance.png`. ‖∂h_T / ∂x_(T−k)‖ relative to k = 0, after training,
on 100-character records.

| | k = 10 | k = 50 | k = 99 |
|---|---|---|---|
| BiLSTM | 3e-2 | 2e-2 | 1.5e-4 |
| BiGRU | 1.6e-1 | 6e-2 | 6.7e-2 |
| BiRNN | 5e-3 | **1e-12** | **2e-25** |

- Vanilla: `∂h_T/∂h_t = ∏ diag(1 − h_s²) W`, a product of shrinking Jacobians, so ×10 less every 4 characters.
- LSTM / GRU: additive gated path (`∂c_t/∂c_(t−1) = f_t`). Training learns to open it; at init all
  three are ≈1e-11.
- The gap is only 0.01 because pooling over all steps + bidirectionality give every character a short
  path. What is lost is long-range evidence (name ↔ locality), which shows up in singleton accuracy.

---

## SLIDE S1: Per-script F0.5 for every model

**Title:** Script is not what makes a record hard

Figure: `analysis/out_script/per_script_f05.png` (heatmap, 21 matchers × 5 groups).

| | US Latin | India Latin | Devanagari | Malayalam | other Indic (283) |
|---|---|---|---|---|---|
| mDeBERTa-v3 | .992 | **.990** | .997 | .997 | .982 |
| bge full FT | .989 | **.988** | .994 | .992 | .944 |
| BiLSTM | .983 | **.975** | .983 | .978 | .905 |
| bge zero-shot | .924 | .894 | **.883** | **.886** | .962 |
| **Gated MoE fusion (07)** | .993 | **.992** | .996 | .997 | .997 |

- **India Latin is the hardest group for almost every trained model, fusion included.** Native-script
  records are linked *better* than Latin ones after full fine-tuning.
- Fusion gains most where single models are weakest: India Latin (+0.002) and rare scripts (+0.016).
- Zero-shot bge is the only model weak on native scripts. Fine-tuning adds **+0.11** on Devanagari.
- Char-RNNs drop to ≈0.90 on rare scripts (Telugu, Gujarati, …): they see only the romanised text,
  and romanisation fails there.

*Notes:* about half of Devanagari/Malayalam records have the company name itself in that script.

---

## SLIDE S2: IndicBERTv2 vs bge-reranker-base

**Title:** An Indic-specific backbone does not help, even on Indic scripts

Same recipe (full fine-tune, 160k pairs, 1 epoch); only the backbone differs.

| India records | F0.5 IndicBERTv2 | F0.5 bge | only Indic right | only bge right | McNemar p |
|---|---|---|---|---|---|
| Latin (34k) | 0.968 | **0.988** | 173 | 775 | 2e-91 |
| Devanagari (7.9k) | 0.973 | **0.994** | 27 | 139 | 2e-19 |
| Malayalam (10.4k) | 0.978 | **0.991** | 35 | 149 | 6e-18 |
| other Indic (283) | **0.972** | 0.944 | 7 | 6 | 1.0 |

- bge wins significantly everywhere except the 283 rare-script records (a tie).
- Why: bge was pre-trained **as a cross-encoder for relevance**; IndicBERTv2 is MLM-only, so its pair
  head starts from scratch. The task hinges on names and numbers, not Indic morphology.
- IndicBERTv2 also loses most on **US** records (1,192 missed links vs 436), so the deficit is not about script.

---

## SLIDE S3: Five typical errors

**Title:** What every model gets wrong, and what only IndicBERTv2 gets wrong

| | record | what happened | models wrong |
|---|---|---|---|
| 1 | *Eye Care Associates of Hermiston*, **1139** Hartley Ave | linked to S1 at **1135**, but truth is *no match* | 20 / 21 |
| 2 | *The Dent Seafood Corporation*, no address | two S1 branches share the name | 21 / 21 |
| 3 | *Tech അഗ്രോ പ്രൈവറ്റ് ലിമിറ്റഡ്*, no address | true S1 never retrieved (blocking miss) | 21 / 21 |
| 4 | *SREE SYSTEMS PRIVATE LIMITED*, 436/A Machamthode | IndicBERTv2 picks another company at the same address | 1 / 21 |
| 5 | *ಸೂರ್ಯ ಇನ್‌ಫ್ರಾಸ್ಟ್ರಕ್ಚರ್ …* (Kannada) | IndicBERTv2 (+ MoE head, 3 Ditto runs) miss *Surya Infrastructure* | 5 / 21 |

- Shared errors come from the **data**: hard negatives, address-less siblings, blocking misses.
  None is script-specific.
- Script-specific errors belong to a few single models, and **fusion fixes them**. The shared ones
  survive fusion too (counts over 21 matchers: 10 single models, 6 Ditto runs, 5 fusion models).
