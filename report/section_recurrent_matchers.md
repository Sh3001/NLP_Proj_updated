# Report section: Recurrent matchers

**Author:** Rishi (issues #7, #9) · **Notebooks:** `05a_siamese_bilstm`, `05b_siamese_bigru`, `05c_siamese_birnn`
**Course topics:** RNN, LSTM, GRU, vanishing gradients

Figures: `analysis/out_rnn/rnn_val_loss.png`, `analysis/out_rnn/rnn_grad_by_distance.png`.
Tables: `analysis/out_rnn/rnn_table.md`, `rnn_epochs.md`, `rnn_grad_by_distance.md`.
Reproduce with `python3 analysis/rnn_compare.py <folder with the three er-05*-output zips>`.

---

## R.1 Model: a Siamese character-level recurrent network

The recurrent matchers read each record as a **sequence of characters**, not words. Business names and
addresses are full of typos, abbreviations, transliteration variants and missing spaces
(`Glyfelle` / `Glyphelle`, `Pvt` / `Private`, `Gangtoch` / `Gangtech`). A character model sees these as small
edits instead of unknown words.

**Input.** One string per record, `name | address`, after the shared normalisation from `03_candidates`.
That step does NFKC, romanises Devanagari, Malayalam and other Indic scripts, applies the transliteration
lexicon learned on training pairs, lower-cases and expands abbreviations. The string is cut at 100
characters. The vocabulary is the 250 most frequent training characters seen at least 20 times, plus
padding and unknown.

**Encoder (shared by both records, hence *Siamese*).**

```
characters ──► Embedding(≈250, 64) ──► bidirectional RNN cell, 128 units per direction
            ──► per-step outputs h_1 … h_T ∈ ℝ^256
            ──► [ max-pool over t ; mean-pool over t ] = record vector u ∈ ℝ^512
```

**Pair head.** For the two record vectors `u` (S2/S3 record) and `v` (S1 candidate):

```
x = [ u ; v ; |u − v| ; u ⊙ v ] ∈ ℝ^2048  ──► Linear(2048, 256) ──► ReLU ──► Dropout(0.1) ──► Linear(256, 1) ──► logit
```

`|u − v|` and `u ⊙ v` are the standard Siamese comparison features: `|u − v|` measures where the two encodings differ,
and `u ⊙ v` where they agree. With them, a small MLP can learn a similarity function rather than
memorise record pairs. The two encoders share weights, so each record is encoded once and the head
runs on all of its candidate pairs.

**Three variants that differ only in the cell.**

| notebook | cell | recurrence | params |
|---|---|---|---|
| 05a | LSTM | input/forget/output gates + cell state `c_t = f_t ⊙ c_(t−1) + i_t ⊙ g_t` | 0.73M |
| 05b | GRU | update/reset gates, `h_t = (1 − z_t) ⊙ h_(t−1) + z_t ⊙ ĥ_t` | 0.68M |
| 05c | vanilla (Elman) RNN | `h_t = tanh(W h_(t−1) + U x_t + b)` | 0.58M |

Most parameters (≈0.52M) sit in the pair head. The recurrent layers themselves have 199k (LSTM),
149k (GRU) and 50k (RNN) weights, a ratio of 4 : 3 : 1 gate blocks.

**Training.** Binary cross-entropy on the **part-A candidate pairs** from `03_candidates`. The negatives
are the hard ones that blocking retrieved (same city, similar name), not random pairs. Settings: 3 epochs
of 600k sampled pairs, batch 512, Adam (lr 1e-3), gradient clipping at norm 1.0, Kaggle T4. Each
model then follows the shared protocol: every S2/S3 record goes to its best-scoring S1 candidate if
p ≥ threshold, the threshold is tuned on val, and test is scored once with macro F0.5.

## R.2 Results

| model | val F0.5 | **test F0.5** | test P | test R | F0.5 US | F0.5 India | singleton acc | params | train min |
|---|---|---|---|---|---|---|---|---|---|
| Siamese char-BiLSTM (05a) | 0.98920 | **0.98335** | 0.9863 | 0.9846 | 0.98390 | 0.98284 | 0.9454 | 0.73M | 13.3 |
| Siamese char-BiGRU (05b) | 0.98814 | **0.98197** | 0.9850 | 0.9835 | 0.98142 | 0.98247 | 0.9404 | 0.68M | 13.1 |
| Siamese char-BiRNN, vanilla (05c) | 0.98374 | **0.97475** | 0.9782 | 0.9789 | 0.97646 | 0.97316 | 0.9085 | 0.58M | 11.9 |

For context, on the same pairs: the GBM baseline (04) scores 0.9908, bge-reranker-base fine-tuned (06b)
0.9912, and mDeBERTa-v3 (06f) 0.9940. The recurrent matchers have 380× fewer parameters than a
cross-encoder and train in 13 minutes. They reach 0.983, which is ≈0.01 below the transformers.

**Per-epoch training** (from `diagnostics/*.json`):

| model | train loss (ep 1 / 2 / 3) | val loss | val F0.5 | mean grad norm | steps clipped |
|---|---|---|---|---|---|
| BiLSTM | 0.069 / 0.034 / 0.029 | 0.020 / 0.019 / 0.025 | 0.986 / 0.988 / 0.989 | 0.22 / 0.16 / 0.13 | 0.1 % / 0 % / 0.1 % |
| BiGRU | 0.068 / 0.036 / 0.030 | 0.021 / 0.023 / 0.018 | 0.984 / 0.987 / 0.988 | 0.29 / 0.23 / 0.18 | 0.3 % / 0 % / 0 % |
| BiRNN | 0.081 / 0.050 / 0.044 | 0.027 / **0.042** / 0.024 | 0.980 / 0.981 / 0.984 | 0.45 / 0.36 / 0.31 | **2.1 %** / 0.3 % / 0 % |

## R.3 Why the vanilla RNN is worse: vanishing gradients, measured

The vanilla BiRNN loses 0.0086 test F0.5 against the BiLSTM. Precision and recall are both lower, and
the largest drop is on singletons (0.909 vs 0.945). Three observations point to an optimisation
problem, not overfitting:

1. **It underfits.** Its *training* loss is the highest at every epoch (0.044 vs 0.029 after 3 epochs).
2. **Training is less stable.** It has the largest gradient norms and 2.1 % clipped steps in epoch 1
   (≤ 0.3 % for the gated cells). Its val loss jumps at epoch 2 (0.027 → 0.042).
3. **Its memory is short, measured directly.** The notebooks record ‖∂h_T / ∂x_(T−k)‖, how much the
   forward state at the last character still depends on the character *k* positions earlier. This is
   measured on 256 full-length (100-character) training records, relative to k = 0, after training:

| model | k = 10 | k = 25 | k = 50 | k = 99 |
|---|---|---|---|---|
| BiLSTM | 3.0e-2 | 1.8e-2 | 2.0e-2 | 1.5e-4 |
| BiGRU | 1.6e-1 | 6.9e-2 | 6.1e-2 | 6.7e-2 |
| BiRNN (vanilla) | 5.4e-3 | 1.4e-6 | **1.0e-12** | **2.4e-25** |

**Explanation.** In a vanilla RNN, `h_t = tanh(W h_(t−1) + U x_t)`, so

```
∂h_T / ∂h_t = ∏_{s=t+1..T} diag(1 − h_s²) · W
```

This is a product of T − t Jacobians. When their norm is below 1 (tanh saturation makes `1 − h_s²` < 1),
the gradient shrinks geometrically: we measure about **×10 less every 4 characters**. After training,
the vanilla RNN's final state effectively depends only on the last ~10 characters.

The LSTM and GRU carry state through an **additive, gated path**. For the LSTM,
`∂c_t / ∂c_(t−1) = diag(f_t)`; for the GRU, the corresponding factor is `diag(1 − z_t)`. Training learns to
keep these gates near 1 where needed, so the gradient stays roughly flat over 100 characters. At
initialisation, all three cells have tiny long-range gradients (≈1e-11 at k = 50). Only the gated
cells *learn* to open the path, which is the point of the gates.

**Why the gap is only ≈0.01.** The encoder max- and mean-pools the outputs of **every** time step, so
each character has a short path to the loss. The network is also bidirectional, and much of the
matching evidence is local (character n-grams in the name). The vanilla RNN still works as a local
n-gram detector. What it loses is the ability to combine evidence across the whole `name | address`
string, e.g. a name token with a locality 60 characters later. That is exactly what separates
near-duplicate branches of the same company, and it is why singleton accuracy drops the most.

**LSTM vs GRU.** The BiLSTM is ahead by 0.0014 test F0.5, with 7 % more parameters and the same
training time. With a single run each, this is within seed noise, so we treat them as equal. The GRU
actually keeps *more* long-range gradient (6.7e-2 vs 1.5e-4 at k = 99), so the LSTM's small edge is not
about memory length. The BiLSTM's val loss rises in epoch 3 while its val F0.5 still improves (its
threshold falls from 0.35 to 0.06): it becomes over-confident on a few hard pairs, which hurts log-loss
but not the ranking.

## R.4 Where the recurrent matchers fail

From the per-script analysis (section *Script-specific analysis*):

- The recurrent matchers behave like the other trained models on the main groups: India Latin is the
  hardest group (BiLSTM 0.975), and Devanagari is as easy as US Latin (0.983).
- They **collapse on the 283 "other Indic" records** (Gujarati, Telugu, Tamil, Kannada, Bengali, …):
  BiLSTM 0.905, BiGRU 0.896, BiRNN 0.909, against 0.94–0.98 for the cross-encoders. Every error there is
  a missed link. The RNNs see only the romanised string, and for these scripts the generic
  romanisation gives spellings far from the English S1 name (`emtrpraijes` for *enterprises*). The
  transliteration lexicon cannot fix these, because it only learns words seen in ≥ 3 aligned training
  pairs. The cross-encoders also see the original-script name, and their subword vocabularies cover it.
- The weakness of the vanilla RNN is spread over all four main groups (−0.008 to −0.014 against the
  BiLSTM). It is not tied to one script, which fits the explanation above: memory length, not vocabulary.

## R.5 Summary

- A 0.7M-parameter Siamese char-BiLSTM reaches **0.983 test F0.5** in 13 minutes on a T4,
  ≈0.01 below 278M-parameter cross-encoders.
- **Gated cells matter:** LSTM ≈ GRU > vanilla RNN by ≈0.009. The difference is explained and measured
  as vanishing gradients: the vanilla RNN's dependence on a character 50 positions back is 1e-12, against
  ≈2e-2 for the LSTM.
- Pooling over all time steps and bidirectionality hide most of the vanilla RNN's weakness. The rest
  shows up where long-range evidence matters: singletons and near-duplicate branches.
