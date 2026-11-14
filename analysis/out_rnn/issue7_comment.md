05a / 05b / 05c are done (full runs, `QUICK_RUN = False`, T4, 3 epochs × 600k part-A pairs each).

**Outputs (public Kaggle datasets)**
- `er-05a-output`: <link>
- `er-05b-output`: <link>
- `er-05c-output`: <link>

Each zip has `results/`, `scores/` (B/val/test logits for 07), `preds/`, `models/*.rnn.pt` (for 08) and the new `diagnostics/<method>.json` (per-epoch history + vanishing-gradient probe).

### Results (held-out test split, threshold tuned on val)

| model | val F0.5 | **test F0.5** | test P | test R | F0.5 US | F0.5 India | singleton acc | params | train min |
|---|---|---|---|---|---|---|---|---|---|
| Siamese char-BiLSTM (05a) | 0.98920 | **0.98335** | 0.9863 | 0.9846 | 0.98390 | 0.98284 | 0.9454 | 0.73M | 13.3 |
| Siamese char-BiGRU (05b) | 0.98814 | **0.98197** | 0.9850 | 0.9835 | 0.98142 | 0.98247 | 0.9404 | 0.68M | 13.1 |
| Siamese char-BiRNN, vanilla (05c) | 0.98374 | **0.97475** | 0.9782 | 0.9789 | 0.97646 | 0.97316 | 0.9085 | 0.58M | 11.9 |

### Per-epoch training (from `diagnostics/*.json`)

| model | epoch | train loss | val loss | val F0.5 | mean grad norm | steps clipped |
|---|---|---|---|---|---|---|
| BiLSTM | 1 / 2 / 3 | 0.0690 / 0.0339 / 0.0291 | 0.0197 / 0.0186 / 0.0248 | 0.9862 / 0.9880 / 0.9892 | 0.22 / 0.16 / 0.13 | 0.1% / 0% / 0.1% |
| BiGRU | 1 / 2 / 3 | 0.0679 / 0.0355 / 0.0303 | 0.0209 / 0.0229 / 0.0184 | 0.9839 / 0.9870 / 0.9881 | 0.29 / 0.23 / 0.18 | 0.3% / 0% / 0% |
| BiRNN | 1 / 2 / 3 | 0.0811 / 0.0503 / 0.0436 | 0.0272 / **0.0416** / 0.0236 | 0.9801 / 0.9811 / 0.9837 | 0.45 / 0.36 / 0.31 | **2.1%** / 0.3% / 0% |

Plots: `analysis/out_rnn/rnn_val_loss.png`, `analysis/out_rnn/rnn_grad_by_distance.png` (branch `rishi/issue-7-siamese-rnn`).

### Does the vanilla RNN train worse on 100-character inputs? Yes.

- **Accuracy:** −0.0086 test F0.5 against the BiLSTM, with both P and R lower and the largest drop on singletons (0.909 vs 0.945). India records drop the most (0.973).
- **Training:** its training loss is the highest at every epoch (0.044 vs 0.029 after 3 epochs), so the model underfits, and the gap is not a regularisation effect. Its val loss jumps at epoch 2 (0.027 → 0.042), and it has the largest gradient norms and the most clipped steps in epoch 1 (2.1% vs ≤ 0.3%). Optimisation is less stable.
- **Why (vanishing gradients), measured directly.** The notebooks now measure ‖∂h_T / ∂x_(T−k)‖, how much the forward state at the last character still depends on the character *k* positions earlier, on 256 full-length training records. Relative to k = 0, after training:

| model | k = 10 | k = 25 | k = 50 | k = 99 |
|---|---|---|---|---|
| BiLSTM | 3.0e-2 | 1.8e-2 | 2.0e-2 | 1.5e-4 |
| BiGRU | 1.6e-1 | 6.9e-2 | 6.1e-2 | 6.7e-2 |
| BiRNN (vanilla) | 5.4e-3 | 1.4e-6 | **1.0e-12** | **2.4e-25** |

  In a vanilla RNN, h_t = tanh(W h_(t−1) + U x_t), so ∂h_T/∂h_t = ∏ diag(1 − h_s²) W. That is a product of T − t Jacobians whose norm is below 1, so it shrinks geometrically. We measure about ×10 less every 4 characters, so after training the vanilla RNN effectively "remembers" only the last ~10 characters. The LSTM and GRU carry state through an **additive, gated path**: ∂c_t/∂c_(t−1) = f_t for the LSTM, and (1 − z_t) for the GRU. Training learns to keep those gates near 1, so the gradient stays roughly flat over 100 characters. At initialisation all three cells have tiny long-range gradients (≈1e-11 at k = 50); only the gated cells *learn* to open the path.

- **Why the gap is only 0.009 and not larger:** the encoder max- and mean-pools the outputs of **all** time steps, so every character has a short path to the loss. It is also bidirectional, and much of the matching signal is local (character n-grams of the name). The vanilla RNN still works as a local n-gram detector. What it loses is the ability to combine evidence across the whole `name | address` string (e.g. a name token with a distant postcode or locality), which is exactly what separates near-duplicate branches. That is also why its singleton accuracy drops most.

**LSTM vs GRU:** the BiLSTM is ahead by 0.0014 test F0.5 with 7% more parameters and the same training time. That is within what one would expect from seed noise (single run each), so we treat them as equal. The GRU actually keeps more long-range gradient (6.7e-2 vs 1.5e-4 at k = 99), so the LSTM's small edge is not about memory length. Note that the BiLSTM's val loss rises at epoch 3 while its val F0.5 still improves (threshold falls from 0.35 to 0.06). It gets over-confident on a few hard pairs, which hurts log-loss but not the ranking.
