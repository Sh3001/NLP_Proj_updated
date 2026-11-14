# Results from the Kaggle runs

Small result files only (JSON / CSV). The weights, scores and predictions are in the `out_<nn>.zip` files,
shared as Kaggle Datasets (see `TEAM.md`). Every number: `data/small`, same candidate pairs from `03_candidates`,
threshold tuned on **val**, **test** scored once, `QUICK_RUN = False`.

| Notebook | Method | Val F0.5 | **Test F0.5** | P | R | India | US | Train min |
|---|---|---|---|---|---|---|---|---|
| 06a | bge-reranker-base zero-shot | 0.9419 | **0.9167** | 0.9161 | 0.9595 | 0.9071 | 0.9271 | 0 |
| 06b | bge-reranker-base full fine-tune (160k pairs) | 0.9934 | **0.9912** | 0.9941 | 0.9878 | 0.9930 | 0.9894 | 19.4 |

## 06g Ditto ablation (40k training pairs each, 1 epoch, one seed per column)

Two complete runs of the same six configurations. In the first, DK used the **rule-based** address parser
(`results/06g/ablation_rules/`); in the second, the **BiLSTM-CRF** parser from 02 (`results/06g/ablation_bilstm_crf/`,
`parsed_fields: BiLSTM-CRF (02)` in the JSON files). Columns P, R and pair F1 are from the BiLSTM-CRF run.

| Run | Val (rules) | **Test (rules)** | Val (BiLSTM-CRF) | **Test (BiLSTM-CRF)** | P | R | Pair F1 |
|---|---|---|---|---|---|---|---|
| R1 plain | 0.9843 | 0.9788 | 0.9845 | **0.9782** | 0.9812 | 0.9804 | 0.9741 |
| R2 +DK | 0.9845 | 0.9803 | 0.9869 | **0.9823** | 0.9880 | 0.9752 | 0.9753 |
| R3 +SU | 0.9818 | 0.9754 | 0.9839 | **0.9774** | 0.9829 | 0.9744 | 0.9729 |
| R4 +DA (MixDA, all ops) | 0.9874 | 0.9817 | 0.9837 | **0.9761** | 0.9816 | 0.9741 | 0.9680 |
| R5 DK+SU+DA | 0.9809 | 0.9779 | 0.9852 | **0.9798** | 0.9861 | 0.9732 | 0.9719 |
| R6 DK+SU+DA on bge-reranker-base | 0.9871 | 0.9821 | 0.9871 | **0.9828** | 0.9889 | 0.9753 | 0.9766 |

Notes for the report:
* **Run-to-run noise is about 0.003 to 0.006.** R1 plain and R4 +DA are identical configurations in both runs, but R4
  moved from 0.9817 to 0.9761 (from +0.003 above plain to 0.006 below it). So the effect of DA (MixDA) is not
  established: it flipped sign between runs. The same goes for SU (negative both times, but only by 0.0008 to 0.003).
* **DK is the only optimisation positive in both runs**: +0.0015 with the rule parser, **+0.0041 with the BiLSTM-CRF
  parser** (val +0.0024). With the BiLSTM-CRF parser, R2 +DK (0.9823) and R6 on bge (0.9828) are the best.
  That is a gain from the project's own parser, though a single seed does not prove it.
* 06b (full fine-tune of the same backbone on plain text) is higher than every Ditto run (0.9912), but it was
  trained on 160k pairs against 40k here, so they are not comparable until the `final` run (160k pairs) is done.
* `ditto_lengths.csv`: 5.8 % of pairs are cut at 96 tokens without summarisation, 3.6 % with SU; Malayalam
  22.6 % vs 12.9 %.
* Not run yet: `operators`, `label_eff`, `final`.
