# Results from the Kaggle runs

Small result files only (JSON / CSV). The weights, scores and predictions are in the `out_<nn>.zip` files,
shared as Kaggle Datasets (see `TEAM.md`). Every number: `data/small`, same candidate pairs from `03_candidates`,
threshold tuned on **val**, **test** scored once, `QUICK_RUN = False`.

| Notebook | Method | Val F0.5 | **Test F0.5** | P | R | India | US | Train min |
|---|---|---|---|---|---|---|---|---|
| 06a | bge-reranker-base zero-shot | 0.9419 | **0.9167** | 0.9161 | 0.9595 | 0.9071 | 0.9271 | 0 |
| 06b | bge-reranker-base full fine-tune (160k pairs) | 0.9934 | **0.9912** | 0.9941 | 0.9878 | 0.9930 | 0.9894 | 19.4 |

## 06g Ditto ablation (40k training pairs each, 1 epoch, one seed)

| Run | DK | SU | DA | Backbone | Val F0.5 | **Test F0.5** | P | R | Pair F1 |
|---|---|---|---|---|---|---|---|---|---|
| R1 plain | | | | xlm-roberta-base | 0.9843 | **0.9788** | 0.9827 | 0.9786 | 0.9731 |
| R2 +DK | x | | | xlm-roberta-base | 0.9845 | **0.9803** | 0.9890 | 0.9675 | 0.9739 |
| R3 +SU | | x | | xlm-roberta-base | 0.9818 | **0.9754** | 0.9842 | 0.9650 | 0.9702 |
| R4 +DA (MixDA, all ops) | | | x | xlm-roberta-base | 0.9874 | **0.9817** | 0.9853 | 0.9807 | 0.9769 |
| R5 DK+SU+DA | x | x | x | xlm-roberta-base | 0.9809 | **0.9779** | 0.9883 | 0.9622 | 0.9696 |
| R6 DK+SU+DA | x | x | x | bge-reranker-base | 0.9871 | **0.9821** | 0.9856 | 0.9813 | 0.9767 |

Notes for the report:
* All differences between Ditto runs are at most 0.007 on one seed. Treat the ordering as indicative, not established.
* DK ran with the **rule-based** address parser (`parsed_fields: rules` in the JSON files), because the output of
  `02_parser_bilstm_crf` was not attached. The BiLSTM-CRF version of DK is not measured yet.
* 06b (full fine-tune of the same backbone on plain text) is higher than every Ditto run, but it was trained on
  160k pairs against 40k here, so they are not comparable until the `final` run (160k pairs) is done.
* `ditto_lengths.csv`: 5.8 % of pairs are cut at 96 tokens without summarisation, 3.6 % with SU; Malayalam
  22.6 % vs 12.9 %.
* Not run yet: `operators`, `label_eff`, `final`.
