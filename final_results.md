| section | method | val_f05 | test_f05 | test_P | test_R | test_f05_US | test_f05_India | test_pair_f1 | test_singleton_acc | params | train_min | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 Classical baseline | GBM (string + parsed-address features) | 0.99286 | 0.99081 | 0.9938 | 0.9873 | 0.98946 | 0.99205 |  | 0.9857 | 1609 trees | 4.6 | 0.36 |
| 6 RNN / LSTM | Siamese char-BiLSTM | 0.9892 | 0.98335 | 0.9863 | 0.9846 | 0.9839 | 0.98284 |  | 0.9454 | 0.73M | 13.3 | 0.06 |
| 6 RNN / LSTM | Siamese char-BiGRU | 0.98814 | 0.98197 | 0.985 | 0.9835 | 0.98142 | 0.98247 |  | 0.9404 | 0.68M | 13.1 | 0.15 |
| 6 RNN / LSTM | Siamese char-BiRNN (vanilla) | 0.98374 | 0.97475 | 0.9782 | 0.9789 | 0.97646 | 0.97316 |  | 0.9085 | 0.58M | 11.9 | 0.26 |
| 7 Transformers | CE mDeBERTa-v3 full fine-tune (fp32) | 0.99561 | 0.99402 | 0.9953 | 0.9931 | 0.99261 | 0.99533 |  | 0.9866 | 279M (100.0% trained) | 60.4 | 0.11 |
| 7 Transformers | CE bge-reranker-base full fine-tune | 0.99338 | 0.99124 | 0.9941 | 0.9878 | 0.98935 | 0.99299 |  | 0.984 | 278M (100.0% trained) | 19.4 | 0.05 |
| 7 Transformers | CE bge-reranker-base + MoE head (top-2 of 4) | 0.99226 | 0.98888 | 0.9914 | 0.9871 | 0.98719 | 0.99044 |  | 0.9706 | 279M (100.0% trained) | 20.0 | 0.08 |
| 7 Transformers | CE bge-reranker-base LoRA r=16 | 0.99142 | 0.98708 | 0.99 | 0.9854 | 0.98566 | 0.98839 |  | 0.9681 | 280M (0.5% trained) | 10.1 | 0.11 |
| 7 Transformers | CE IndicBERTv2 full fine-tune | 0.98249 | 0.97611 | 0.9841 | 0.9666 | 0.97268 | 0.97929 |  | 0.9446 | 278M (100.0% trained) | 19.3 | 0.15 |
| 7 Transformers | CE bge-reranker-base zero-shot | 0.94192 | 0.91674 | 0.9161 | 0.9595 | 0.92713 | 0.90712 |  | 0.6868 | 278M (0.0% trained) | 0.0 | 0.28 |
| 7b Ditto ablation | Ditto bge DK+SU+DA-all 40k | 0.98713 | 0.98214 | 0.9856 | 0.9813 | 0.98158 | 0.98266 | 0.97665 | 0.9379 | 278M | 7.5 | 0.05 |
| 7b Ditto ablation | Ditto xlmr DA-all 40k | 0.98744 | 0.98172 | 0.9853 | 0.9807 | 0.98142 | 0.982 | 0.97689 | 0.9395 | 278M | 7.5 | 0.05 |
| 7b Ditto ablation | Ditto xlmr DK 40k | 0.98453 | 0.98028 | 0.989 | 0.9675 | 0.97907 | 0.98141 | 0.97394 | 0.958 | 278M | 5.0 | 0.05 |
| 7b Ditto ablation | Ditto xlmr plain 40k | 0.98429 | 0.9788 | 0.9827 | 0.9786 | 0.97561 | 0.98176 | 0.97313 | 0.9295 | 278M | 5.0 | 0.05 |
| 7b Ditto ablation | Ditto xlmr DK+SU+DA-all 40k | 0.98093 | 0.97792 | 0.9883 | 0.9622 | 0.97938 | 0.97657 | 0.96962 | 0.9538 | 278M | 7.5 | 0.05 |
| 7b Ditto ablation | Ditto xlmr SU 40k | 0.98176 | 0.97541 | 0.9842 | 0.965 | 0.9748 | 0.97597 | 0.97023 | 0.9345 | 278M | 5.0 | 0.06 |
| 8 Fusion | Gated Mixture of Experts | 0.99648 | 0.99481 | 0.9961 | 0.9935 | 0.99304 | 0.99645 |  | 0.9933 |  | 0.17 | 0.3 |
| 8 Fusion | Logistic-regression stacking | 0.99648 | 0.99477 | 0.9967 | 0.992 | 0.99322 | 0.9962 |  | 0.9933 |  |  | 0.41 |
| 8 Fusion | MoE, constant gate (ablation) | 0.99642 | 0.99474 | 0.9963 | 0.9928 | 0.99299 | 0.99636 |  | 0.9933 |  |  | 0.34 |
| 8 Fusion | LightGBM stacking (same inputs) | 0.99632 | 0.9947 | 0.9968 | 0.9918 | 0.99305 | 0.99622 |  | 0.9924 |  |  | 0.41 |
| 8 Fusion | Mean of 9 calibrated experts | 0.99525 | 0.99286 | 0.9941 | 0.9927 | 0.99204 | 0.99363 |  | 0.9857 |  |  | 0.3 |

Blocking ceiling (test macro F0.5): 0.99957

### parser_eval.csv

| input          | parser     |   token_acc |   macro_f1 |   exact_address |
|:---------------|:-----------|------------:|-----------:|----------------:|
| with commas    | BiLSTM-CRF |      0.9994 |     0.9989 |          0.9969 |
| with commas    | rules      |      1      |     1      |          1      |
| commas removed | BiLSTM-CRF |      0.95   |     0.9063 |          0.8545 |
| commas removed | rules      |      0.5758 |     0.3691 |          0      |
