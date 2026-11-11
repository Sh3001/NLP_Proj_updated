| model | section | test F0.5 | US Latin | India Latin | India Devanagari | India Malayalam | India Other indic |
|---|---|---|---|---|---|---|---|
| GBM baseline | 5 Classical baseline | 0.99081 | 0.98885 | 0.98596 | 0.99241 | 0.99087 | 1.0 |
| BiLSTM | 6 RNN / LSTM | 0.98335 | 0.983 | 0.97542 | 0.98307 | 0.97807 | 0.90482 |
| BiGRU | 6 RNN / LSTM | 0.98197 | 0.9804 | 0.97521 | 0.98117 | 0.97787 | 0.89607 |
| BiRNN | 6 RNN / LSTM | 0.97475 | 0.97521 | 0.96409 | 0.96944 | 0.97 | 0.90901 |
| mDeBERTa-v3 full FT (fp32) | 7 Transformers | 0.99402 | 0.99218 | 0.99014 | 0.99692 | 0.99699 | 0.98149 |
| bge full FT | 7 Transformers | 0.99124 | 0.98874 | 0.98775 | 0.99352 | 0.99146 | 0.94412 |
| bge LoRA r=16 | 7 Transformers | 0.98708 | 0.98485 | 0.9824 | 0.98886 | 0.98618 | 0.95782 |
| IndicBERTv2 full FT | 7 Transformers | 0.97611 | 0.97118 | 0.96839 | 0.9728 | 0.97782 | 0.97153 |
| bge zero-shot | 7 Transformers | 0.91674 | 0.92417 | 0.8944 | 0.8826 | 0.88574 | 0.96146 |
| Ditto bge DK+SU+DA-all 40k | 7b Ditto ablation | 0.98214 | 0.98057 | 0.97441 | 0.98108 | 0.97867 | 0.93049 |
| Ditto xlmr DA-all 40k | 7b Ditto ablation | 0.98172 | 0.98039 | 0.97716 | 0.98141 | 0.97436 | 0.95673 |
| Ditto xlmr DK 40k | 7b Ditto ablation | 0.98028 | 0.97789 | 0.97032 | 0.97705 | 0.97643 | 0.93097 |
| Ditto xlmr plain 40k | 7b Ditto ablation | 0.9788 | 0.97431 | 0.97696 | 0.97714 | 0.97321 | 0.94077 |
| Ditto xlmr DK+SU+DA-all 40k | 7b Ditto ablation | 0.97792 | 0.97822 | 0.96357 | 0.96734 | 0.96903 | 0.93182 |
| Ditto xlmr SU 40k | 7b Ditto ablation | 0.97541 | 0.97342 | 0.9677 | 0.96906 | 0.96769 | 0.94172 |
| Gated Mixture of Experts | 8 Fusion | 0.99481 | 0.99263 | 0.99219 | 0.99635 | 0.99748 | 0.99746 |
| Logistic-regression stacking | 8 Fusion | 0.99477 | 0.99282 | 0.99089 | 0.99611 | 0.99734 | 0.99109 |
| MoE, constant gate (ablation) | 8 Fusion | 0.99474 | 0.99258 | 0.99175 | 0.99643 | 0.99719 | 0.99109 |
| LightGBM stacking (same inputs) | 8 Fusion | 0.9947 | 0.99264 | 0.99115 | 0.99693 | 0.99764 | 0.98346 |
| Mean of 9 calibrated experts | 8 Fusion | 0.99286 | 0.99158 | 0.98896 | 0.99355 | 0.99464 | 0.97386 |

Records per group (test S2/S3): US Latin 48,710, India Latin 34,342, India Devanagari 7,868, India Malayalam 10,371, India Other indic 283
