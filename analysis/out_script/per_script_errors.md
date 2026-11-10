| model | group | records | wrong match | false match | missed | of which blocking miss | error rate % | linked | unlinked |
|---|---|---|---|---|---|---|---|---|---|
| GBM baseline | US Latin | 48710 | 92 | 157 | 433 | 35 | 1.4 | 35788 | 12922 |
| GBM baseline | India Latin | 34342 | 71 | 107 | 269 | 57 | 1.3 | 25488 | 8854 |
| GBM baseline | India Devanagari | 7868 | 3 | 16 | 30 | 2 | 0.62 | 5703 | 2165 |
| GBM baseline | India Malayalam | 10371 | 2 | 24 | 35 | 2 | 0.59 | 7374 | 2997 |
| GBM baseline | India Other indic | 283 | 0 | 0 | 0 | 0 | 0.0 | 266 | 17 |
| BiLSTM | US Latin | 48710 | 114 | 339 | 409 | 35 | 1.77 | 35788 | 12922 |
| BiLSTM | India Latin | 34342 | 138 | 198 | 328 | 57 | 1.93 | 25488 | 8854 |
| BiLSTM | India Devanagari | 7868 | 5 | 64 | 40 | 2 | 1.39 | 5703 | 2165 |
| BiLSTM | India Malayalam | 10371 | 12 | 77 | 76 | 2 | 1.59 | 7374 | 2997 |
| BiLSTM | India Other indic | 283 | 0 | 0 | 31 | 0 | 10.95 | 266 | 17 |
| BiGRU | US Latin | 48710 | 130 | 394 | 435 | 35 | 1.97 | 35788 | 12922 |
| BiGRU | India Latin | 34342 | 129 | 211 | 362 | 57 | 2.04 | 25488 | 8854 |
| BiGRU | India Devanagari | 7868 | 5 | 65 | 58 | 2 | 1.63 | 5703 | 2165 |
| BiGRU | India Malayalam | 10371 | 13 | 90 | 57 | 2 | 1.54 | 7374 | 2997 |
| BiGRU | India Other indic | 283 | 0 | 0 | 37 | 0 | 13.07 | 266 | 17 |
| BiRNN | US Latin | 48710 | 147 | 546 | 533 | 35 | 2.52 | 35788 | 12922 |
| BiRNN | India Latin | 34342 | 184 | 354 | 498 | 57 | 3.02 | 25488 | 8854 |
| BiRNN | India Devanagari | 7868 | 10 | 100 | 80 | 2 | 2.41 | 5703 | 2165 |
| BiRNN | India Malayalam | 10371 | 15 | 139 | 86 | 2 | 2.31 | 7374 | 2997 |
| BiRNN | India Other indic | 283 | 0 | 0 | 30 | 0 | 10.6 | 266 | 17 |
| mDeBERTa-v3 full FT (fp32) | US Latin | 48710 | 79 | 118 | 213 | 35 | 0.84 | 35788 | 12922 |
| mDeBERTa-v3 full FT (fp32) | India Latin | 34342 | 81 | 36 | 134 | 57 | 0.73 | 25488 | 8854 |
| mDeBERTa-v3 full FT (fp32) | India Devanagari | 7868 | 1 | 11 | 7 | 2 | 0.24 | 5703 | 2165 |
| mDeBERTa-v3 full FT (fp32) | India Malayalam | 10371 | 2 | 14 | 5 | 2 | 0.2 | 7374 | 2997 |
| mDeBERTa-v3 full FT (fp32) | India Other indic | 283 | 0 | 0 | 7 | 0 | 2.47 | 266 | 17 |
| bge full FT | US Latin | 48710 | 88 | 138 | 436 | 35 | 1.36 | 35788 | 12922 |
| bge full FT | India Latin | 34342 | 87 | 73 | 226 | 57 | 1.12 | 25488 | 8854 |
| bge full FT | India Devanagari | 7868 | 1 | 15 | 24 | 2 | 0.51 | 5703 | 2165 |
| bge full FT | India Malayalam | 10371 | 5 | 21 | 28 | 2 | 0.52 | 7374 | 2997 |
| bge full FT | India Other indic | 283 | 0 | 0 | 13 | 0 | 4.59 | 266 | 17 |
| bge LoRA r=16 | US Latin | 48710 | 95 | 240 | 565 | 35 | 1.85 | 35788 | 12922 |
| bge LoRA r=16 | India Latin | 34342 | 96 | 164 | 265 | 57 | 1.53 | 25488 | 8854 |
| bge LoRA r=16 | India Devanagari | 7868 | 1 | 46 | 23 | 2 | 0.89 | 5703 | 2165 |
| bge LoRA r=16 | India Malayalam | 10371 | 3 | 60 | 43 | 2 | 1.02 | 7374 | 2997 |
| bge LoRA r=16 | India Other indic | 283 | 0 | 0 | 12 | 0 | 4.24 | 266 | 17 |
| IndicBERTv2 full FT | US Latin | 48710 | 275 | 305 | 1192 | 35 | 3.64 | 35788 | 12922 |
| IndicBERTv2 full FT | India Latin | 34342 | 216 | 197 | 575 | 57 | 2.88 | 25488 | 8854 |
| IndicBERTv2 full FT | India Devanagari | 7868 | 15 | 38 | 99 | 2 | 1.93 | 5703 | 2165 |
| IndicBERTv2 full FT | India Malayalam | 10371 | 23 | 48 | 97 | 2 | 1.62 | 7374 | 2997 |
| IndicBERTv2 full FT | India Other indic | 283 | 0 | 0 | 12 | 0 | 4.24 | 266 | 17 |
| bge zero-shot | US Latin | 48710 | 193 | 2322 | 1052 | 35 | 7.32 | 35788 | 12922 |
| bge zero-shot | India Latin | 34342 | 313 | 1866 | 1178 | 57 | 9.78 | 25488 | 8854 |
| bge zero-shot | India Devanagari | 7868 | 77 | 738 | 59 | 2 | 11.11 | 5703 | 2165 |
| bge zero-shot | India Malayalam | 10371 | 63 | 904 | 107 | 2 | 10.36 | 7374 | 2997 |
| bge zero-shot | India Other indic | 283 | 0 | 0 | 16 | 0 | 5.65 | 266 | 17 |
