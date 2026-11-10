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
| Ditto bge DK+SU+DA-all 40k | US Latin | 48710 | 111 | 375 | 541 | 35 | 2.11 | 35788 | 12922 |
| Ditto bge DK+SU+DA-all 40k | India Latin | 34342 | 127 | 200 | 472 | 57 | 2.33 | 25488 | 8854 |
| Ditto bge DK+SU+DA-all 40k | India Devanagari | 7868 | 4 | 61 | 50 | 2 | 1.46 | 5703 | 2165 |
| Ditto bge DK+SU+DA-all 40k | India Malayalam | 10371 | 7 | 87 | 74 | 2 | 1.62 | 7374 | 2997 |
| Ditto bge DK+SU+DA-all 40k | India Other indic | 283 | 0 | 0 | 21 | 0 | 7.42 | 266 | 17 |
| Ditto xlmr DA-all 40k | US Latin | 48710 | 102 | 328 | 744 | 35 | 2.41 | 35788 | 12922 |
| Ditto xlmr DA-all 40k | India Latin | 34342 | 108 | 224 | 348 | 57 | 1.98 | 25488 | 8854 |
| Ditto xlmr DA-all 40k | India Devanagari | 7868 | 2 | 83 | 28 | 2 | 1.44 | 5703 | 2165 |
| Ditto xlmr DA-all 40k | India Malayalam | 10371 | 5 | 139 | 73 | 2 | 2.09 | 7374 | 2997 |
| Ditto xlmr DA-all 40k | India Other indic | 283 | 0 | 0 | 13 | 0 | 4.59 | 266 | 17 |
| Ditto xlmr DK 40k | US Latin | 48710 | 101 | 231 | 1203 | 35 | 3.15 | 35788 | 12922 |
| Ditto xlmr DK 40k | India Latin | 34342 | 111 | 155 | 743 | 57 | 2.94 | 25488 | 8854 |
| Ditto xlmr DK 40k | India Devanagari | 7868 | 6 | 52 | 93 | 2 | 1.92 | 5703 | 2165 |
| Ditto xlmr DK 40k | India Malayalam | 10371 | 5 | 79 | 115 | 2 | 1.92 | 7374 | 2997 |
| Ditto xlmr DK 40k | India Other indic | 283 | 0 | 0 | 27 | 0 | 9.54 | 266 | 17 |
| Ditto xlmr plain 40k | US Latin | 48710 | 101 | 541 | 743 | 35 | 2.84 | 35788 | 12922 |
| Ditto xlmr plain 40k | India Latin | 34342 | 99 | 239 | 417 | 57 | 2.2 | 25488 | 8854 |
| Ditto xlmr plain 40k | India Devanagari | 7868 | 4 | 70 | 65 | 2 | 1.77 | 5703 | 2165 |
| Ditto xlmr plain 40k | India Malayalam | 10371 | 3 | 113 | 135 | 2 | 2.42 | 7374 | 2997 |
| Ditto xlmr plain 40k | India Other indic | 283 | 0 | 0 | 19 | 0 | 6.71 | 266 | 17 |
| Ditto xlmr DK+SU+DA-all 40k | US Latin | 48710 | 118 | 244 | 1140 | 35 | 3.08 | 35788 | 12922 |
| Ditto xlmr DK+SU+DA-all 40k | India Latin | 34342 | 151 | 147 | 1020 | 57 | 3.84 | 25488 | 8854 |
| Ditto xlmr DK+SU+DA-all 40k | India Devanagari | 7868 | 2 | 45 | 181 | 2 | 2.9 | 5703 | 2165 |
| Ditto xlmr DK+SU+DA-all 40k | India Malayalam | 10371 | 4 | 70 | 202 | 2 | 2.66 | 7374 | 2997 |
| Ditto xlmr DK+SU+DA-all 40k | India Other indic | 283 | 0 | 0 | 33 | 0 | 11.66 | 266 | 17 |
| Ditto xlmr SU 40k | US Latin | 48710 | 82 | 376 | 1349 | 35 | 3.71 | 35788 | 12922 |
| Ditto xlmr SU 40k | India Latin | 34342 | 115 | 255 | 784 | 57 | 3.36 | 25488 | 8854 |
| Ditto xlmr SU 40k | India Devanagari | 7868 | 1 | 93 | 111 | 2 | 2.61 | 5703 | 2165 |
| Ditto xlmr SU 40k | India Malayalam | 10371 | 5 | 131 | 154 | 2 | 2.8 | 7374 | 2997 |
| Ditto xlmr SU 40k | India Other indic | 283 | 0 | 0 | 24 | 0 | 8.48 | 266 | 17 |
