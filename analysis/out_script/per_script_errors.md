| model | group | records | wrong match | false match | missed | of which blocking miss | error rate % | linked | unlinked |
|---|---|---|---|---|---|---|---|---|---|
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
| IndicBERTv2 full FT | US Latin | 48710 | 275 | 305 | 1192 | 35 | 3.64 | 35788 | 12922 |
| IndicBERTv2 full FT | India Latin | 34342 | 216 | 197 | 575 | 57 | 2.88 | 25488 | 8854 |
| IndicBERTv2 full FT | India Devanagari | 7868 | 15 | 38 | 99 | 2 | 1.93 | 5703 | 2165 |
| IndicBERTv2 full FT | India Malayalam | 10371 | 23 | 48 | 97 | 2 | 1.62 | 7374 | 2997 |
| IndicBERTv2 full FT | India Other indic | 283 | 0 | 0 | 12 | 0 | 4.24 | 266 | 17 |
