AUC: pair-level ROC AUC of the test logits. top-1: share of linked records whose true S1 is among the candidates and gets the highest score (no threshold).

| model | AUC US Latin | top-1 US Latin | AUC India Latin | top-1 India Latin | AUC India Devanagari | top-1 India Devanagari | AUC India Malayalam | top-1 India Malayalam | AUC India Other indic | top-1 India Other indic |
|---|---|---|---|---|---|---|---|---|---|---|
| BiLSTM | 0.9996 | 0.9967 | 0.99949 | 0.9948 | 0.99977 | 0.9993 | 0.99974 | 0.9974 | 0.99882 | 1.0 |
| BiGRU | 0.99956 | 0.9958 | 0.99941 | 0.995 | 0.99971 | 0.9989 | 0.9997 | 0.9976 | 0.99633 | 1.0 |
| BiRNN | 0.99917 | 0.995 | 0.99864 | 0.9924 | 0.99935 | 0.9981 | 0.99937 | 0.9966 | 0.99652 | 1.0 |
| IndicBERTv2 full FT | 0.99829 | 0.9867 | 0.99897 | 0.9886 | 0.99937 | 0.9968 | 0.99963 | 0.9961 | 0.99878 | 0.9962 |
