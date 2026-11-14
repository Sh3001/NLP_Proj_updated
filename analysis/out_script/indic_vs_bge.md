## IndicBERTv2 (06e) vs bge-reranker-base full fine-tune (06b), India records

| script | records | F0.5 Indic | F0.5 bge | both right | only Indic right | only bge right | both wrong | AUC Indic | AUC bge | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|
| latin | 34342 | 0.96839 | 0.98775 | 33181 | 173 | 775 | 213 | 0.99897 | 0.99983 | 1.52e-91 |
| devanagari | 7868 | 0.9728 | 0.99352 | 7689 | 27 | 139 | 13 | 0.99937 | 0.99996 | 2.27e-19 |
| malayalam | 10371 | 0.97782 | 0.99146 | 10168 | 35 | 149 | 19 | 0.99963 | 0.99996 | 6e-18 |
| other_indic | 283 | 0.97153 | 0.94412 | 264 | 7 | 6 | 6 | 0.99878 | 0.99607 | 1.0 |
