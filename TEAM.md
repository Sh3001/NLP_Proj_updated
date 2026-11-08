# Team and work split

Five members, three issues each. Every member owns:
1. **one or more notebooks to run on Kaggle** (the real run, `QUICK_RUN = False`)
2. **an analysis** of their results
3. **their section of the report and the slides**

On top of that, each member has one shared job: the shared Kaggle dataset, the report document, the slide
deck, the README results table, or the live demo.

| Member | Notebooks | Course topics | Shared job |
|---|---|---|---|
| **Akshat** | `01_tokenization`, `03_candidates` | Tokenization, Transformers (bi-encoder fine-tuning) | Shared Kaggle datasets; assembles the **slide deck** |
| **Keshav** | `02_parser_bilstm_crf`, `04_gbm_baseline` | Parsing, LSTM (BiLSTM-CRF), classical baseline | Assembles the **final report** |
| **Rishi** | `05a/b/c_siamese_*`, `06e_ce_indicbert` | RNN, LSTM, GRU; Fine-tuning (Indic backbone) | Per-script error analysis across all models |
| **Shivam** | `06a`, `06b`, `06c`, `06f` (cross-encoders) | Transformers, Fine-tuning (zero-shot vs full vs LoRA) | Keeps the **README results table** and repo up to date |
| **Ayush** | `06d_ce_bge_moe_head`, `07_moe_fusion_and_results`, `08_demo` | Mixture of Experts | Runs the **live demo** in the presentation |

## Order of work (dependencies)

```
Akshat: upload data/small  ──►  03_candidates  ──►  everyone's model notebooks (04, 05a–c, 06a–f)  ──►  07 (Ayush)  ──►  08 demo (Ayush)
                            └─►  01 (Akshat) and 02 (Keshav) need only the dataset: start immediately
```

* **Akshat's 03 run blocks everybody.** Run it first.
* 02 and 01 do not need 03, so Keshav and Akshat can start on them on day 1.
* 07 needs the outputs of 04, 05a–c and 06a–f.
* 08 needs 02–07 run **with the current notebooks**, which save model weights in `models/`.

## Rules for every run

* Use the notebooks from this repo (`kaggle/notebooks/`), **re-imported**. Older copies don't save model weights.
* Run with `QUICK_RUN = True` once (a few minutes), then with `QUICK_RUN = False` (the real run).
* Settings: GPU T4 (CPU for 01, 04, 07), Internet on.
* When it finishes, download `out_<nn>.zip` from the Output panel. Upload it as a Kaggle Dataset named
  `er-<nn>-output` and **share it with the team** (Dataset → Settings → Collaborators).
* Post the result numbers (val/test F0.5, P, R, per-country, training minutes) as a comment on your issue.
* Thresholds are tuned on **val**; **test** is scored once. Do not tune anything on test.

## How to run

See [kaggle/README.md](kaggle/README.md).
