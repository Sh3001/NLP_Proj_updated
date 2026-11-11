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
| **Keshav** | `02_parser_bilstm_crf`, `04_gbm_baseline`, `06c_ce_bge_lora`, `06f_ce_mdeberta` | Parsing, LSTM (BiLSTM-CRF), edit distance (classical baseline), parameter-efficient fine-tuning (LoRA) | Assembles the **final report** |
| **Rishi** | `05a/b/c_siamese_*`, `06e_ce_indicbert` | RNN, LSTM, GRU; Fine-tuning (Indic backbone) | Per-script error analysis across all models |
| **Shivam** | `06a`, `06b` (cross-encoder zero-shot vs full fine-tune), Ditto-style cross-encoder | Transformers, pretraining + fine-tuning (Ditto, [Li et al. 2020](https://arxiv.org/abs/2004.00584)) | Keeps the **README results table** and repo up to date |
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

## Priority if time runs short

GPU time is limited, so run the notebooks in this order. The first tier alone gives the full story
(classical → RNN → zero-shot transformer → fine-tuned transformer → MoE fusion).

| Tier | Notebooks | Why |
|---|---|---|
| 1, must have | 03, 01, 04, 05a, 06a, 06b, 07 | One model per course block, plus the final table |
| 2, should have | 02, 05b, 06c, 06d, Ditto variant | Parsing, LSTM vs GRU, LoRA vs full, MoE head |
| 3, if time allows | 05c, 06e, 06f (1.5–2.5 h), live 08 | Extra comparisons; 08 can fall back to stored test scores |

## Shared Kaggle datasets

| Dataset | Contents | Attach to |
|---|---|---|
| [`akshatiit/er-small`](https://www.kaggle.com/datasets/akshatiit/er-small) | `data/small` splits (train / val / test / test_labels), 68 MB | every notebook |
| [`akshatiit/er-03-output`](https://www.kaggle.com/datasets/akshatiit/er-03-output) | output of `03_candidates`: candidate pairs (`prep/`), blocking numbers, fine-tuned e5, 420 MB | 04, 05a–c, 06a–f, 07, 08 |
| [`xenoncipher/er-02-output`](https://www.kaggle.com/datasets/xenoncipher/er-02-output) | output of `02_parser_bilstm_crf`: `prep/parsed_fields.parquet`, parser model | 04, 08 |
| [`xenoncipher/er-04-output`](https://www.kaggle.com/datasets/xenoncipher/er-04-output) | output of `04_gbm_baseline` (run with 02's parsed fields): scores, results, LightGBM model | 07, 08 |
| [`xenoncipher/er-06c-output`](https://www.kaggle.com/datasets/xenoncipher/er-06c-output) | output of `06c_ce_bge_lora`: scores, results, LoRA adapters | 07, 08 |
| [`xenoncipher/er-06f-output`](https://www.kaggle.com/datasets/xenoncipher/er-06f-output) | output of `06f_ce_mdeberta`: scores, results, fine-tuned weights | 07, 08 |
| [`ayush122222/notebook7ebbf31256`](https://www.kaggle.com/code/ayush122222/notebook7ebbf31256) (`er-06d-output`) | output of `06d_ce_bge_moe_head`: BGE + MoE head weights & gate statistics | 07, 08 |
| [`ayush122222/notebook2a4366b080`](https://www.kaggle.com/code/ayush122222/notebook2a4366b080) (`er-07-output`) | output of `07_moe_fusion_and_results`: gated MoE fusion model, master results & plots | 08 |

Both are public. In your notebook: **Add Input → Datasets**, search for `er-small` / `er-03-output` (or
paste the link). No upload and no permission needed. Your Kaggle account must be phone-verified, or
*Internet* stays off and the Hugging Face downloads fail.

Your own `er-<nn>-output` datasets (see the rules below) should also be public, so the rest of the team can attach them.

## Rules for every run

* Use the notebooks from this repo (`kaggle/notebooks/`), **re-imported**. Older copies don't save model weights.
* Run with `QUICK_RUN = True` once (a few minutes), then with `QUICK_RUN = False` (the real run).
* Settings: GPU T4 (CPU for 01, 04, 07), Internet on.
* When it finishes, download `out_<nn>.zip` from the Output panel. Upload it as a Kaggle Dataset named
  `er-<nn>-output` and **make it public** (Dataset → Settings → Sharing), then add its link to the table above.
* Post the result numbers (val/test F0.5, P, R, per-country, training minutes) as a comment on your issue.
* Thresholds are tuned on **val**; **test** is scored once. Do not tune anything on test.

## How to run

See [kaggle/README.md](kaggle/README.md).
