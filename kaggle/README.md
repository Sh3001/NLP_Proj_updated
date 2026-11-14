# Kaggle notebooks: NLP course project (reduced dataset `data/small`)

There is one notebook per method. Every method scores **the same candidate pairs**, uses the **same
decision rule** (each S2/S3 record goes to its best S1 if p ≥ threshold), has its **threshold tuned on
val**, and is scored **once on the held-out test split** with the competition metric (macro F0.5).
Test labels are read only inside the final scoring function.

| Notebook | Course topic | GPU | Needs as input |
|---|---|---|---|
| `01_tokenization` | Tokenization | – | dataset |
| `02_parser_bilstm_crf` | Parsing, LSTM | T4 | dataset |
| `03_candidates` | Transformers, fine-tuning (bi-encoder) | T4 | dataset |
| `04_gbm_baseline` | classical baseline | – | dataset, 03, (02) |
| `05a_siamese_bilstm` | LSTM | T4 | dataset, 03 |
| `05b_siamese_bigru` | RNN (GRU) | T4 | dataset, 03 |
| `05c_siamese_birnn` | RNN (vanilla) | T4 | dataset, 03 |
| `06a_ce_bge_zeroshot` | Transformers (no training) | T4 | dataset, 03 |
| `06b_ce_bge_full_finetune` | Fine-tuning (full) | T4 | dataset, 03 |
| `06c_ce_bge_lora` | Fine-tuning (LoRA) | T4 | dataset, 03 |
| `06d_ce_bge_moe_head` | Mixture of Experts (top-2 MoE head) | T4 | dataset, 03 |
| `06e_ce_indicbert` | Fine-tuning (IndicBERTv2) | T4 | dataset, 03 |
| `06f_ce_mdeberta` *(optional, slow)* | Fine-tuning (mDeBERTa, fp32) | T4 | dataset, 03 |
| `06g_ditto` | **Ditto** (Li et al. 2021): Transformers, Fine-tuning, DK / SU / DA ablation | T4 | dataset, 03, 02 (recommended); `final` also needs the `ablation` output |
| `07_moe_fusion_and_results` | Mixture of Experts (gated fusion) + final table | – | dataset, 03, all of 04–06 (01, 02 optional) |
| `08_demo` | **Live demo**: one record through all five steps | T4 | dataset, 02, 03, 04, 05a–c, 06a–g, 07 |

All models are MIT or Apache-2.0: `multilingual-e5-small` (118M), `bge-reranker-base` (278M),
`IndicBERTv2-MLM-only` (278M), `mdeberta-v3-base` (279M), and the mBERT tokenizer (Apache-2.0).

## 1. Attach the shared datasets (no upload needed)

The team's data is already public on Kaggle:

* **`er-small`**, the `data/small` splits: https://www.kaggle.com/datasets/akshatiit/er-small
* **`er-03-output`**, output of `03_candidates` (candidate pairs, blocking numbers, fine-tuned e5):
  https://www.kaggle.com/datasets/akshatiit/er-03-output

In a notebook: **Add Input → Datasets**, search for the name or paste the link. The notebooks find the
folder that contains `train/train_source1.tsv` by themselves, wherever it is under `/kaggle/input`.

*Rebuilding the data yourself (optional):* run `python3 src/make_splits.py` as in the main README
(it takes about 11 minutes on Kaggle's CPU), zip `data/small`, and upload it with *Datasets → New
Dataset → drag the zip in*.

## 2. Import and run the notebooks, in this order

For each notebook: *Create → New Notebook → File → Import Notebook* → pick the `.ipynb` from
`kaggle/notebooks/`. Then:

1. Right panel, **Settings**: *Accelerator* = **GPU T4 x2** (or *None* where the table says –),
   and *Internet* = **On** (models come from Hugging Face; a few pip packages are installed).
2. **Add Input** → *Datasets* → `er-small`. For every notebook that needs `03` (all of them except 01 and 02),
   also add **`er-03-output`**. For the later notebooks, add the other team members' public
   `er-<nn>-output` datasets (listed in [TEAM.md](../TEAM.md)), or attach a notebook version under
   *Your Work → Notebooks*.
3. *Save Version → Save & Run All (Commit)*. It runs in the background. Results end up in the
   version's **Output** tab, and that output is what you attach to the later notebooks.

Order: **01, 02, 03 first** (01 and 02 are independent of 03). Then **04, 05a–c, 06a–f** in any
order, even several at once (they only need 03). **07 last**, with every finished notebook
attached. It finds all scores and results by itself.

`QUICK_RUN = True` (configuration cell) runs a tiny sample (3,000 / 600 / 600 S1) in a few minutes,
to check that everything works before the real run. Run 03 in the same mode as the notebooks that
use it. 07 skips a model whose pair count does not match.

## 3. What each notebook writes (Output tab)

* `results/<method>.json`: val F0.5, **test F0.5**, precision, recall, per-country F0.5, singleton
  accuracy, parameters, training minutes.
* `scores/<method>_{B,val,test}.npy`: logits for every candidate pair (used by 07).
* `preds/<method>/matching_results.tsv`: test predictions in the competition format.
  `03` also writes `preds/candidate_pairs.tsv`.
* `01`: `tokenizer_stats.csv`, `tokenization_blocking.csv`. `02`: `parser_eval.csv`,
  `prep/parsed_fields.parquet`. `03`: `blocking.json` (recall, ceiling), `prep/*.parquet`.
  `06d`: MoE-head gate statistics.
  `05a–c`: `diagnostics/<method>.json` (per-epoch train/val loss, val F0.5, gradient norms, and the
  vanishing-gradient probe ‖∂h_T/∂x_(T−k)‖ by distance k). `python3 analysis/rnn_compare.py <unzipped outputs>`
  turns the three runs into the comparison table and plots.
* `07`: `final_results.csv` / `final_results.md` (the table for the report),
  `results_test_f05.png`, `moe_gates.png`, `moe_gate_by_script.csv`.

### Downloading the output

The last cell of every notebook packs its output into `out_<nn>.zip` in `/kaggle/working`. Click
refresh in the Output panel, hover over the zip, then *⋮ → Download*. `prep/` is included only by
03 (all of it) and 02 (`parsed_fields.parquet`), so the attached inputs never hold two copies.
Instead of attaching a notebook version, you can upload the zip as a Kaggle Dataset and attach that.
The notebooks find the files under `/kaggle/input` either way.

## Live demo (`08_demo`)

The demo follows one record through the whole pipeline: normalise + tokenize → parse (BiLSTM-CRF) →
retrieve (TF-IDF + e5) → score (every matcher) → fuse (gated MoE) → decision, checked against the
ground truth. It runs the **saved models** on the record, so it needs the `models/` folder that
02–07 now write:

| notebook | saves for the demo | size |
|---|---|---|
| 02 | `models/parser_bilstm_crf.pt` | ~15 MB |
| 03 | `models/e5_finetuned/` (fp16) | ~240 MB |
| 04 | `models/gbm.txt`, `gbm_meta.json` | ~5 MB |
| 05a–c | `models/<method>.rnn.pt` | ~2 MB each |
| 06a | `models/<method>.ce.pt` (settings only; the model comes from Hugging Face) | tiny |
| 06b, 06d, 06e | `models/<method>.ce.pt` (trained weights, fp16) | ~560 MB each |
| 06c | `models/<method>.ce.pt` (LoRA adapters + head only) | ~5 MB |
| 06f | `models/<method>.ce.pt` (fp32) | ~1.1 GB |
| 06g (`final` run only) | `models/<method>.ditto.pt` (fp16) | ~560 MB |
| 07 | `models/moe_fusion.pt` | tiny |

Notebooks that ran **before** this change have no `models/` folder. For those, the demo shows their
stored test scores (test records only) instead of running them. Rerun a notebook to make it live.
A typed-in record needs live models. The status table printed after loading shows what runs live.

How to run it:
1. Import `08_demo.ipynb`. Settings: **GPU T4**, **Internet on**.
2. Add the dataset and the outputs of 02–07 (notebook versions or uploaded zips) as inputs.
3. **Run All** in the interactive session (not *Save & Run All*). Loading takes a few minutes.
4. Open the public `https://….gradio.live` link printed by the last cell. It stays up while the
   session runs. Choose an example from the dropdown (grouped: easy US, Devanagari, Malayalam,
   TF-IDF wrong, sibling branch, no commas, distractor, hard), type a record, or paste a test id.
5. Without Gradio: `show(list(EXAMPLES)[k])` or `show(typed=(name, address, country, 'S2'))` in the
   cell before renders the same five panels in the notebook.
6. The notebook automatically saves a self-contained fallback HTML file (`demo_fallback.html`) covering
   the four curated demo records. See [`demo/README.md`](../demo/README.md) for the 5-minute presentation
   script, report section, slide deck, and offline fallback HTML.

## Ditto (`06g_ditto`)

Ditto serializes a record pair as `[COL] name [VAL] ... [COL] city [VAL] ...` and fine-tunes a pre-trained
language model on it. The notebook is an **ablation** of its three optimisations: **DK** (span
normalisation + the parser's typed address spans), **SU** (TF-IDF summarisation), **DA** (MixDA with 5
augmentation operators). Set `RUN_SET` in the configuration cell:

| `RUN_SET` | runs | pairs each | time (T4, estimate) |
|---|---|---|---|
| `ablation` | R1 plain, R2 +DK, R3 +SU, R4 +DA, R5 DK+SU+DA, R6 = R5 on bge | 40k | ~2 h |
| `operators` | each augmentation operator, and `all` | 20k | ~1.5 h |
| `label_eff` | R1 and R5 on 2k and 10k pairs | 2k, 10k | ~1 h |
| `final` | best ablation run by **val** F0.5 (attach the `ablation` output); saves scores + weights | 160k | ~1.5 h |
| `all` | all of the above, then `final` | | ~6 h |

Each run also scores the full val and test pair sets (~10 min on a T4), which is part of the times above.
Run the sets in separate sessions (even on separate accounts); `final` reads the ablation results
from the attached output, or set `DITTO_FINAL = 'full'` to skip the selection. Only `final` writes
`scores/` and `models/`, so 07 and 08 get exactly one Ditto expert. Extra outputs: `ditto_ablation.csv`
(R1–R6 + R0 = `06b`, with pair-level F1 and per-script F0.5), `ditto_label_efficiency.png`,
`ditto_lengths.csv` (token lengths and truncation rate with and without SU: use it to check that
`DITTO_LEN` and `DITTO_SU_WORDS` make SU matter), and the operator table.

## Rough run times (T4, full `data/small`)

These are estimates, not measurements: 01 ≈ 10–15 min · 02 ≈ 20–30 min · 03 ≈ 20–30 min · 04 ≈ 15–25 min (CPU) ·
05a–c ≈ 10–20 min each · 06a ≈ 10 min · 06b–e ≈ 30–50 min each · 06f ≈ 1.5–2.5 h · 07 ≈ 5–10 min.
In total that is well inside Kaggle's weekly GPU quota.

## Regenerating the notebooks

The notebooks are generated. Edit `build_kaggle.py`, then run `python3 kaggle/build_kaggle.py`.
`kaggle/scripts/*.py` hold the same code as plain Python (used for the local smoke test).
