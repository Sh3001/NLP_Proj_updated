# Entity Resolution Live Demo & MoE Artifacts (Issue #15)

This directory contains the documentation, presentation script, fallback HTML, report section, and slide deck for **GitHub Issue #15: [Ayush] Live demo (08_demo) + report/slides: Mixture of Experts and results**.

---

## 📂 Deliverables Directory

| File | Purpose | Audience / Use |
|---|---|---|
| `demo/README.md` | Complete documentation on running the demo, dependencies, artifacts, and examples | Team & Graders |
| `demo/demo_fallback.html` | Self-contained HTML fallback showing all 4 demo records rendered with pipeline stages and model predictions | Standalone offline presentation if Gradio link fails |
| `demo/demo_script_5min.md` | 5-minute timed presentation script with cell-by-cell execution instructions | Ayush (live demo presenter) |
| `demo/report_section_moe_results.md` | Full academic report section covering MoE architecture (06d, 07) and experimental results | Keshav (Report assembler) |
| `demo/slides_moe_results.md` | 2–3 slide designs with diagrams, comparison tables, and speaker notes | Akshat (Slide deck assembler) |

---

## 🚀 How to Launch the Demo

### 1. Interactive Execution (Recommended: Kaggle or Google Colab GPU T4)

1. Open `kaggle/notebooks/08_demo.ipynb` on **Kaggle** or **Google Colab**.
2. **Environment Settings:**
   - **Accelerator:** GPU T4 (or CPU for degraded inference)
   - **Internet:** **ON** (required to download Hugging Face models: `bge-reranker-base`, `IndicBERTv2`, `multilingual-e5-small`)
3. **Attach Required Datasets / Artifacts:**
   - Dataset: `akshatiit/er-small` (contains `data/small` splits: `train/`, `val/`, `test/`, `test_labels/`)
   - Candidates: `akshatiit/er-03-output` (contains candidate pairs in `prep/`, blocking metadata, fine-tuned e5)
   - Model outputs from previous stages (`er-02-output` through `er-07-output` or attached notebook versions):
     - `models/parser_bilstm_crf.pt` (from 02)
     - `models/e5_finetuned/` (from 03)
     - `models/gbm.txt`, `gbm_meta.json` (from 04)
     - `models/*.rnn.pt` (from 05a–c)
     - `models/*.ce.pt` (from 06a–f)
     - `models/moe_fusion.pt` (from 07)
4. Click **Run All** (in interactive session; **not** *Save & Run All*).
5. Watch the **live status table** printed in the initialization cell to confirm which models are loaded as `live` vs. falling back to `stored test scores`.
6. Use the **Gradio link** (`https://....gradio.live`) generated at the bottom of the notebook to interact with the UI, or use the inline notebook cells.

---

## 📦 Required Dependencies

All dependencies are installed automatically by cell 1 of `08_demo.ipynb` if missing:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install anyascii scikit-learn transformers "rapidfuzz>=3.6" lightgbm pytorch-crf peft sentencepiece protobuf gradio
```

---

## 🎯 The Four Selected Presentation Records

The notebook contains a deterministic selector (`pick_demo4()`, seed 42) that extracts four records representing key challenge categories from the test split:

1. **Record 1: Easy US · Latin Script**
   - *Example:* US entity with clean Latin name and full address.
   - *Behavior:* TF-IDF and e5 bi-encoder both rank the true S1 at rank 1. All matchers (LightGBM, RNN, cross-encoders) agree with high confidence.
2. **Record 2: Indic Script (Devanagari or Malayalam)**
   - *Example:* Indian business in Devanagari/Malayalam script.
   - *Behavior:* Tests NFKC normalization, romanization, and the learned transliteration lexicon. The Gated MoE detects `script = devanagari / malayalam` and up-weights `IndicBERTv2`.
3. **Record 3: Sibling Branch / Distractor Case**
   - *Example:* Sibling branches sharing identical core business names but located at different addresses.
   - *Behavior:* String similarity models struggle with high token overlap; parsed address features (house number, postcode agreement) enable correct disambiguation.
4. **Record 4: Failure Case**
   - *Example:* Business with unexpanded acronym, missing address, or blocking recall failure.
   - *Behavior:* Shows where the best individual model fails, how the MoE gating responds, and analyzes the root cause (blocking ceiling).

---

## 🛡️ Fallback HTML: Viewing Without Gradio

If the Gradio live session cannot be established or internet disconnects during the presentation:
- Open [`demo/demo_fallback.html`](demo_fallback.html) directly in any web browser.
- The file is **100% self-contained** (CSS, table layouts, prediction badges, gating weights embedded).
- In addition, running `08_demo.ipynb` automatically saves an updated `demo_fallback.html` reflecting the specific session's live model outputs in `/kaggle/working/demo_fallback.html`.

---

## 🤝 Team Handover

- **To Akshat:** Please import the slides from [`demo/slides_moe_results.md`](slides_moe_results.md) into the final slide deck.
- **To Keshav:** Please integrate the report section from [`demo/report_section_moe_results.md`](report_section_moe_results.md) into the final report document.
- **To Ayush:** Follow [`demo/demo_script_5min.md`](demo_script_5min.md) during the live presentation.

