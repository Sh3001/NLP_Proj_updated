# Five-Minute Demo Script — Business Entity Resolution

**Issue #15 · [Ayush] Live demo (08_demo) + report/slides: Mixture of Experts and results**

---

## ⏱ Timing Overview

| Segment | Duration |
|---------|----------|
| Introduction & project goal | ~45 s |
| Dataset & supported languages | ~30 s |
| Pipeline overview (5 steps) | ~45 s |
| Models at a glance | ~30 s |
| MoE architecture deep-dive | ~45 s |
| Four demo records (live) | ~90 s |
| Results table & key observations | ~30 s |
| Conclusion | ~15 s |
| **Total** | **~5 min** |

---

## Segment 1 — Introduction & Goal (0:00–0:45)

> *Open `08_demo.ipynb` on Kaggle in the interactive session. The status table should already be printed — confirm it shows "live" for the key models.*

**Say:**
"Our project is **Business Entity Resolution** — given a large database of businesses from multiple sources and countries, we have to decide which Source-2 and Source-3 records describe the same real-world entity as a Source-1 record. Think of it as deduplication at scale across languages.

The official competition used the full dataset — roughly 2 million Source-1 entities. For this course we work on a carefully stratified subset covering four states: Oregon and Utah in the US, and Kerala and Madhya Pradesh in India — different scripts, different address formats, different naming conventions.

We evaluate with **macro F0.5**, which weights precision twice as heavily as recall, because a wrong match is more damaging than a missed one."

---

## Segment 2 — Dataset & Languages (0:45–1:15)

> *Point to the status table, noting the script breakdown printed at startup.*

**Say:**
"The small dataset has about 21,500 Source-1 entities in the test split. Roughly half are US businesses with Latin-script names, and half are Indian businesses that may use Devanagari (Hindi, Marathi) or Malayalam script.

Non-Latin names go through a normalisation pipeline: Unicode NFKC, then romanisation via `anyascii`, then a learned transliteration lexicon aligned from linked record pairs in the training data. This matters because the same business can appear as 'श्री लक्ष्मी ट्रेडर्स' in one source and 'Sri Lakshmi Traders' in another."

---

## Segment 3 — Pipeline Overview (1:15–2:00)

> *Point to the five tabs in the Gradio app (or the five `show()` sections).*

**Say:**
"Every record goes through five steps:

1. **Normalise + tokenize** — abbreviation expansion, three tokenizers (mBERT WordPiece, XLM-R SentencePiece, char 3-grams).
2. **Parse the address** — a rule-based parser and a trained BiLSTM-CRF (notebook 02) tag each address token: house number, street, locality, city, state, postcode.
3. **Retrieve candidates** — TF-IDF char-3-gram cosine and a fine-tuned multilingual-e5-small bi-encoder each return top-10 candidates; reciprocal-rank fusion gives the final 5.
4. **Score** — every matcher (LightGBM, three Siamese RNNs, six cross-encoders) scores each candidate pair.
5. **Fuse + decide** — the Gated Mixture of Experts from notebook 07 combines the matchers; the decision rule picks the best candidate if its probability exceeds the val-tuned threshold."

---

## Segment 4 — Models at a Glance (2:00–2:30)

> *Optionally show the results bar chart from notebook 07 output if displayed.*

**Say:**
"We have eight matchers across four course modules:

- **Classical (04):** LightGBM on 30+ string and address features — fast, interpretable baseline.
- **RNN/LSTM (05a–c):** Character-level Siamese networks with BiLSTM, BiGRU, and vanilla BiRNN encoders.
- **Transformers (06a–f):** BGE cross-encoder zero-shot, full fine-tune, LoRA r=16, MoE head variant; then IndicBERTv2 fine-tuned on the Indic-script pairs; and mDeBERTa-v3 for the strongest multilingual signal.
- **Fusion (07):** A gated MoE that takes all matchers' logits as inputs."

---

## Segment 5 — MoE Architecture (2:30–3:15)

> *Show Panel 5 (Fuse + decide) for a live record — point to the gate-weight bars.*

**Say:**
"The gated Mixture of Experts is the core contribution of notebook 07.

**Architecture:** Each matcher produces a calibrated logit zᵢ. The gate network — a two-layer MLP with 11 input features — computes per-pair softmax weights w(x). The final score is:

> p = σ(Σᵢ wᵢ(x) · (aᵢ zᵢ + bᵢ) + c)

where aᵢ and bᵢ are learned per-expert affine calibration parameters, initialised from individual logistic fits so training starts at the 'mean of calibrated experts'.

**Gate inputs (11 features):** script one-hot (Latin / Devanagari / Malayalam / other), source S3 flag, address-empty flag, address-has-number flag, name length / 40, address length / 80, TF-IDF cosine, e5 cosine, fused rank / K, candidates / K.

**Key insight:** The gate never sees the country label. It routes pairs by script and data quality. Devanagari pairs get higher weight on IndicBERTv2; Latin pairs with addresses rely more on LightGBM and BGE.

We compare against: mean of calibrated experts, logistic-regression stacking, MoE with a constant gate (ablation), and LightGBM stacking on the same 11+N inputs."

---

## Segment 6 — Four Demo Records (3:15–4:45)

> *Run each record either via Gradio dropdown or `show(i=qi)` in sequence. Spend ~20-25s per record.*

### Record 1 — Easy US, Latin script
> `show(i=DEMO4[list(DEMO4)[0]])`

**Say:** "Record 1 is a straightforward US business — name and address in plain Latin script. TF-IDF retrieves the true S1 at rank 1. Every matcher gives a high probability; the gate allocates most weight to LightGBM and BGE full fine-tune. Decision: correct match."

### Record 2 — Devanagari / Malayalam script
> `show(i=DEMO4[list(DEMO4)[1]])`

**Say:** "Record 2 is an Indian business — the original name is in Devanagari or Malayalam. Watch the normalisation panel: anyascii romanises the script, the learned lexicon further aligns tokens to their English equivalents. The gate shifts weight toward IndicBERTv2, which was fine-tuned on these Indic pairs."

### Record 3 — Sibling branch / distractor
> `show(i=DEMO4[list(DEMO4)[2]])`

**Say:** "Record 3 is a harder case — a sibling branch or unlinked record appears among the top candidates with the same core business name but a different address. The matchers give moderate scores to both. The gate uses address features (house-number agreement, postcode) to push the right candidate to the top. This is where the parsed address fields add real value over string similarity alone."

### Record 4 — Failure case
> `show(i=DEMO4[list(DEMO4)[3]])`

**Say:** "Record 4 is a known failure. The best single matcher picks the wrong candidate — perhaps because the true S1 has an ambiguous abbreviation or the address is empty. The MoE gating softens the weight on the overconfident matcher and sometimes corrects the decision — but not always. This is an honest look at where the pipeline breaks."

---

## Segment 7 — Results & Key Observations (4:45–5:15)

> *Refer to the final_results table from notebook 07 output or the printed bar chart.*

**Say:**
"Our results on the held-out test split (macro F0.5):

- LightGBM baseline: solid — parsed address features carry a lot of signal.
- Siamese RNNs: slightly below LightGBM on precision-heavy F0.5 — they help recall but misfire more.
- BGE cross-encoder full fine-tune: strong — the transformer reads both records jointly.
- LoRA: comparable to full fine-tune at a fraction of the trained parameters.
- IndicBERTv2: best on Indian-script pairs; slightly weaker on US Latin — the trade-off the gate is designed to resolve.
- **Gated MoE: best overall** — consistently beats every single matcher and the constant-gate ablation, confirming that input-conditional gating adds value beyond simple calibrated averaging.

Two caveats: our splits are from training data, so F0.5 is optimistic compared with the leaderboard; and the small dataset is 3.5% of S1 entities, so absolute numbers don't transfer to the full model."

---

## Segment 8 — Conclusion (5:15–5:30)

**Say:**
"To summarise: Business entity resolution across languages is a retrieval + reranking problem. We showed that a Gated Mixture of Experts that routes by script and data quality, without ever seeing the country label, is the most robust fusion strategy for a multilingual, multi-source dataset. The fallback HTML and notebook are shared in the repo; the slides go to Akshat and Keshav for the final deck."

---

## Notebook Cell Execution Order During Presentation

```
# Already executed on session startup — just confirm status table printed
# Then for each demo record:
show(i=DEMO4[list(DEMO4)[0]])   # Record 1
show(i=DEMO4[list(DEMO4)[1]])   # Record 2
show(i=DEMO4[list(DEMO4)[2]])   # Record 3
show(i=DEMO4[list(DEMO4)[3]])   # Record 4
```

Or use the Gradio dropdown:
1. Select `Easy · US, Latin script · ...` → click **Run**
2. Select `Devanagari script · ...` → click **Run**
3. Select `Sibling branch · ...` → click **Run**
4. Select `Hard · ...` → click **Run**

