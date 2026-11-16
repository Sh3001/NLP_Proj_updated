# Slides: Transformer cross-encoders and Ditto
### Shivam · issues #10, #12, #20 → hand to Akshat for the deck

Six slides: three on the cross-encoders, three on Ditto. Each has its content, the figure or table to use, and a line of
speaker notes. Full text: `report/latex/sections/09_cross_encoders.tex`. All numbers are test macro F0.5 on `data/small`
(threshold tuned on val, test scored once) from `final_results.csv` and `results/06g/`.

---

## SLIDE C1: A cross-encoder reads the pair, not two vectors

**Title:** Cross-encoders: both records in one input

```
[CLS]  name + romanised name | romanised address  (query)  [SEP]  candidate S1 record  [SEP]
   │
   └─► transformer (12 layers, full attention across both records) ──► [CLS] vector ──► linear ──► match logit
```

- Same recipe for every model: 160,000 candidate pairs from part A, 5,000 steps, batch 32, max length 96,
  binary cross-entropy, fp16 autocast, one T4. Only the backbone, learning rate and trained parameters change.
- The retriever (03) gives the top 5 candidates; the cross-encoder scores each pair; "best S1 if p ≥ threshold".
- Course topics: **Transformers** (attention over both records) and **fine-tuning**.

*Notes:* A bi-encoder (our e5 retriever) embeds each record alone, which is fast but cannot compare word by word. The
cross-encoder can, and it is the expensive step we only run on 5 candidates per record.

---

## SLIDE C2: Fine-tuning is what matters

**Title:** Zero-shot 0.9167 → fine-tuned 0.9912

| model | trained params | test F0.5 | P | R | train min |
|---|---|---|---|---|---|
| bge-reranker zero-shot (06a) | 0 | 0.9167 | 0.9161 | 0.9595 | 0 |
| **bge-reranker full fine-tune (06b)** | 278M (100 %) | **0.9912** | 0.9941 | 0.9878 | 19.4 |
| bge-reranker LoRA r=16 (06c) | 1.5M (0.5 %) | 0.9871 | 0.9900 | 0.9854 | 10.1 |
| mDeBERTa-v3 full fine-tune (06f) | 279M (100 %) | **0.9940** | 0.9953 | 0.9931 | 60.4 |

- Fine-tuning adds **+0.0745**. Zero-shot has high recall but low precision, and only **0.687** accuracy on singletons
  (it matches a business to its near-duplicate branch); after fine-tuning singleton accuracy is **0.984**.
- Figure to use: `results_test_f05.png` (test F0.5 per model).

*Notes:* The pre-trained reranker knows "these two texts are similar", not "these are the same branch". The 160k
labelled pairs teach it the second.

---

## SLIDE C3: Cost against accuracy: LoRA, and why mDeBERTa needs fp32

**Title:** LoRA keeps 94 % of the gain with 0.5 % of the weights

- **LoRA** (rank 16 on query / key / value, plus the head): 0.9871 vs 0.9912, **2.4× less GPU memory**, a 3 MB checkpoint
  instead of 1.1 GB, but only 1.9× less training time (the forward and backward still run through the whole backbone).
- **mDeBERTa-v3 is the best single model (0.9940)** but costs 3.1× the training time and 1.7× the memory of bge.
  It must train in **fp32**: the checkpoint is stored in fp16, so our first run trained in pure fp16, the loss was NaN
  from the first step and it predicted no matches (F0.5 0.055). Casting the master weights to fp32 fixed it.
- Not every change helps: the **MoE head** (06d, 0.9889) and **IndicBERTv2** (06e, 0.9761) are below plain bge fine-tuning.
- A **LightGBM** on string and retrieval features reaches 0.9908 in 4.6 min on a CPU, so the cross-encoders buy at most
  +0.003 over a strong classical model.

*Notes:* Last bullet is the honest message of the whole project: most of the gain comes from the retriever and
fine-tuning; fusion adds another +0.0008 on top of the best single model.

---

## SLIDE D1: Ditto: the same cross-encoder with three optimisations

**Title:** Ditto (Li et al., VLDB 2021) on our data

```
[COL] name [VAL] balaji care private limited [COL] house [VAL] [NUM] 110 [COL] locality [VAL] narela shankari [COL] city [VAL] bhopal
```

- **Serialization** with `[COL]` / `[VAL]` markers, fed to the language model as a sentence pair, `[CLS]` classified.
- **DK (domain knowledge):** the **BiLSTM-CRF parser from 02** types the address spans (house, street, locality,
  landmark, city, state, postcode); our transliteration lexicon and abbreviation expansion normalise them;
  numbers get a `[NUM]` marker.
- **SU (summarisation):** keep the highest-TF-IDF words when a record is too long. 5.8 % of pairs hit the 96-token
  limit without it, 3.6 % with it (Malayalam 22.6 % → 12.9 %).
- **DA (MixDA):** an augmented copy of each pair (span delete / shuffle, attribute delete / shuffle, entry swap);
  the two `[CLS]` vectors are interpolated, lam ~ Beta(0.8, 0.8), so the label stays valid.
- Course topics: **parsing → tokenization → transformer fine-tuning**, in one model.

*Notes:* Ditto's DK uses generic regexes in the paper. Our DK uses the project's own parser and lexicon: that is the
contribution on top of the reproduction.

---

## SLIDE D2: The ablation: domain knowledge helps, the rest doesn't

**Title:** DK is the one optimisation that holds up

Test F0.5, 40k training pairs per run, three complete runs (rule parser, BiLSTM-CRF parser ×2):

| run | rules | BiLSTM-CRF 1 | BiLSTM-CRF 2 |
|---|---|---|---|
| R1 plain | 0.9788 | 0.9782 | 0.9782 |
| **R2 +DK** | 0.9803 | **0.9823** | **0.9823** |
| R3 +SU | 0.9754 | 0.9774 | 0.9774 |
| R4 +DA | 0.9817 | 0.9761 | 0.9791 |
| R5 DK+SU+DA | 0.9779 | 0.9798 | 0.9808 |
| **R6 DK+SU+DA on bge** | 0.9821 | **0.9828** | **0.9834** |

- **DK: +0.0041** with our parser (twice with the same result), +0.0015 with rules. The parser makes the gain larger.
- **DA: no measurable effect** (0.9817, 0.9761, 0.9791 on the same configuration, around plain). **SU:** slightly negative.
- **The backbone matters more than the optimisations:** bge-reranker start (R6) is best every time.
- With a quarter of the data, Ditto (0.9821 to 0.9834) stays below the plain 06b fine-tune (0.9912). The 160k-pair
  160k-pair `final` run (DK+SU+DA, XLM-R, same data as 06b): **0.9876**, still 0.0036 below 06b (0.9912).

*Notes:* Say it as a negative result: on this data, a well fine-tuned cross-encoder leaves little room for Ditto's
tricks, except the one that uses our own parser. Single seed, so differences under about 0.003 are noise.

---

## SLIDE D3: Label efficiency: Ditto's tricks help most with few labels

**Title:** With 2,000 training pairs the full Ditto recipe is +0.013 better

Figure: `results/06g/label_efficiency_3points.png`

| training pairs | plain | DK+SU+DA | gain |
|---|---|---|---|
| 2,000 | 0.9466 | 0.9592 | **+0.0127** |
| 10,000 | 0.9622 | 0.9717 | +0.0095 |
| 40,000 | 0.9782 | 0.9808 | +0.0026 |

- The gain shrinks as labels grow: Ditto's low-resource claim holds on our data.
- Which part gives it? DK alone is +0.004 at 40k and DA alone does nothing, so the early gain is probably DK (our parser),
  but we did not run that split at 2k and 10k.
- Augmentation operators (20k pairs, no baseline run): span-level operators (span_shuffle 0.9708, span_del 0.9702) beat
  attribute-level ones (entry_swap 0.9638, attr_del 0.9633).

*Notes:* One seed. Say "consistent with the paper", not "proven".
