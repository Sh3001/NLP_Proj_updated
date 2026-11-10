06c and 06f are done (full runs, `QUICK_RUN = False`, T4).

**Outputs (public Kaggle datasets)**
- `er-06c-output`: https://www.kaggle.com/datasets/xenoncipher/er-06c-output (LoRA adapters only, 3 MB)
- `er-06f-output`: https://www.kaggle.com/datasets/xenoncipher/er-06f-output (full fp32 weights, 1.1 GB)

Each zip has `results/`, `scores/` (B/val/test logits for 07), `preds/` and `models/*.ce.pt` (for 08).

**Setup (all cross-encoders):** 1 pass over 160,000 part-A candidate pairs (14.7% positive), batch 32, 5,000 steps, max length 96, BCE loss, threshold tuned on val, test scored once. Peak GPU memory is `torch.cuda.max_memory_allocated()` during training.

### Full fine-tuning vs LoRA vs zero-shot (held-out test split)

| model | trained params | val F0.5 | **test F0.5** | test P | test R | F0.5 India | F0.5 US | train min | peak GPU (GB) |
|---|---|---|---|---|---|---|---|---|---|
| bge-reranker-base zero-shot (06a, #10) | 0 | 0.94192 | **0.91674** | 0.9161 | 0.9595 | 0.90712 | 0.92713 | 0 | 1.04 |
| bge-reranker-base full fine-tune (06b, #10) | 278M (100%) | 0.99338 | **0.99124** | 0.9941 | 0.9878 | 0.99299 | 0.98935 | 19.4 | 5.53 |
| bge-reranker-base **LoRA r=16** (06c) | **1.48M (0.53%)** | 0.99142 | **0.98708** | 0.9900 | 0.9854 | 0.98839 | 0.98566 | 10.1 | **2.26** |
| mDeBERTa-v3-base full fine-tune, fp32 (06f) | 278.8M (100%) | 0.99561 | **0.99402** | 0.9953 | 0.9931 | 0.99533 | 0.99261 | 60.4 | 9.31 |

06a/06b numbers are from Shivam's `results/06a`, `results/06b` on main (a999fa8). 06a–06c share the backbone, the training data (06b/06c) and AMP, so they compare directly. 06f is a different backbone and runs without AMP.

### Full fine-tuning vs LoRA vs zero-shot (same backbone)

- **Fine-tuning is what matters:** zero-shot → full fine-tune adds **+0.0745** test F0.5. Zero-shot has high recall (0.96) but low precision (0.92), because without training it can't separate a business from its near-duplicate branches.
- **LoRA keeps 94% of that gain with 0.53% of the weights:** 0.98708 vs 0.99124, a gap of **0.0042**. The gap is consistent rather than noise: full fine-tuning is ahead in every script group with enough records (US Latin, India Latin, Devanagari, Malayalam; Rishi's per-script table in #8).
- **What LoRA saves:** 2.4× less GPU memory (2.26 vs 5.53 GB) and a 3 MB vs about 1.1 GB checkpoint, but only 1.9× less training time (10.1 vs 19.4 min). On a T4 with 16 GB, full fine-tuning of this 278M model fits anyway, so here LoRA is the cheaper option, not the necessary one.

### LoRA (06c)

- **What is trained:** rank-16 adapters on query / key / value of every attention layer, plus the classification head:
  - 12 layers × 3 × 2 × 768 × 16 = 0.88M adapter weights;
  - 0.59M head weights.

  The 278M backbone weights stay frozen. In total 1.48M parameters (0.53%) are trained, so the saved model is **3 MB** instead of about 1.1 GB.
- **Memory:** 2.26 GB at its peak. Full fine-tuning with AdamW keeps a gradient and two Adam moments in fp32 for every weight, besides the weight itself: 16 bytes × 278M ≈ 4.2 GB before any activations. LoRA keeps those only for 1.48M weights.
- **Time:** LoRA does not save much training time. The forward and backward passes still go through the whole frozen backbone; only the weight-gradient computation for the frozen matrices is skipped. Most of the 10.1 min is that backbone compute.
- **Reproducible:** the rerun (to measure GPU memory) gave exactly the same scores as the first run (seed 42).

### mDeBERTa-v3 (06f), and why it runs in fp32

- **Best single model:** test F0.5 0.99402, ahead of bge full fine-tune 06b (0.99124), LightGBM 04 (0.99081) and LoRA (0.98708). It is the best on both countries, most of all on India (0.9953).
- **Why fp32:** the T4 (Turing) has no bf16 support, so the only fast precision there is fp16, and DeBERTa-v3 is known to be numerically unstable in fp16. For that reason 06f is configured without AMP (`amp=False`).
- **What went wrong in our first run:** transformers v5 loads a model in the dtype its checkpoint is stored in, and the `microsoft/mdeberta-v3-base` checkpoint is stored in **fp16**. (bge-reranker, IndicBERTv2 and e5 are stored in fp32.) So, without AMP, the whole model trained in *pure* fp16: fp16 weights, gradients and AdamW moments, no fp32 master copy, no loss scaling.
  - In fp16, small squared gradients in Adam's second moment underflow to 0, and AdamW's ε = 1e-8 is itself 0 in fp16 (smallest fp16 value ≈ 6e-8). So the update divides by zero.
  - Also, an update of lr 2e-5 is below fp16's resolution for most weights.
  - The loss was **NaN from the first logged step** (step 500) to the end. The model then scored test F0.5 0.055, i.e. it predicted no matches at all.
- **Fix (#19):** `.float()` right after `from_pretrained`, so weights, gradients and optimizer state are fp32. The cross-encoder loop also now stops with an error on a non-finite loss, instead of scoring a broken model. After the fix the loss trains normally (0.0049 at step 500, 0.0002 at step 5000).
- **Cost of fp32:**
  - training took 60.4 min, against 17.2 min for the broken fp16 run (3.5× slower);
  - scoring B + val + test took another 89 min;
  - against the similar-sized bge full fine-tune with AMP (06b), it takes 3.1× the training time (60.4 vs 19.4 min) and 1.7× the GPU memory (9.31 vs 5.53 GB). Part of that is fp32 and part is DeBERTa's heavier disentangled attention; these runs can't separate the two.
