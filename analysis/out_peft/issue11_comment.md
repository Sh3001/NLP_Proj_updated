06c and 06f are done (full runs, `QUICK_RUN = False`, T4).

**Outputs (public Kaggle datasets)**
- `er-06c-output`: https://www.kaggle.com/datasets/xenoncipher/er-06c-output (LoRA adapters only, 3 MB)
- `er-06f-output`: https://www.kaggle.com/datasets/xenoncipher/er-06f-output (full fp32 weights, 1.1 GB)

Each zip has `results/`, `scores/` (B/val/test logits for 07), `preds/` and `models/*.ce.pt` (for 08).

**Setup (all cross-encoders):** 1 pass over 160,000 part-A candidate pairs (14.7% positive), batch 32, 5,000 steps, max length 96, BCE loss, threshold tuned on val, test scored once. Peak GPU memory is `torch.cuda.max_memory_allocated()` during training.

### Full fine-tuning vs LoRA vs zero-shot (held-out test split)

| model | trained params | val F0.5 | **test F0.5** | test P | test R | F0.5 India | F0.5 US | train min | peak GPU (GB) |
|---|---|---|---|---|---|---|---|---|---|
| bge-reranker-base zero-shot (06a, #10) | 0 | _pending_ | _pending_ | | | | | 0 | _pending_ |
| bge-reranker-base full fine-tune (06b, #10) | 278M (100%) | _pending_ | _pending_ | | | | | _pending_ | _pending_ |
| bge-reranker-base **LoRA r=16** (06c) | **1.48M (0.53%)** | 0.99142 | **0.98708** | 0.9900 | 0.9854 | 0.98839 | 0.98566 | 10.1 | **2.26** |
| mDeBERTa-v3-base full fine-tune, fp32 (06f) | 278.8M (100%) | 0.99561 | **0.99402** | 0.9953 | 0.9931 | 0.99533 | 0.99261 | 60.4 | 9.31 |

Shivam: the 06a/06b rows need your `results/*.json` (`val_f05`, `test_f05`, `test_P`, `test_R`, per-country, `train_min`, `gpu_mem_gb`). 06b is the fair comparison for LoRA: same backbone, same data, same AMP.

### LoRA (06c)

- **What is trained:** rank-16 adapters on query / key / value of every attention layer, plus the classification head:
  - 12 layers × 3 × 2 × 768 × 16 = 0.88M adapter weights;
  - 0.59M head weights.

  The 278M backbone weights stay frozen. In total 1.48M parameters (0.53%) are trained, so the saved model is **3 MB** instead of about 1.1 GB.
- **Memory:** 2.26 GB at its peak. Full fine-tuning with AdamW keeps a gradient and two Adam moments in fp32 for every weight, besides the weight itself: 16 bytes × 278M ≈ 4.2 GB before any activations. LoRA keeps those only for 1.48M weights.
- **Time:** LoRA does not save much training time. The forward and backward passes still go through the whole frozen backbone; only the weight-gradient computation for the frozen matrices is skipped. Most of the 10.1 min is that backbone compute.
- **Reproducible:** the rerun (to measure GPU memory) gave exactly the same scores as the first run (seed 42).
- The full-fine-tune vs LoRA gap in F0.5 will be filled in from 06b.

### mDeBERTa-v3 (06f), and why it runs in fp32

- **Best single model so far:** test F0.5 0.99402, ahead of LightGBM 04 (0.99081) and LoRA (0.98708). It is the best on both countries, most of all on India (0.9953).
- **Why fp32:** the T4 (Turing) has no bf16 support, so the only fast precision there is fp16, and DeBERTa-v3 is known to be numerically unstable in fp16. For that reason 06f is configured without AMP (`amp=False`).
- **What went wrong in our first run:** transformers v5 loads a model in the dtype its checkpoint is stored in, and the `microsoft/mdeberta-v3-base` checkpoint is stored in **fp16**. (bge-reranker, IndicBERTv2 and e5 are stored in fp32.) So, without AMP, the whole model trained in *pure* fp16: fp16 weights, gradients and AdamW moments, no fp32 master copy, no loss scaling.
  - In fp16, small squared gradients in Adam's second moment underflow to 0, and AdamW's ε = 1e-8 is itself 0 in fp16 (smallest fp16 value ≈ 6e-8). So the update divides by zero.
  - Also, an update of lr 2e-5 is below fp16's resolution for most weights.
  - The loss was **NaN from the first logged step** (step 500) to the end. The model then scored test F0.5 0.055, i.e. it predicted no matches at all.
- **Fix (#19):** `.float()` right after `from_pretrained`, so weights, gradients and optimizer state are fp32. The cross-encoder loop also now stops with an error on a non-finite loss, instead of scoring a broken model. After the fix the loss trains normally (0.0049 at step 500, 0.0002 at step 5000).
- **Cost of fp32:**
  - training took 60.4 min, against 17.2 min for the broken fp16 run (3.5× slower);
  - scoring B + val + test took another 89 min;
  - peak memory was 9.31 GB, against 2.26 GB for LoRA.
