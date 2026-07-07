# Fine-Tuning Explanation

## 1. Why full fine-tuning is expensive

Full fine-tuning updates every parameter in the model. Even for a "small" 0.5B–1B parameter model, that means storing full-precision gradients and optimizer states (Adam keeps two extra moving-average tensors per parameter) for every single weight, which multiplies memory use 3-4x over just holding the model itself. At larger scales (7B+) this quickly exceeds what a single consumer GPU can hold, and training all parameters also risks catastrophically forgetting the model's general knowledge if the fine-tuning dataset is small and narrow (like our finance domain data).

## 2. What LoRA does

LoRA (Low-Rank Adaptation) freezes the original model weights entirely and instead injects small trainable "adapter" matrices alongside key layers (attention projections, MLP projections). Instead of learning a full-size weight update, it learns two much smaller low-rank matrices (`A` and `B`) whose product approximates the update. Only these small matrices are trained and saved — typically <1% of the original parameter count — while the frozen base model supplies the rest of the computation.

## 3. What QLoRA does

QLoRA combines LoRA with 4-bit quantization of the frozen base model. The base model's weights are loaded and stored in 4-bit precision (via bitsandbytes' NF4 format), cutting memory footprint roughly 4x compared to 16-bit weights, while the LoRA adapter matrices themselves are still trained in higher precision (bf16/fp16) for stable gradients. This is what Unsloth uses by default in `load_in_4bit=True`.

## 4. Why QLoRA is useful on limited GPU

QLoRA is what makes it realistic to fine-tune on a single free-tier Colab T4 GPU (~15GB VRAM). Without quantization, even loading a 1B+ parameter model with an optimizer for full fine-tuning could exceed available memory, and larger models wouldn't fit at all. Quantizing the frozen base to 4-bit and only keeping LoRA adapters in higher precision means the memory-hungry part (billions of frozen weights) shrinks dramatically, while the trainable part stays small enough to fit comfortably alongside activations and gradients.

## 5. What is non-instruction fine-tuning?

Non-instruction fine-tuning (Stage 1 in this project) trains the model on raw, unstructured domain text with the standard causal language modeling objective — predict the next token, no question/answer structure. The goal isn't to teach the model to *answer questions*; it's to shift its internal representations toward domain vocabulary, tone, and recurring concepts (e.g., "cost basis," "liquidity," "APR") before it ever sees an instruction format. Think of it as reading comprehension of the domain before the model learns question-answering behavior.

## 6. What is instruction fine-tuning?

Instruction fine-tuning (Stage 2, SFT) trains the model on explicit instruction/response pairs, using a consistent prompt template (`### Question: ... ### Response: ...` in this project). This teaches the model the actual task: given a user's finance question, produce a helpful, on-topic answer in the expected format. It builds directly on top of the Stage 1 adapter so the domain adaptation isn't lost.

## 7. What is DPO?

DPO (Direct Preference Optimization) is a preference-alignment technique that trains directly on pairs of (chosen, rejected) responses to the same prompt, without needing a separate reward model (unlike classic RLHF/PPO). It adjusts the model so the log-probability it assigns to the *chosen* response increases relative to the *rejected* one, using a reference model (the pre-DPO SFT model) to keep the update anchored and prevent the policy from drifting too far or degenerating.

## 8. Difference between SFT and DPO

- **SFT** trains on single "correct" examples — it only ever sees what a good answer looks like, not what a bad one looks like.
- **DPO** trains on *contrastive pairs* — it sees both a good and a bad answer to the same prompt and learns the relative preference between them.

In practice, SFT teaches the model the domain and the response format; DPO refines response *quality* along axes SFT alone doesn't directly optimize for — tone, safety, specificity, and avoiding weak/generic answers — because it explicitly penalizes the rejected style rather than just rewarding one correct style.

## 9. Configuration values used

| Hyperparameter | Stage 1 (non-instruction) | Stage 2 (instruction SFT) | Stage 3 (DPO) |
|---|---|---|---|
| LoRA rank (`r`) | 16 | 16 | 16 (inherited from SFT adapter) |
| LoRA alpha | 16 | 16 | 16 |
| LoRA dropout | 0 | 0 | 0 |
| Learning rate | 2e-4 | 2e-4 | 5e-6 |
| Batch size (per device) | 2 | 4 | 2 |
| Gradient accumulation steps | 4 | 4 | 4 |
| Effective batch size | 8 | 16 | 8 |
| Steps / epochs | 60 max steps | 3 epochs | 3 epochs |
| DPO beta | — | — | 0.1 |

Rationale: LoRA rank 16 / alpha 16 (1:1 ratio) is a common, stable starting point for small models like Qwen2.5-0.5B — high enough to give the adapter real capacity without overfitting on a ~100-500 example dataset. SFT uses a higher learning rate (2e-4) since we're teaching a new behavior (instruction following) from a relatively small dataset; DPO uses a much lower learning rate (5e-6), which is standard practice — preference optimization is a *fine* adjustment on top of an already-competent SFT model, and a high LR would risk destabilizing it or overfitting to the fairly small (55-example) preference set.

_Update this table with your actual observed values if you change any settings while running the notebooks._
