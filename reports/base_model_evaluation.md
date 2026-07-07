# Base Model Evaluation

**Model:** `unsloth/Qwen2.5-0.5B` (untrained, straight from Hugging Face — no fine-tuning applied)
**Purpose:** establish a "before" baseline for the Finance FAQ Assistant, using the same 10 questions from `reports/eval_questions.json` that will be reused in `sft_model_comparison.md` and `final_evaluation.md`.

## How to fill this in

1. Run `notebooks/instruction_finetuning.ipynb` up through the model-loading cell, but load `unsloth/Qwen2.5-0.5B` directly (skip attaching/loading any adapter).
2. Ask each of the 10 questions below with plain generation (no prompt template needed for the raw base model, or use the same `### Question / ### Response` template for a fair comparison — keep it consistent with later stages).
3. Paste the raw output into the table and note what's wrong with it (generic, factually vague, off-topic, unsafe, too short, etc.).

## Results

| # | Question | Base Model Answer | Problem |
|---|----------|--------------------|---------|
| 1 | How can I start building an emergency fund? | _fill in after running base model_ | |
| 2 | What factors affect my credit score the most? | | |
| 3 | Should I pay off debt first or start investing? | | |
| 4 | What is the difference between a Roth IRA and a traditional IRA? | | |
| 5 | How much of my income should I allocate to savings each month? | | |
| 6 | What should I consider before taking out a mortgage? | | |
| 7 | How does compound interest affect long-term savings? | | |
| 8 | What is a good strategy for creating a monthly budget? | | |
| 9 | When does it make sense to hire a tax consultant instead of filing myself? | | |
| 10 | What is the difference between a stock and a bond? | | |

## Summary observations

_After filling in the table, summarize the recurring failure pattern(s) of the base model here — e.g. generic non-domain-specific phrasing, lack of concrete numbers/steps, hedging without answering, etc. This motivates Stage 1–3 fine-tuning._
