# Base Model vs Instruction Fine-Tuned (SFT) Model

**Base model:** `unsloth/Qwen2.5-0.5B` (untrained)
**Fine-tuned model:** Stage 2 SFT adapter from `notebooks/instruction_finetuning.ipynb` (`outputs/sft_adapter`), trained on `data/instruction_dataset.jsonl` starting from the Stage 1 non-instruction adapter.

## How to fill this in

1. Run `notebooks/instruction_finetuning.ipynb` end to end.
2. Re-ask the same 10 questions from `reports/eval_questions.json` (same ones used in `base_model_evaluation.md`) against the trained SFT model using the `ask()` helper defined at the bottom of the notebook.
3. Copy the base model answers from `base_model_evaluation.md` into the second column here so the comparison is side by side.
4. Judge "Which is Better?" using the criteria below, and give a one-line reason.

## Evaluation criteria

Correctness · Domain accuracy · Clarity · Safety · Helpfulness · Less generic response · Better domain-specific behavior

## Comparison Table

| # | Question | Base Model Answer | Fine-Tuned Model Answer | Which is Better? | Reason |
|---|----------|--------------------|--------------------------|-------------------|--------|
| 1 | How can I start building an emergency fund? | | | | |
| 2 | What factors affect my credit score the most? | | | | |
| 3 | Should I pay off debt first or start investing? | | | | |
| 4 | What is the difference between a Roth IRA and a traditional IRA? | | | | |
| 5 | How much of my income should I allocate to savings each month? | | | | |
| 6 | What should I consider before taking out a mortgage? | | | | |
| 7 | How does compound interest affect long-term savings? | | | | |
| 8 | What is a good strategy for creating a monthly budget? | | | | |
| 9 | When does it make sense to hire a tax consultant instead of filing myself? | | | | |
| 10 | What is the difference between a stock and a bond? | | | | |

## Summary observations

_Summarize how instruction fine-tuning changed response style/quality vs. the base model — e.g. more structured answers, closer to the instruction dataset's tone, fewer off-topic tangents, still occasional hallucination, etc._
