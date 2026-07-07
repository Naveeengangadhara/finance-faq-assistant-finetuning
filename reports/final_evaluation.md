# Final Evaluation — Base vs SFT vs DPO

**Base model:** `unsloth/Qwen2.5-0.5B` (untrained)
**SFT model:** Stage 2 adapter (`outputs/sft_adapter`), trained on `data/instruction_dataset.jsonl`
**DPO model:** Stage 3 adapter (`outputs/dpo_adapter` / merged at `outputs/dpo_merged`), aligned on `data/preference_dataset.jsonl`

## How to fill this in

1. Run all three notebooks in order: `non_instruction_finetuning.ipynb` → `instruction_finetuning.ipynb` → `dpo_alignment.ipynb`.
2. Ask the same 10 questions from `reports/eval_questions.json` against all three models.
3. Reuse the Base and SFT answers already captured in `base_model_evaluation.md` and `sft_model_comparison.md`.
4. Add the DPO model's answers and pick the overall best answer per question with a reason.

## Evaluation criteria

Correctness · Helpfulness · Domain accuracy · Safety · Tone · Clarity · Hallucination reduction · Professional response quality

## Final Comparison Table

| # | Question | Base Model Answer | SFT Model Answer | DPO Model Answer | Best Answer | Reason |
|---|----------|--------------------|--------------------|--------------------|-------------|--------|
| 1 | How can I start building an emergency fund? | | | | | |
| 2 | What factors affect my credit score the most? | | | | | |
| 3 | Should I pay off debt first or start investing? | | | | | |
| 4 | What is the difference between a Roth IRA and a traditional IRA? | | | | | |
| 5 | How much of my income should I allocate to savings each month? | | | | | |
| 6 | What should I consider before taking out a mortgage? | | | | | |
| 7 | How does compound interest affect long-term savings? | | | | | |
| 8 | What is a good strategy for creating a monthly budget? | | | | | |
| 9 | When does it make sense to hire a tax consultant instead of filing myself? | | | | | |
| 10 | What is the difference between a stock and a bond? | | | | | |

## Final observations

_Summarize the overall progression Base → SFT → DPO: what improved at each stage, whether DPO measurably reduced generic/unsafe/incorrect answers compared to SFT alone, and any cases where DPO didn't help or made things worse._

## Interview-ready summary

> I built a domain-specific Finance FAQ Assistant using Unsloth. I first performed non-instruction fine-tuning on raw finance domain text, then instruction fine-tuning on finance Q&A data, and finally DPO alignment using a finance preference dataset. I compared the base model, SFT model, and DPO-aligned model on 10 fixed finance questions and built a simple inference script (`src/inference.py`) to query the final model.
