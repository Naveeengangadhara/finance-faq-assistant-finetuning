# Finance FAQ Assistant — Fine-Tuning with Unsloth

A domain-specific AI assistant for personal-finance FAQs, built by fine-tuning
`Qwen2.5-0.5B` in three stages with [Unsloth](https://github.com/unslothai/unsloth):
non-instruction fine-tuning → instruction fine-tuning (SFT) → DPO preference
alignment.

## Domain selected

**Finance FAQ Assistant.**

## Business problem

As a GenAI Engineer, the task is to build an internal assistant that can
answer personal-finance questions — budgeting, savings, credit, investing
basics, retirement accounts, mortgages, taxes — with clear, correct,
domain-specific answers, instead of the generic, hedge-everything answers a
general-purpose base model gives out of the box.

## Dataset details

All three datasets are derived from the public, **MIT-licensed**
[`gbharti/finance-alpaca`](https://huggingface.co/datasets/gbharti/finance-alpaca)
dataset (a combination of Stanford Alpaca + FiQA finance Q&A, ~69k rows),
filtered down to finance-FAQ-relevant, self-contained Q&A pairs via
[`scripts/prepare_datasets.py`](scripts/prepare_datasets.py). No paid API
calls or model-generated content were used — all text comes from the
original public dataset.

| File | Rows | Description |
|---|---|---|
| `data/non_instruction_data.txt` | 300 paragraphs | Raw finance Q&A answer text, used for Stage 1 domain-adaptation (causal LM) training |
| `data/instruction_dataset.jsonl` | 600 pairs | `{"instruction", "response"}` pairs for Stage 2 SFT |
| `data/preference_dataset.jsonl` | 300 triples | `{"prompt", "chosen", "rejected"}` for Stage 3 DPO. `chosen` = real dataset answer; `rejected` = a rule-based generic/unhelpful response drawn from a small template pool (see `REJECTED_TEMPLATES` in the prep script) |

Regenerate all three files at any time with:

```bash
python3 scripts/prepare_datasets.py
```

The 10 fixed evaluation questions used throughout `reports/` live in
[`reports/eval_questions.json`](reports/eval_questions.json).

## Base model used

[`unsloth/Qwen2.5-0.5B`](https://huggingface.co/unsloth/Qwen2.5-0.5B), loaded
4-bit via Unsloth's `FastLanguageModel`. Chosen for its small size (fast to
train on a free Colab T4 GPU) while still being large enough to show a clear
before/after fine-tuning improvement.

## Non-instruction fine-tuning approach

Notebook: [`notebooks/non_instruction_finetuning.ipynb`](notebooks/non_instruction_finetuning.ipynb)

Raw finance paragraphs from `data/non_instruction_data.txt` are chunked and
trained with a plain causal-LM (next-token prediction) objective — no
instruction/response structure — so the model absorbs finance vocabulary,
tone, and recurring concepts before it ever learns to answer questions. This
produces the Stage 1 adapter at `outputs/non_instruction_adapter`.

## Instruction fine-tuning approach

Notebook: [`notebooks/instruction_finetuning.ipynb`](notebooks/instruction_finetuning.ipynb)

Continues from the Stage 1 adapter and trains on
`data/instruction_dataset.jsonl`, wrapped in a consistent
`### Question: ... ### Response: ...` prompt template, using `trl`'s
`SFTTrainer`. This teaches the model to actually answer finance questions
helpfully. Produces the Stage 2 adapter at `outputs/sft_adapter`.

## DPO alignment approach

Notebook: [`notebooks/dpo_alignment.ipynb`](notebooks/dpo_alignment.ipynb)

Continues from the Stage 2 SFT adapter and trains on
`data/preference_dataset.jsonl` using `trl`'s `DPOTrainer` (patched via
Unsloth's `PatchDPOTrainer`), teaching the model to prefer the "chosen"
finance answer over the "rejected" generic/unhelpful one for the same
prompt. Produces the Stage 3 adapter at `outputs/dpo_adapter` and a merged,
full-precision model at `outputs/dpo_merged`.

## LoRA / QLoRA configuration

| Stage | Rank | Alpha | Dropout | LR | Batch size (eff.) |
|---|---|---|---|---|---|
| Non-instruction | 16 | 16 | 0 | 2e-4 | 8 |
| Instruction SFT | 16 | 16 | 0 | 2e-4 | 16 |
| DPO | 16 (inherited) | 16 | 0 | 5e-6 | 8 |

Full rationale for these values is in
[`reports/fine_tuning_explanation.md`](reports/fine_tuning_explanation.md).
All stages load the base model 4-bit (QLoRA) via `load_in_4bit=True`.

## Training screenshots or logs

_Add screenshots of the Colab training run (loss curves / trainer logs) here
after running each notebook, e.g.:_

```
reports/screenshots/stage1_training_log.png
reports/screenshots/stage2_training_log.png
reports/screenshots/stage3_training_log.png
```

## Before vs after output comparison

See:
- [`reports/base_model_evaluation.md`](reports/base_model_evaluation.md) — base model on 10 questions
- [`reports/sft_model_comparison.md`](reports/sft_model_comparison.md) — base vs SFT
- [`reports/final_evaluation.md`](reports/final_evaluation.md) — base vs SFT vs DPO, final verdict

These are pre-built with the same 10 fixed questions; fill in the answer
columns after running each notebook in Colab.

## Final observations

_Fill in after completing all three stages and the final evaluation —
summarize what improved at each stage and whether DPO meaningfully improved
response quality over SFT alone._

## Challenges faced

_Document any real issues you hit while running this in Colab — e.g. GPU
memory limits, dependency version conflicts between `unsloth`/`trl`/`peft`,
runtime disconnects, dataset formatting edge cases, etc._

## Future improvements

- Scale up the raw/instruction/preference datasets beyond the assignment
  minimums for more robust generalization.
- Try a larger base model (Qwen2.5-1.5B / Llama-3.2-1B) if GPU budget allows.
- Add an automated evaluation script (e.g. LLM-as-judge or a rubric scorer)
  instead of manual table filling for the comparison reports.
- Expand the preference dataset with harder negative examples (subtly wrong
  answers) rather than only generic/rude rejected responses.

## Repository structure

```
Finance-Assistance/
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
├── reports/
│   ├── eval_questions.json
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
├── scripts/
│   └── prepare_datasets.py
├── src/
│   └── inference.py
├── space/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── README.md
└── requirements.txt
```

## How to run this project end to end

1. **Data prep (local, no GPU):** `python3 scripts/prepare_datasets.py` — already run; regenerate if you want a different sample.
2. **Push this repo to GitHub** so the Colab notebooks can `git clone` it (or upload files manually per notebook — see the upload cells in each notebook).
3. **Stage 1:** open `notebooks/non_instruction_finetuning.ipynb` in Colab (T4 GPU runtime), run all cells. The `notebook_login()` cell will prompt for a Hugging Face access token (Settings → Access Tokens, "Write" role) — paste it in when asked. The final cell pushes the adapter to `Naveengangadhara/finance-qwen-stage1-adapter` on the Hub.
4. **Stage 2:** open `notebooks/instruction_finetuning.ipynb`, run all cells. It loads the Stage 1 adapter from the local `outputs/` folder if present in the same runtime, otherwise pulls it from the Hub. Pushes the result to `Naveengangadhara/finance-qwen-sft-adapter`.
5. Fill in `reports/base_model_evaluation.md` and `reports/sft_model_comparison.md` using the `ask()` helpers in these notebooks.
6. **Stage 3:** open `notebooks/dpo_alignment.ipynb`, run all cells (loads the Stage 2 SFT adapter the same local-then-Hub way). Pushes the final adapter to `Naveengangadhara/finance-qwen-dpo-adapter` and the merged, full-precision model to `Naveengangadhara/finance-qwen-dpo-merged` — **this merged repo is what both `src/inference.py` and the demo app load.**
7. Fill in `reports/final_evaluation.md`.
8. Run `python src/inference.py "your question"` from anywhere (no GPU/Colab needed) — it pulls the merged model straight from the Hub.

Because every stage pushes to the Hub as soon as it finishes, none of this depends on keeping a single Colab runtime alive across sessions — you can run Stage 1 today and Stage 3 next week in a brand-new runtime.

## Deploying the demo (Hugging Face Spaces)

The `space/` folder is a self-contained Gradio app that loads
`Naveengangadhara/finance-qwen-dpo-merged` from the Hub and serves a chat UI.
It has no dependency on this GitHub repo at runtime, so it can be deployed on
its own once Stage 3 has been run at least once:

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space), pick a name (e.g. `finance-faq-assistant`), SDK: **Gradio**, hardware: free **CPU basic** is enough for a 0.5B model.
2. Clone the new Space repo it creates: `git clone https://huggingface.co/spaces/Naveengangadhara/finance-faq-assistant`
3. Copy `space/app.py`, `space/requirements.txt`, and `space/README.md` into that cloned folder.
4. `git add . && git commit -m "Add Finance FAQ Assistant demo" && git push`
5. The Space builds automatically and gives you a public URL (`https://huggingface.co/spaces/Naveengangadhara/finance-faq-assistant`) you can share directly with users.

No secrets are needed in the Space since `finance-qwen-dpo-merged` is a public model repo. First load takes ~30-60s (cold start on free CPU); after that each answer generates in a few seconds.
