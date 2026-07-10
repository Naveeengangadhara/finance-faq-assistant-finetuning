"""
Generate answers to the 10 fixed evaluation questions (reports/eval_questions.json)
from the base model, the SFT adapter, and the merged DPO model, then fill the
answer columns in reports/base_model_evaluation.md, reports/sft_model_comparison.md,
and reports/final_evaluation.md automatically.

Pulls everything from the Hugging Face Hub — no Colab/GPU required, runs fine
on CPU for a 0.5B model (just slower, a few seconds per answer).

Usage:
    python3 scripts/evaluate_models.py

Leaves the qualitative columns (Problem / Which is Better? / Reason / Best
Answer) blank for you to judge after reading the generated answers — that
part is a judgment call, not something to auto-fill.
"""

import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "unsloth/Qwen2.5-0.5B"
SFT_ADAPTER_ID = "Naveengangadhara/finance-qwen-sft-adapter"
DPO_MERGED_ID = "Naveengangadhara/finance-qwen-dpo-merged"

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = json.loads((ROOT / "reports" / "eval_questions.json").read_text())

PROMPT_TEMPLATE = """Below is a question about personal finance. Write a clear, accurate, and helpful response.

### Question:
{instruction}

### Response:
"""


def generate(model, tokenizer, question: str, max_new_tokens: int = 200) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=question)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()


def answer_all(model, tokenizer, label: str) -> list[str]:
    answers = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"  [{label}] {i}/{len(QUESTIONS)}: {q}")
        answers.append(generate(model, tokenizer, q))
    return answers


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def fill_table(md_path: Path, column_answers: dict[int, list[str]]) -> None:
    """column_answers: {column_index (0-based, after '# | Question'): [10 answers]}"""
    text = md_path.read_text()
    lines = text.splitlines()
    row_i = 0
    for i, line in enumerate(lines):
        if re.match(r"^\|\s*\d+\s*\|", line):
            cells = line.split("|")
            for col, answers in column_answers.items():
                # cells[0] is '', cells[1] is '#', cells[2] is 'Question', data cols start at 3
                cell_idx = 3 + col
                if cell_idx < len(cells):
                    cells[cell_idx] = f" {md_escape(answers[row_i])} "
            lines[i] = "|".join(cells)
            row_i += 1
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Updated {md_path.relative_to(ROOT)}")


def main():
    print(f"Loading base model {BASE_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32)
    base_model.eval()
    base_answers = answer_all(base_model, tokenizer, "base")

    print(f"\nLoading SFT adapter {SFT_ADAPTER_ID} on top of base ...")
    sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_ID)
    sft_model.eval()
    sft_answers = answer_all(sft_model, tokenizer, "sft")

    del base_model, sft_model
    print(f"\nLoading merged DPO model {DPO_MERGED_ID} ...")
    dpo_tokenizer = AutoTokenizer.from_pretrained(DPO_MERGED_ID)
    dpo_model = AutoModelForCausalLM.from_pretrained(DPO_MERGED_ID, torch_dtype=torch.float32)
    dpo_model.eval()
    dpo_answers = answer_all(dpo_model, dpo_tokenizer, "dpo")

    reports = ROOT / "reports"
    fill_table(reports / "base_model_evaluation.md", {0: base_answers})
    fill_table(reports / "sft_model_comparison.md", {0: base_answers, 1: sft_answers})
    fill_table(
        reports / "final_evaluation.md",
        {0: base_answers, 1: sft_answers, 2: dpo_answers},
    )

    print("\nDone. Answer columns are filled in — go add the qualitative")
    print("judgments (Problem / Which is Better? / Reason / Best Answer / Summary")
    print("observations) in reports/*.md by reading the generated text.")


if __name__ == "__main__":
    main()
