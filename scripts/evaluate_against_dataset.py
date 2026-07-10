"""
Standard "generated vs. reference" LLM evaluation: sample questions straight
from the training data (data/instruction_dataset.jsonl), generate the
fine-tuned model's answer to each one, and compare it against the actual
dataset answer it was trained on — with a text-similarity score.

This tells you how well the model reproduces/aligns with the answers it was
trained on, which is a different (and complementary) question from
scripts/evaluate_models.py (base vs. SFT vs. DPO on a fixed question set not
drawn from the training data).

Usage:
    python3 scripts/evaluate_against_dataset.py [--n 15]

Pulls the merged DPO model from the Hub — no GPU required, runs on CPU.
"""

import argparse
import json
import random
from difflib import SequenceMatcher
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Naveengangadhara/finance-qwen-dpo-merged"
ROOT = Path(__file__).resolve().parent.parent

PROMPT_TEMPLATE = """Below is a question about personal finance. Write a clear, accurate, and helpful response.

### Question:
{instruction}

### Response:
"""


def load_dataset_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15, help="number of sampled rows to evaluate")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = load_dataset_rows(ROOT / "data" / "instruction_dataset.jsonl")
    sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    results = []
    for i, row in enumerate(sample, 1):
        print(f"  {i}/{len(sample)}: {row['instruction'][:70]}")
        model_answer = generate(model, tokenizer, row["instruction"])
        score = similarity(row["response"], model_answer)
        results.append(
            {
                "question": row["instruction"],
                "reference": row["response"],
                "generated": model_answer,
                "similarity": score,
            }
        )

    avg_score = sum(r["similarity"] for r in results) / len(results)

    lines = [
        "# Dataset Reference vs. Fine-Tuned Model Answer",
        "",
        f"**Model:** `{MODEL_ID}`",
        f"**Sample size:** {len(results)} questions randomly drawn from "
        "`data/instruction_dataset.jsonl` (the model's own training data)",
        "",
        "Similarity is a plain character-sequence ratio (difflib `SequenceMatcher`,"
        " 0-1) between the model's generated answer and the dataset's actual"
        " answer for the same question — a rough proxy for how closely the"
        " model reproduces what it was trained on, not a judgment of"
        " correctness. A low score isn't necessarily bad (a differently-phrased"
        " but equally correct answer scores low); read the text side by side.",
        "",
        f"**Average similarity: {avg_score:.3f}**",
        "",
        "| # | Question | Dataset Answer | Model Answer | Similarity |",
        "|---|----------|-----------------|--------------|------------|",
    ]
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {md_escape(r['question'])} | {md_escape(r['reference'])} "
            f"| {md_escape(r['generated'])} | {r['similarity']:.3f} |"
        )

    out_path = ROOT / "reports" / "dataset_vs_model_comparison.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(ROOT)} (average similarity: {avg_score:.3f})")


if __name__ == "__main__":
    main()
