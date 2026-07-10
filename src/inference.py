"""
Simple inference script for the final DPO-aligned Finance FAQ Assistant.

Usage:
    python src/inference.py "How can I start building an emergency fund?"
    python src/inference.py            # interactive prompt loop

Loads the merged model from the Hugging Face Hub (MODEL_ID below) by
default. If a local outputs/dpo_merged directory exists (e.g. right after
running dpo_alignment.ipynb and downloading it), that takes priority.
Works on CPU or GPU — no unsloth/bitsandbytes required, since the merged
model is plain full-precision weights.
"""

import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Naveengangadhara/finance-qwen-dpo-merged")
LOCAL_MERGED_DIR = "outputs/dpo_merged"

PROMPT_TEMPLATE = """Below is a question about personal finance. Write a clear, accurate, and helpful response.

### Question:
{instruction}

### Response:
"""

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    model_path = LOCAL_MERGED_DIR if Path(LOCAL_MERGED_DIR).is_dir() else MODEL_ID
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    _model, _tokenizer = model, tokenizer
    return model, tokenizer


def generate_answer(question: str, max_new_tokens: int = 200) -> str:
    model, tokenizer = _load_model()
    prompt = PROMPT_TEMPLATE.format(instruction=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(generate_answer(question))
        return

    print("Finance FAQ Assistant (type 'exit' to quit)")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print("\nAnswer:", generate_answer(question))


if __name__ == "__main__":
    main()
