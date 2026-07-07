"""
Simple inference script for the final DPO-aligned Finance FAQ Assistant.

Usage:
    python src/inference.py "How can I start building an emergency fund?"
    python src/inference.py            # interactive prompt loop

Loads the merged model saved by notebooks/dpo_alignment.ipynb at
outputs/dpo_merged (falls back to the raw LoRA adapter at
outputs/dpo_adapter if the merged model isn't present).

Requires the same GPU environment (unsloth, torch, transformers) the
notebooks were trained in.
"""

import sys
from pathlib import Path

from unsloth import FastLanguageModel

MAX_SEQ_LENGTH = 2048
MERGED_MODEL_DIR = "outputs/dpo_merged"
ADAPTER_DIR = "outputs/dpo_adapter"

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

    if Path(MERGED_MODEL_DIR).is_dir():
        model_path = MERGED_MODEL_DIR
    elif Path(ADAPTER_DIR).is_dir():
        model_path = ADAPTER_DIR
    else:
        raise FileNotFoundError(
            "No trained model found. Run notebooks/dpo_alignment.ipynb first "
            f"to produce {MERGED_MODEL_DIR} or {ADAPTER_DIR}."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    _model, _tokenizer = model, tokenizer
    return model, tokenizer


def generate_answer(question: str, max_new_tokens: int = 200) -> str:
    model, tokenizer = _load_model()
    prompt = PROMPT_TEMPLATE.format(instruction=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=0.7,
        do_sample=True,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer = generate_answer(question)
        print(answer)
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
