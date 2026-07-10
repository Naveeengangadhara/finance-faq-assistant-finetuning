"""
Gradio demo for the Finance FAQ Assistant — meant to run as a Hugging Face
Space (SDK: gradio). Loads the merged, DPO-aligned model straight from the
Hugging Face Hub, so this app has no dependency on the training notebooks
or local files.
"""

import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Naveengangadhara/finance-qwen-dpo-merged")

PROMPT_TEMPLATE = """Below is a question about personal finance. Write a clear, accurate, and helpful response.

### Question:
{instruction}

### Response:
"""

print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()
print("Model loaded.")


def answer(question: str, history=None, max_new_tokens: int = 200) -> str:
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


demo = gr.ChatInterface(
    fn=lambda message, history: answer(message),
    title="Finance FAQ Assistant",
    description=(
        "A personal-finance FAQ assistant, fine-tuned from `Qwen2.5-0.5B` in three "
        "stages with Unsloth: non-instruction domain adaptation, instruction SFT, and "
        "DPO preference alignment. Ask about budgeting, savings, credit, investing "
        "basics, retirement accounts, mortgages, or taxes."
    ),
    examples=[
        "How can I start building an emergency fund?",
        "What is the difference between a Roth IRA and a traditional IRA?",
        "Should I pay off debt first or start investing?",
        "What factors affect my credit score the most?",
    ],
)

if __name__ == "__main__":
    demo.launch()
