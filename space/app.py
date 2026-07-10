"""
Gradio demo for the Finance FAQ Assistant — meant to run as a Hugging Face
Space (SDK: gradio). Loads both the untrained base model and the merged,
DPO-aligned fine-tuned model straight from the Hugging Face Hub, and shows
their answers to the same question side by side so the effect of
fine-tuning is directly visible.
"""

import os

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "unsloth/Qwen2.5-0.5B")
FINETUNED_MODEL_ID = os.environ.get(
    "FINETUNED_MODEL_ID", "Naveengangadhara/finance-qwen-dpo-merged"
)

PROMPT_TEMPLATE = """Below is a question about personal finance. Write a clear, accurate, and helpful response.

### Question:
{instruction}

### Response:
"""

print(f"Loading base model {BASE_MODEL_ID} ...")
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16)
base_model.eval()

print(f"Loading fine-tuned model {FINETUNED_MODEL_ID} ...")
ft_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_ID)
ft_model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_ID, torch_dtype=torch.float16)
ft_model.eval()
print("Both models loaded.")


def _generate(model, tokenizer, question: str, device: str, max_new_tokens: int = 200) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
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


@spaces.GPU
def compare(question: str):
    if not question or not question.strip():
        return "", ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)
    ft_model.to(device)
    base_answer = _generate(base_model, base_tokenizer, question, device)
    ft_answer = _generate(ft_model, ft_tokenizer, question, device)
    return base_answer, ft_answer


EXAMPLES = [
    "How can I start building an emergency fund?",
    "What is the difference between a Roth IRA and a traditional IRA?",
    "Should I pay off debt first or start investing?",
    "What factors affect my credit score the most?",
]

with gr.Blocks(title="Finance FAQ Assistant — Base vs Fine-Tuned") as demo:
    gr.Markdown(
        "# Finance FAQ Assistant\n"
        "Ask a personal-finance question and compare the answer from the "
        "**base model** (untrained `Qwen2.5-0.5B`) against the **fine-tuned "
        "model** (non-instruction domain adaptation → instruction SFT → "
        "DPO alignment) side by side."
    )
    question = gr.Textbox(
        label="Your question",
        placeholder="e.g. How can I start building an emergency fund?",
    )
    submit = gr.Button("Compare", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Base model (before fine-tuning)")
            base_output = gr.Textbox(label="", lines=14, interactive=False)
        with gr.Column():
            gr.Markdown("### Fine-tuned model (after DPO alignment)")
            ft_output = gr.Textbox(label="", lines=14, interactive=False)

    gr.Examples(examples=EXAMPLES, inputs=question)

    submit.click(fn=compare, inputs=question, outputs=[base_output, ft_output])
    question.submit(fn=compare, inputs=question, outputs=[base_output, ft_output])

if __name__ == "__main__":
    demo.launch()
