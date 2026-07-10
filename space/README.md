---
title: Finance FAQ Assistant
emoji: 💰
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Finance FAQ Assistant

A personal-finance FAQ assistant fine-tuned from `Qwen2.5-0.5B` in three
stages with [Unsloth](https://github.com/unslothai/unsloth): non-instruction
domain adaptation → instruction fine-tuning (SFT) → DPO preference
alignment.

Model: [Naveengangadhara/finance-qwen-dpo-merged](https://huggingface.co/Naveengangadhara/finance-qwen-dpo-merged)

Training pipeline and datasets: see the
[finance-faq-assistant-finetuning GitHub repo](https://github.com/Naveeengangadhara/finance-faq-assistant-finetuning).
