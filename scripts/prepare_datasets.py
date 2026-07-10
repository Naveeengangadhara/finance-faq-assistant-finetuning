"""
Builds the three required datasets for the Finance FAQ Assistant project from
the public MIT-licensed gbharti/finance-alpaca dataset on Hugging Face:

  data/non_instruction_data.txt   - 50+ raw domain paragraphs
  data/instruction_dataset.jsonl  - 100+ instruction/response pairs
  data/preference_dataset.jsonl   - 50+ prompt/chosen/rejected pairs

Uses only the standard library (urllib) so it can run without installing
torch/datasets locally. Source rows are fetched via the HF datasets-server
REST API and split into three non-overlapping pools.

Dataset license: MIT (see https://huggingface.co/datasets/gbharti/finance-alpaca)
"""

import json
import random
import re
import time
import urllib.request
from pathlib import Path

DATASET = "gbharti/finance-alpaca"
API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
MAX_ROWS_TO_SCAN = 20000  # scan more of the ~69k rows for a larger candidate pool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RNG = random.Random(42)

REJECTED_TEMPLATES = [
    "Just Google it, I don't know the exact details.",
    "That's not something I can help with, contact someone else.",
    "It depends, honestly it could be anything.",
    "I'm not sure, but it's probably fine either way.",
    "You should just figure it out yourself, it's not complicated.",
    "That's a dumb question, everyone knows the answer already.",
    "Sure, whatever you think is best, doesn't really matter.",
    "I don't have time to explain this, look it up online.",
    "Not my problem, ask your bank or a friend.",
    "It's basically the same as everything else in finance, nothing special.",
]


def fetch_rows():
    rows = []
    offset = 0
    while offset < MAX_ROWS_TO_SCAN:
        url = (
            f"{API}?dataset={DATASET.replace('/', '%2F')}"
            f"&config=default&split=train&offset={offset}&length={PAGE_SIZE}"
        )
        backoff = 2
        payload = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    payload = json.load(resp)
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                print(f"  giving up on offset={offset} ({exc}), skipping page")
                break
            except Exception as exc:
                if attempt == 5:
                    print(f"  giving up on offset={offset} ({exc}), skipping page")
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

        if payload is None:
            offset += PAGE_SIZE
            time.sleep(0.3)
            continue

        page_rows = payload.get("rows", [])
        if not page_rows:
            break
        rows.extend(r["row"] for r in page_rows)
        offset += PAGE_SIZE
        time.sleep(0.3)  # be polite to the shared datasets-server API
    return rows


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", " ", text)
    text = text.replace("\n", " ")
    return text.strip()


def is_finance_faq_like(instruction: str) -> bool:
    finance_keywords = [
        "bank", "account", "loan", "credit", "debit", "invest", "stock",
        "tax", "budget", "saving", "retire", "insurance", "mortgage",
        "interest", "fund", "debt", "payment", "financ", "money", "asset",
        "expense", "income", "portfolio", "dividend", "bond",
    ]
    lowered = instruction.lower()
    return any(k in lowered for k in finance_keywords)


def build():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching rows from {DATASET} ...")
    rows = fetch_rows()
    print(f"Fetched {len(rows)} raw rows")

    seen_instructions = set()
    candidates = []
    for row in rows:
        instruction = clean_text(row.get("instruction", ""))
        extra_input = clean_text(row.get("input", ""))
        output = clean_text(row.get("output", ""))

        if extra_input:
            continue  # keep only self-contained Q&A pairs
        if not instruction or not output:
            continue
        if not (40 <= len(output) <= 900):
            continue
        if not (10 <= len(instruction) <= 250):
            continue
        if not is_finance_faq_like(instruction):
            continue

        key = instruction.lower()
        if key in seen_instructions:
            continue
        seen_instructions.add(key)
        candidates.append({"instruction": instruction, "output": output})

    print(f"Filtered down to {len(candidates)} finance FAQ-style candidates")

    RNG.shuffle(candidates)

    need = {"non_instruction": 300, "instruction": 600, "preference": 300}
    total_need = sum(need.values())
    if len(candidates) < total_need:
        raise SystemExit(
            f"Not enough filtered candidates ({len(candidates)}) for {total_need} required. "
            "Increase MAX_ROWS_TO_SCAN."
        )

    non_instruction_pool = candidates[: need["non_instruction"]]
    instruction_pool = candidates[
        need["non_instruction"] : need["non_instruction"] + need["instruction"]
    ]
    preference_pool = candidates[
        need["non_instruction"] + need["instruction"] : need["non_instruction"]
        + need["instruction"]
        + need["preference"]
    ]

    # --- data/non_instruction_data.txt ---
    paragraphs = [c["output"] for c in non_instruction_pool]
    non_instruction_path = DATA_DIR / "non_instruction_data.txt"
    non_instruction_path.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    print(f"Wrote {len(paragraphs)} paragraphs -> {non_instruction_path}")

    # --- data/instruction_dataset.jsonl ---
    instruction_path = DATA_DIR / "instruction_dataset.jsonl"
    with instruction_path.open("w", encoding="utf-8") as f:
        for c in instruction_pool:
            f.write(
                json.dumps(
                    {"instruction": c["instruction"], "response": c["output"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {len(instruction_pool)} instruction pairs -> {instruction_path}")

    # --- data/preference_dataset.jsonl ---
    preference_path = DATA_DIR / "preference_dataset.jsonl"
    with preference_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(preference_pool):
            rejected = RNG.choice(REJECTED_TEMPLATES)
            f.write(
                json.dumps(
                    {
                        "prompt": c["instruction"],
                        "chosen": c["output"],
                        "rejected": rejected,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {len(preference_pool)} preference pairs -> {preference_path}")


if __name__ == "__main__":
    build()
