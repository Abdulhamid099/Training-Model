import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def read_jsonl_examples(jsonl_path: str | Path) -> List[Dict[str, str]]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    examples: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj or "response" not in obj:
                raise ValueError("Each JSONL line must contain 'prompt' and 'response'")
            examples.append({"prompt": obj["prompt"], "response": obj["response"]})
    return examples


def format_example(prompt: str, response: str, template: Optional[str] = None) -> str:
    if template is None:
        template = "<s>[INST] {prompt} [/INST] {response}</s>"
    return template.format(prompt=prompt, response=response)


def build_hf_dataset(jsonl_path: str | Path, template: Optional[str] = None):
    # Local import to avoid hard dependency at module import time
    from datasets import Dataset

    rows = read_jsonl_examples(jsonl_path)
    texts = [format_example(r["prompt"], r["response"], template) for r in rows]
    return Dataset.from_dict({"text": texts})