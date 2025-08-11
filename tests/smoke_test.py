from pathlib import Path
from src.data.build_dataset import read_jsonl_examples, format_example


def main():
    path = Path("data/sample.jsonl")
    rows = read_jsonl_examples(path)
    assert len(rows) >= 3, "Expected at least 3 examples"
    formatted = [format_example(r["prompt"], r["response"]) for r in rows]
    assert all("[INST]" in t and "</s>" in t for t in formatted)
    print("Smoke test passed: dataset utilities working.")


if __name__ == "__main__":
    main()