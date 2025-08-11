# 🧠 Training-Model

A hands-on repository to experiment with training/fine-tuning large language models like `Mistral-7B` using Hugging Face, LoRA, and custom datasets.

## 📂 Project Scope

- Fine-tuning transformer models
- Hyperparameter optimization
- Using datasets from local and custom inputs
- Learning rate scheduling and precision handling
- Experiment tracking and analysis (you can wire in W&B)

## 🚀 Tech Stack

- Python
- Transformers (Hugging Face)
- Datasets (JSONL)
- PEFT / LoRA
- 4-bit k-bit training (optional)

## ✅ Quickstart

1) Create an environment and install deps

```bash
make install
```

2) Run a quick smoke test (no heavy deps needed)

```bash
make smoke
```

3) Fine-tune with LoRA (edit the YAML first!)

```bash
make train
```

- Config: `configs/mistral_lora_example.yaml`
- Output: `outputs/mistral7b-lora-demo`

## 🗂️ Data Format

Provide a JSONL file where each line contains `prompt` and `response` fields. See `data/sample.jsonl`:

```json
{"prompt": "What is LoRA?", "response": "Low-Rank Adapters..."}
```

The trainer will render each pair into this template by default:

```text
<s>[INST] {prompt} [/INST] {response}</s>
```

You can change the template via `text_template` in the config.

## ⚙️ Config Highlights

See `configs/mistral_lora_example.yaml`:
- **base_model_name_or_path**: e.g. `mistralai/Mistral-7B-Instruct-v0.2`
- **use_4bit**: toggle 4-bit k-bit training (requires CUDA + bitsandbytes)
- **LoRA**: `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`
- **Training**: batch size, lr, epochs, scheduler, precision

## 🧪 Scripts

- `src/training/train_lora.py`: LoRA fine-tuning script
- `src/data/build_dataset.py`: JSONL utilities and HF dataset builder
- `tests/smoke_test.py`: minimal data utility test
- `scripts/clone_cashew.sh`: optional utility to clone a related HF Space

## 📌 TODO

- [ ] Add cleaned dataset
- [ ] Upload training logs / metrics
- [ ] Automate preprocessing scripts
- [ ] Experiment with smaller models (e.g. Falcon-1B)

---

Made with ☕, curiosity, and experimentation.