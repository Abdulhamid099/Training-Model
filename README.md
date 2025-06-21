# 🧠 Training-Model

A hands-on repository where I experiment with training/fine-tuning large language models like `Mistral-7B` using tools such as Hugging Face, LoRA, and custom datasets.

## 📂 Project Scope

- Fine-tuning transformer models
- Hyperparameter optimization
- Using datasets from local and custom inputs
- Learning rate scheduling and precision handling
- Experiment tracking and analysis

## 🚀 Tech Stack

- Python
- Transformers (HuggingFace)
- Datasets (custom, JSONL)
- PEFT / LoRA
- fp16 Training
- AutoTrain / Colab

## ⚙️ Current Experiments

| Model                | Dataset | Batch Size | Epochs | Notes                    |
|---------------------|---------|------------|--------|--------------------------|
| `Mistral-7B-Instruct` | Custom PDF -> JSONL | 2          | 3      | Using fp16 + LoRA |

## 🧠 Goals

- Understand model behavior during fine-tuning
- Explore optimal training configurations
- Build custom language model applications

## 📌 TODO

- [ ] Add cleaned dataset
- [ ] Upload training logs / metrics
- [ ] Automate preprocessing scripts
- [ ] Experiment with smaller models (e.g. Falcon-1B)

---

Made with ☕, curiosity, and experimentation.