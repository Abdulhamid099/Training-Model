# 🧠 LLM Fine-Tuning Framework

A comprehensive framework for fine-tuning large language models using LoRA (Low-Rank Adaptation) with Hugging Face Transformers. This repository provides a complete pipeline for training, evaluating, and deploying fine-tuned language models.

## 🚀 Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **Multiple Model Support**: Mistral, GPT-2, and other Hugging Face models
- **Flexible Data Processing**: Support for JSONL, JSON, CSV, and text files
- **Experiment Tracking**: Integration with Weights & Biases and TensorBoard
- **Comprehensive Evaluation**: Built-in evaluation metrics and utilities
- **Easy Configuration**: YAML-based configuration management
- **Production Ready**: Proper logging, error handling, and modular design

## 📁 Project Structure

```
├── src/
│   ├── config/          # Configuration management
│   ├── data/            # Data preprocessing utilities
│   ├── training/        # Training pipeline
│   ├── utils/           # Evaluation and utility functions
│   └── models/          # Model-specific code
├── scripts/             # Executable scripts
├── configs/             # Configuration files
├── data/                # Data storage
│   ├── raw/            # Raw input data
│   └── processed/      # Preprocessed data
├── experiments/         # Training outputs and logs
├── notebooks/           # Jupyter notebooks
└── logs/               # Log files
```

## 🛠️ Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd training-model

# Run setup script
python scripts/setup.py

# Or manually install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create a JSONL file with your training data:

```json
{"instruction": "What is the capital of France?", "input": "", "output": "The capital of France is Paris."}
{"instruction": "Explain photosynthesis.", "input": "", "output": "Photosynthesis is the process..."}
```

### 3. Configure Training

Edit `configs/default_config.yaml` or create your own configuration:

```yaml
model:
  model_name: "mistralai/Mistral-7B-Instruct-v0.2"
  max_length: 2048

training:
  num_train_epochs: 3
  learning_rate: 2e-4
  per_device_train_batch_size: 2
```

### 4. Preprocess Data

```bash
python scripts/preprocess.py --input-file data/raw/your_data.jsonl
```

### 5. Start Training

```bash
python scripts/train.py --config configs/default_config.yaml --evaluate
```

## 📊 Supported Models

- **Mistral-7B-Instruct**: Large instruction-tuned model
- **GPT-2**: Smaller, faster training
- **Custom Models**: Any Hugging Face causal language model

## 🔧 Configuration

The framework uses YAML configuration files with the following sections:

### Model Configuration
```yaml
model:
  model_name: "mistralai/Mistral-7B-Instruct-v0.2"
  model_type: "mistral"
  max_length: 2048
  use_flash_attention: true
```

### LoRA Configuration
```yaml
lora:
  r: 16                    # LoRA rank
  lora_alpha: 32           # LoRA alpha
  target_modules:          # Target modules for LoRA
    - "q_proj"
    - "v_proj"
    - "k_proj"
  lora_dropout: 0.1
```

### Training Configuration
```yaml
training:
  num_train_epochs: 3
  learning_rate: 2e-4
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  fp16: true               # Use mixed precision
```

## 📈 Experiment Tracking

The framework supports multiple experiment tracking options:

- **Weights & Biases**: Set `report_to: ["wandb"]` in config
- **TensorBoard**: Set `report_to: ["tensorboard"]` in config
- **Local Logging**: All logs saved to `logs/` directory

## 🧪 Evaluation

Built-in evaluation metrics:

- **Instruction Following**: Measures how well the model follows instructions
- **Perplexity**: Language modeling quality
- **Custom Metrics**: Extensible evaluation framework

```bash
# Evaluate a trained model
python -c "
from src.utils.evaluation import evaluate_model
results = evaluate_model('mistralai/Mistral-7B-Instruct-v0.2', 'experiments/outputs')
print(results)
"
```

## 📝 Data Formats

The framework supports multiple input formats:

### Instruction Format (Recommended)
```json
{"instruction": "Task description", "input": "Optional context", "output": "Expected response"}
```

### Q&A Format
```json
{"question": "What is...?", "answer": "The answer is..."}
```

### Text Format
```json
{"text": "Input text", "continuation": "Expected continuation"}
```

## 🚀 Advanced Usage

### Custom Training Script

```python
from src.training.trainer import LLMTrainer
from src.config.training_config import get_config

# Load configuration
config = get_config("configs/my_config.yaml")

# Initialize trainer
trainer = LLMTrainer(config)

# Train model
trainer.train("data/processed/train.jsonl", "data/processed/validation.jsonl")
```

### Custom Data Preprocessing

```python
from src.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(config)
formatted_data = preprocessor.format_for_training(raw_data)
```

## 🔍 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir experiments/outputs
```

### Weights & Biases
Training metrics are automatically logged to your W&B project.

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for large models
- 50GB+ disk space for model storage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- Microsoft for LoRA research
- The open-source AI community

---

Made with ☕, curiosity, and experimentation.