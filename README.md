# 🧠 Training-Model

A comprehensive repository for experimenting with training/fine-tuning large language models like `Mistral-7B`, `Llama2`, and `GPT-2` using modern tools such as Hugging Face Transformers, LoRA (Low-Rank Adaptation), and custom datasets.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd training-model

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```bash
# Run a simple training example
python examples/simple_training_example.py

# Train with a specific model
python scripts/train.py --model-name gpt2-medium --epochs 3

# Use a preset configuration
python scripts/train.py --preset mistral --config configs/mistral_7b.yaml
```

## 📂 Project Structure

```
training-model/
├── src/training_model/          # Main package
│   ├── configs/                 # Configuration classes
│   ├── data/                    # Data processing utilities
│   ├── training/                # Training modules
│   ├── evaluation/              # Evaluation metrics
│   ├── models/                  # Model utilities
│   └── utils/                   # Helper utilities
├── scripts/                     # CLI scripts
├── examples/                    # Example scripts
├── notebooks/                   # Jupyter notebooks
├── configs/                     # Configuration files
├── datasets/                    # Dataset directory
├── tests/                       # Unit tests
└── docs/                        # Documentation
```

## 🔧 Features

### ✅ Core Features
- **Multiple Model Support**: GPT-2, Mistral-7B, Llama2, Falcon
- **LoRA Fine-tuning**: Memory-efficient training with PEFT
- **Multiple Data Formats**: JSONL, JSON, CSV, plain text
- **Flexible Configuration**: YAML configs and command-line overrides
- **Evaluation Metrics**: Perplexity, BLEU, ROUGE scores
- **Logging Integration**: Weights & Biases, TensorBoard support
- **Mixed Precision**: FP16/BF16 training support

### 🎯 Training Capabilities
- Instruction-response fine-tuning
- Conversation-based training
- Custom dataset preprocessing
- Automatic data splitting
- Early stopping and checkpointing
- Hyperparameter optimization

## 📖 Usage Examples

### 1. Basic Training

```python
from training_model.configs.training_config import TrainingConfig
from training_model.training.trainer import ModelTrainer

# Configure training
config = TrainingConfig(
    model_name="gpt2-medium",
    dataset_path="datasets",
    train_file="train.jsonl",
    num_train_epochs=3,
    use_lora=True
)

# Train the model
trainer = ModelTrainer(config)
trainer.run_full_training_pipeline()
```

### 2. Command Line Training

```bash
# Basic training
python scripts/train.py \
    --model-name mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset-path datasets \
    --epochs 3 \
    --batch-size 2 \
    --use-lora \
    --fp16

# With Weights & Biases logging
python scripts/train.py \
    --preset mistral \
    --wandb \
    --run-name "mistral-experiment-1"
```

### 3. Data Preprocessing

```python
from training_model.data.data_processor import DataProcessor

# Split dataset
DataProcessor.split_dataset(
    input_path="datasets/full_dataset.jsonl",
    train_path="datasets/train.jsonl",
    eval_path="datasets/eval.jsonl",
    train_ratio=0.8
)

# Create instruction dataset
instructions = [
    {"instruction": "What is AI?", "response": "AI is..."},
    # ... more examples
]
DataProcessor.create_instruction_dataset(instructions, "datasets/instructions.jsonl")
```

## 📊 Supported Models

| Model | Size | Context Length | LoRA Targets | Notes |
|-------|------|----------------|--------------|-------|
| GPT-2 | 355M - 1.5B | 1024 | `c_attn` | Good for testing |
| Mistral-7B | 7B | 8192 | `q_proj, k_proj, v_proj, o_proj` | Recommended |
| Llama2-7B | 7B | 4096 | `q_proj, k_proj, v_proj, o_proj` | Popular choice |
| Falcon-7B | 7B | 2048 | `query_key_value` | Alternative option |

## 🎯 Configuration Presets

Use preset configurations for common models:

```bash
# Mistral-7B optimized settings
python scripts/train.py --preset mistral

# Llama2-7B optimized settings  
python scripts/train.py --preset llama2

# Quick testing with GPT-2
python scripts/train.py --preset gpt2
```

## 📈 Monitoring and Evaluation

### Weights & Biases Integration
```bash
python scripts/train.py --wandb --run-name "my-experiment"
```

### Custom Metrics
- **Perplexity**: Standard language modeling metric
- **BLEU Score**: For text generation quality
- **ROUGE Score**: For summarization tasks
- **Token Accuracy**: Training progress indicator

## 🔧 Advanced Configuration

### LoRA Settings
```yaml
use_lora: true
lora_r: 16          # Rank (lower = more efficient)
lora_alpha: 32      # Scaling factor
lora_dropout: 0.1   # Dropout rate
lora_target_modules: ["q_proj", "v_proj"]
```

### Training Optimization
```yaml
fp16: true                    # Mixed precision
gradient_checkpointing: true  # Memory optimization
gradient_accumulation_steps: 4 # Effective batch size
warmup_steps: 100            # Learning rate warmup
```

## 📚 Examples and Tutorials

1. **Getting Started**: `examples/simple_training_example.py`
2. **Advanced Training**: `notebooks/getting_started.ipynb`
3. **Custom Data**: See `src/training_model/data/` for processors
4. **Configuration**: Example configs in `configs/`

## 🔍 Troubleshooting

### Common Issues

**Out of Memory?**
- Reduce `per_device_train_batch_size`
- Enable `gradient_checkpointing`
- Use smaller `max_length`
- Try CPU training with `--cpu-only`

**Slow Training?**
- Enable `fp16` training
- Increase `gradient_accumulation_steps`
- Use multiple GPUs with `device_map: "auto"`

**Poor Results?**
- Check your dataset quality
- Adjust learning rate (try 1e-4 to 5e-5)
- Increase training epochs
- Tune LoRA parameters

## 🧠 Project Goals

- ✅ Understand model behavior during fine-tuning
- ✅ Explore optimal training configurations  
- ✅ Build custom language model applications
- ✅ Create reusable training infrastructure
- 🔄 Experiment with advanced techniques
- 🔄 Build evaluation benchmarks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ☕, curiosity, and experimentation.**

*Explore the frontiers of language model fine-tuning!* 🚀