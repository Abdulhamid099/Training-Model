#!/usr/bin/env python3
"""
Command-line interface for training models.
"""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training_model.configs.training_config import TrainingConfig, ModelConfig
from training_model.training.trainer import ModelTrainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a language model with LoRA")
    
    # Model configuration
    parser.add_argument("--model-name", type=str, default="gpt2-medium",
                       help="Name of the pretrained model")
    parser.add_argument("--config", type=str, help="Path to training config YAML file")
    
    # Dataset configuration
    parser.add_argument("--dataset-path", type=str, default="datasets",
                       help="Path to dataset directory")
    parser.add_argument("--train-file", type=str, default="train.jsonl",
                       help="Training dataset file")
    parser.add_argument("--eval-file", type=str, default="eval.jsonl",
                       help="Evaluation dataset file")
    
    # Training parameters
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Training batch size per device")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    
    # LoRA configuration
    parser.add_argument("--use-lora", action="store_true", default=True,
                       help="Use LoRA for fine-tuning")
    parser.add_argument("--lora-r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha")
    
    # Hardware configuration
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--cpu-only", action="store_true",
                       help="Force CPU-only training")
    
    # Logging
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--run-name", type=str,
                       help="Name for the training run")
    
    # Preset configurations
    parser.add_argument("--preset", type=str, choices=["mistral", "llama2", "falcon", "gpt2"],
                       help="Use preset model configuration")
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config:
        config = TrainingConfig.load_config(args.config)
    else:
        config = TrainingConfig()
    
    # Apply preset if specified
    if args.preset:
        preset_configs = {
            "mistral": ModelConfig.MISTRAL_7B,
            "llama2": ModelConfig.LLAMA2_7B,
            "falcon": ModelConfig.FALCON_7B,
            "gpt2": ModelConfig.GPT2_MEDIUM,
        }
        config.update_from_dict(preset_configs[args.preset])
    
    # Override with command line arguments
    config.model_name = args.model_name
    config.dataset_path = args.dataset_path
    config.train_file = args.train_file
    config.eval_file = args.eval_file
    config.output_dir = args.output_dir
    config.num_train_epochs = args.epochs
    config.per_device_train_batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.max_length = args.max_length
    config.use_lora = args.use_lora
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.fp16 = args.fp16 and not args.cpu_only
    config.use_cpu = args.cpu_only
    config.run_name = args.run_name
    
    if args.wandb:
        config.report_to = "wandb"
    
    # Print configuration
    print("Training Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Dataset: {config.dataset_path}/{config.train_file}")
    print(f"  Output: {config.output_dir}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch Size: {config.per_device_train_batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  LoRA: {config.use_lora} (r={config.lora_r}, alpha={config.lora_alpha})")
    print(f"  FP16: {config.fp16}")
    print()
    
    # Create trainer and run training
    trainer = ModelTrainer(config)
    trainer.run_full_training_pipeline()


if __name__ == "__main__":
    main()