#!/usr/bin/env python3
"""
Setup script for the LLM fine-tuning project.
"""
import argparse
import logging
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.preprocessor import create_sample_data


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "data/raw",
        "data/processed",
        "experiments/outputs",
        "logs",
        "notebooks",
        "configs",
        "src/data",
        "src/models",
        "src/training",
        "src/utils",
        "src/config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        return False
    return True


def create_sample_data_files():
    """Create sample data files for testing."""
    print("Creating sample data files...")
    
    # Create sample training data
    create_sample_data("data/raw/sample_data.jsonl")
    
    # Create a simple config file
    config_content = """model:
  model_name: "microsoft/DialoGPT-medium"
  model_type: "gpt2"
  max_length: 512
  use_flash_attention: false
  trust_remote_code: false

lora:
  r: 8
  lora_alpha: 16
  target_modules:
    - "c_attn"
    - "c_proj"
  lora_dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"

training:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_steps: 50
  max_steps: null
  optim: "adamw_torch"
  lr_scheduler_type: "cosine"
  fp16: false
  bf16: false
  logging_steps: 10
  save_steps: 100
  eval_steps: 100
  save_total_limit: 2
  evaluation_strategy: "steps"
  save_strategy: "steps"
  train_file: "data/processed/train.jsonl"
  validation_file: "data/processed/validation.jsonl"
  max_seq_length: 512
  dataloader_pin_memory: false
  output_dir: "experiments/outputs"
  run_name: "test-run"
  report_to:
    - "tensorboard"

data:
  train_split: 0.8
  validation_split: 0.2
  test_split: 0.0
  random_seed: 42
  max_samples: 100
  preprocessing_num_workers: 2
"""
    
    with open("configs/test_config.yaml", "w") as f:
        f.write(config_content)
    
    print("Sample data files created successfully!")


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("No GPU available. Training will use CPU (slower).")
    except ImportError:
        print("PyTorch not installed. Install requirements first.")


def main():
    parser = argparse.ArgumentParser(description="Setup the LLM fine-tuning project")
    parser.add_argument(
        "--skip-install", 
        action="store_true",
        help="Skip installing requirements"
    )
    parser.add_argument(
        "--skip-data", 
        action="store_true",
        help="Skip creating sample data"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Setting up LLM fine-tuning project...")
    
    # Create directories
    logger.info("Creating project directories...")
    create_directories()
    
    # Install requirements
    if not args.skip_install:
        logger.info("Installing requirements...")
        if not install_requirements():
            logger.error("Failed to install requirements. Please install manually.")
            return
    
    # Create sample data
    if not args.skip_data:
        logger.info("Creating sample data...")
        create_sample_data_files()
    
    # Check GPU
    logger.info("Checking GPU availability...")
    check_gpu()
    
    logger.info("Setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Add your training data to data/raw/")
    logger.info("2. Modify configs/default_config.yaml for your needs")
    logger.info("3. Run: python scripts/preprocess.py --input-file data/raw/your_data.jsonl")
    logger.info("4. Run: python scripts/train.py --config configs/default_config.yaml")


if __name__ == "__main__":
    main()