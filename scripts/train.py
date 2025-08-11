#!/usr/bin/env python3
"""
Main training script for LLM fine-tuning.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config.training_config import get_config
from src.data.preprocessor import DataPreprocessor, create_sample_data
from src.training.trainer import LLMTrainer
from src.utils.evaluation import evaluate_model


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Train a language model with LoRA")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-file", 
        type=str, 
        default="data/raw/sample_data.jsonl",
        help="Path to input data file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="experiments/outputs",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--create-sample-data", 
        action="store_true",
        help="Create sample data if it doesn't exist"
    )
    parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Evaluate the model after training"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting LLM fine-tuning pipeline")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = get_config(args.config)
    
    # Create sample data if requested
    if args.create_sample_data or not Path(args.data_file).exists():
        logger.info("Creating sample data...")
        create_sample_data(args.data_file)
    
    # Initialize data preprocessor
    logger.info("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(config)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    processed_dir = "data/processed"
    preprocessor.preprocess_pipeline(args.data_file, processed_dir)
    
    # Update config with processed data paths
    config.training.train_file = f"{processed_dir}/train.jsonl"
    config.training.validation_file = f"{processed_dir}/validation.jsonl"
    config.training.output_dir = args.output_dir
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = LLMTrainer(config)
    
    # Train model
    logger.info("Starting training...")
    trainer.train(
        train_file=config.training.train_file,
        validation_file=config.training.validation_file
    )
    
    logger.info("Training completed successfully!")
    
    # Evaluate model if requested
    if args.evaluate:
        logger.info("Starting model evaluation...")
        
        # Get the base model path and LoRA weights path
        base_model_path = config.model.model_name
        lora_weights_path = args.output_dir
        
        try:
            results = evaluate_model(base_model_path, lora_weights_path)
            logger.info(f"Evaluation results: {results}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()