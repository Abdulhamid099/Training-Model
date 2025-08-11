#!/usr/bin/env python3
"""
Data preprocessing script for LLM fine-tuning.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config.training_config import get_config
from src.data.preprocessor import DataPreprocessor, create_sample_data


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/preprocessing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for LLM fine-tuning")
    parser.add_argument(
        "--input-file", 
        type=str, 
        required=True,
        help="Path to input data file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--create-sample", 
        action="store_true",
        help="Create sample data instead of processing input file"
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
    
    logger.info("Starting data preprocessing pipeline")
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = get_config(args.config)
    
    # Create sample data if requested
    if args.create_sample:
        logger.info("Creating sample data...")
        create_sample_data(args.input_file)
    
    # Initialize data preprocessor
    logger.info("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(config)
    
    # Preprocess data
    logger.info(f"Preprocessing data from {args.input_file}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        preprocessor.preprocess_pipeline(args.input_file, args.output_dir)
        logger.info("Data preprocessing completed successfully!")
        
        # Print summary
        output_path = Path(args.output_dir)
        for split_file in output_path.glob("*.jsonl"):
            with open(split_file, 'r') as f:
                line_count = sum(1 for _ in f)
            logger.info(f"{split_file.name}: {line_count} examples")
            
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()