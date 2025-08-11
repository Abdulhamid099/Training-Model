"""
Command-line interface for data processing.
"""
import argparse
import sys
from .data_processor import DataProcessor


def main():
    """Main data processing CLI function."""
    parser = argparse.ArgumentParser(description="Data processing utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Split dataset command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/eval/test')
    split_parser.add_argument('input_path', help='Input dataset file')
    split_parser.add_argument('--train-path', default='train.jsonl', help='Training set output path')
    split_parser.add_argument('--eval-path', default='eval.jsonl', help='Evaluation set output path')
    split_parser.add_argument('--test-path', help='Test set output path (optional)')
    split_parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio')
    split_parser.add_argument('--eval-ratio', type=float, default=0.1, help='Evaluation set ratio')
    
    args = parser.parse_args()
    
    if args.command == 'split':
        DataProcessor.split_dataset(
            input_path=args.input_path,
            train_path=args.train_path,
            eval_path=args.eval_path,
            test_path=args.test_path,
            train_ratio=args.train_ratio,
            eval_ratio=args.eval_ratio
        )
        print("Dataset split completed successfully!")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()