"""
Data preprocessing utilities for LLM fine-tuning.
"""
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import re

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing for LLM fine-tuning."""
    
    def __init__(self, config):
        self.config = config
        self.random_seed = config.data.random_seed
        
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from various file formats."""
        file_path = Path(file_path)
        
        if file_path.suffix == '.jsonl':
            return self._load_jsonl(file_path)
        elif file_path.suffix == '.json':
            return self._load_json(file_path)
        elif file_path.suffix == '.csv':
            return self._load_csv(file_path)
        elif file_path.suffix == '.txt':
            return self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # If it's a dict, try to find a list of examples
            for key in ['data', 'examples', 'samples', 'items']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # If no list found, treat the entire dict as one example
            return [data]
        else:
            raise ValueError("Invalid JSON structure")
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV file."""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def _load_txt(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load text file and convert to instruction format."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split by paragraphs or sections
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Convert to instruction format
        data = []
        for i, paragraph in enumerate(paragraphs):
            data.append({
                'instruction': f"Please summarize the following text:",
                'input': paragraph,
                'output': paragraph[:200] + "..." if len(paragraph) > 200 else paragraph
            })
        
        return data
    
    def format_for_training(self, data: List[Dict[str, Any]], 
                          instruction_template: str = None) -> List[Dict[str, str]]:
        """Format data for training with instruction template."""
        if instruction_template is None:
            instruction_template = self._get_default_template()
        
        formatted_data = []
        
        for item in data:
            # Handle different data formats
            if 'instruction' in item and 'output' in item:
                # Already in instruction format
                instruction = item['instruction']
                input_text = item.get('input', '')
                output = item['output']
            elif 'question' in item and 'answer' in item:
                # Q&A format
                instruction = item['question']
                input_text = ''
                output = item['answer']
            elif 'prompt' in item and 'completion' in item:
                # Prompt-completion format
                instruction = item['prompt']
                input_text = ''
                output = item['completion']
            elif 'text' in item:
                # Plain text format
                instruction = "Please continue the following text:"
                input_text = item['text']
                output = item.get('continuation', '')
            else:
                # Try to infer format
                keys = list(item.keys())
                if len(keys) >= 2:
                    instruction = str(item[keys[0]])
                    input_text = ''
                    output = str(item[keys[1]])
                else:
                    logger.warning(f"Skipping item with unclear format: {item}")
                    continue
            
            # Format with template
            formatted_text = instruction_template.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
            
            formatted_data.append({
                'text': formatted_text
            })
        
        return formatted_data
    
    def _get_default_template(self) -> str:
        """Get default instruction template for Mistral."""
        return """<s>[INST] {instruction}

{input} [/INST] {output}</s>"""
    
    def split_data(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Split data into train/validation/test sets."""
        if self.config.data.max_samples:
            data = data[:self.config.data.max_samples]
        
        # Split data
        train_data, temp_data = train_test_split(
            data, 
            test_size=1 - self.config.data.train_split,
            random_state=self.random_seed
        )
        
        if self.config.data.test_split > 0:
            val_ratio = self.config.data.validation_split / (self.config.data.validation_split + self.config.data.test_split)
            val_data, test_data = train_test_split(
                temp_data,
                test_size=1 - val_ratio,
                random_state=self.random_seed
            )
        else:
            val_data = temp_data
            test_data = []
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
    
    def save_jsonl(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """Save data to JSONL format."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def create_dataset_dict(self, split_data: Dict[str, List[Dict[str, Any]]]) -> DatasetDict:
        """Create HuggingFace DatasetDict from split data."""
        dataset_dict = {}
        
        for split_name, split_items in split_data.items():
            if split_items:  # Only create dataset if there are items
                dataset_dict[split_name] = Dataset.from_list(split_items)
        
        return DatasetDict(dataset_dict)
    
    def preprocess_pipeline(self, input_file: str, output_dir: str) -> None:
        """Complete preprocessing pipeline."""
        logger.info(f"Loading data from {input_file}")
        raw_data = self.load_data(input_file)
        
        logger.info(f"Formatting {len(raw_data)} examples for training")
        formatted_data = self.format_for_training(raw_data)
        
        logger.info("Splitting data into train/validation/test sets")
        split_data = self.split_data(formatted_data)
        
        # Save processed data
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_items in split_data.items():
            if split_items:
                output_file = output_path / f"{split_name}.jsonl"
                self.save_jsonl(split_items, str(output_file))
                logger.info(f"Saved {len(split_items)} examples to {output_file}")
        
        # Create and save dataset dict
        dataset_dict = self.create_dataset_dict(split_data)
        dataset_dict.save_to_disk(str(output_path / "dataset"))
        logger.info(f"Saved dataset to {output_path / 'dataset'}")


def create_sample_data(output_file: str = "data/raw/sample_data.jsonl"):
    """Create sample data for testing."""
    sample_data = [
        {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris."
        },
        {
            "instruction": "Explain photosynthesis in simple terms.",
            "input": "",
            "output": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."
        },
        {
            "instruction": "Write a short poem about coding.",
            "input": "",
            "output": "Lines of logic flow like streams,\nFunctions dance in digital dreams,\nBugs may hide in shadows deep,\nBut clean code helps us sleep."
        },
        {
            "instruction": "Translate the following to Spanish:",
            "input": "Hello, how are you today?",
            "output": "Hola, ¿cómo estás hoy?"
        },
        {
            "instruction": "Summarize the main benefits of exercise.",
            "input": "",
            "output": "Exercise provides numerous benefits including improved cardiovascular health, stronger muscles and bones, better mental health, increased energy levels, and weight management."
        }
    ]
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created sample data at {output_file}")