"""
Data processing utilities for preparing datasets for model training.
"""
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer
import logging


class DataProcessor:
    """Data processor for handling different dataset formats."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """Initialize the data processor."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
    
    def load_and_process_dataset(self, file_path: str) -> Dataset:
        """Load and process a dataset from file."""
        self.logger.info(f"Loading dataset from {file_path}")
        
        if file_path.endswith('.jsonl'):
            return self._load_jsonl_dataset(file_path)
        elif file_path.endswith('.json'):
            return self._load_json_dataset(file_path)
        elif file_path.endswith('.csv'):
            return self._load_csv_dataset(file_path)
        elif file_path.endswith('.txt'):
            return self._load_text_dataset(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def _load_jsonl_dataset(self, file_path: str) -> Dataset:
        """Load dataset from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Skipping invalid JSON line: {e}")
        
        return self._process_conversation_data(data)
    
    def _load_json_dataset(self, file_path: str) -> Dataset:
        """Load dataset from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return self._process_conversation_data(data)
        else:
            raise ValueError("JSON file should contain a list of examples")
    
    def _load_csv_dataset(self, file_path: str) -> Dataset:
        """Load dataset from CSV file."""
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
        return self._process_conversation_data(data)
    
    def _load_text_dataset(self, file_path: str) -> Dataset:
        """Load dataset from plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into chunks for training
        chunks = self._split_text_into_chunks(text)
        data = [{"text": chunk} for chunk in chunks]
        
        return self._process_text_data(data)
    
    def _process_conversation_data(self, data: List[Dict[str, Any]]) -> Dataset:
        """Process conversation-style data for instruction tuning."""
        processed_data = []
        
        for example in data:
            # Handle different conversation formats
            if "conversations" in example:
                # Multi-turn conversation format
                text = self._format_conversation(example["conversations"])
            elif "instruction" in example and "response" in example:
                # Instruction-response format
                text = self._format_instruction_response(
                    example.get("instruction", ""),
                    example.get("input", ""),
                    example.get("response", "")
                )
            elif "prompt" in example and "completion" in example:
                # Prompt-completion format
                text = f"{example['prompt']}{example['completion']}"
            elif "text" in example:
                # Simple text format
                text = example["text"]
            else:
                self.logger.warning(f"Unrecognized format: {example}")
                continue
            
            # Tokenize and add to processed data
            tokenized = self._tokenize_text(text)
            if tokenized:
                processed_data.append(tokenized)
        
        return Dataset.from_list(processed_data)
    
    def _process_text_data(self, data: List[Dict[str, Any]]) -> Dataset:
        """Process simple text data."""
        processed_data = []
        
        for example in data:
            text = example["text"]
            tokenized = self._tokenize_text(text)
            if tokenized:
                processed_data.append(tokenized)
        
        return Dataset.from_list(processed_data)
    
    def _format_conversation(self, conversations: List[Dict[str, str]]) -> str:
        """Format multi-turn conversation data."""
        formatted_text = ""
        
        for turn in conversations:
            role = turn.get("from", "").lower()
            content = turn.get("value", "")
            
            if role in ["human", "user"]:
                formatted_text += f"Human: {content}\n"
            elif role in ["assistant", "gpt"]:
                formatted_text += f"Assistant: {content}\n"
            else:
                formatted_text += f"{role}: {content}\n"
        
        return formatted_text.strip()
    
    def _format_instruction_response(self, instruction: str, input_text: str, response: str) -> str:
        """Format instruction-response data."""
        if input_text:
            formatted_text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
        else:
            formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        
        return formatted_text
    
    def _tokenize_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Tokenize text and prepare for training."""
        if not text or not text.strip():
            return None
        
        # Tokenize the text
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )
        
        # For causal language modeling, labels are the same as input_ids
        encoded["labels"] = encoded["input_ids"].copy()
        
        return encoded
    
    def _split_text_into_chunks(self, text: str, chunk_overlap: int = 50) -> List[str]:
        """Split long text into overlapping chunks."""
        # Simple chunking by character count
        # In practice, you might want more sophisticated chunking
        chunk_size = self.max_length * 3  # Rough estimate for characters
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence ending
                for i in range(end, max(start, end - 200), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
        
        return chunks
    
    @staticmethod
    def create_conversation_dataset(
        conversations: List[List[Dict[str, str]]], 
        output_path: str
    ) -> None:
        """Create a conversation dataset in JSONL format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for conversation in conversations:
                example = {"conversations": conversation}
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    @staticmethod
    def create_instruction_dataset(
        instructions: List[Dict[str, str]], 
        output_path: str
    ) -> None:
        """Create an instruction-response dataset in JSONL format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for instruction_data in instructions:
                f.write(json.dumps(instruction_data, ensure_ascii=False) + '\n')
    
    @staticmethod
    def split_dataset(
        input_path: str, 
        train_path: str, 
        eval_path: str, 
        test_path: Optional[str] = None,
        train_ratio: float = 0.8,
        eval_ratio: float = 0.1
    ) -> None:
        """Split a dataset into train/eval/test sets."""
        # Load data
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        # Calculate split indices
        total = len(data)
        train_end = int(total * train_ratio)
        eval_end = train_end + int(total * eval_ratio)
        
        # Split data
        train_data = data[:train_end]
        eval_data = data[train_end:eval_end]
        test_data = data[eval_end:] if test_path else []
        
        # Save splits
        def save_split(split_data: List[Dict], path: str):
            with open(path, 'w', encoding='utf-8') as f:
                for example in split_data:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        save_split(train_data, train_path)
        save_split(eval_data, eval_path)
        
        if test_path and test_data:
            save_split(test_data, test_path)
        
        print(f"Dataset split: {len(train_data)} train, {len(eval_data)} eval"
              + (f", {len(test_data)} test" if test_data else ""))