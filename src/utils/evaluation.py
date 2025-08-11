"""
Evaluation utilities for LLM fine-tuning.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from pathlib import Path
import logging
from tqdm import tqdm
import re

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates fine-tuned models on various metrics."""
    
    def __init__(self, base_model_path: str, lora_weights_path: str):
        self.base_model_path = base_model_path
        self.lora_weights_path = lora_weights_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """Load the fine-tuned model."""
        logger.info("Loading fine-tuned model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.lora_weights_path)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def generate_response(self, prompt: str, max_length: int = 512, 
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate a response for a given prompt."""
        if self.model is None:
            self.load_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the response
        response = response[len(prompt):].strip()
        
        return response
    
    def evaluate_instruction_following(self, test_cases: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate model's ability to follow instructions."""
        if self.model is None:
            self.load_model()
        
        results = []
        
        for test_case in tqdm(test_cases, desc="Evaluating instruction following"):
            instruction = test_case['instruction']
            expected_output = test_case['expected_output']
            
            # Generate response
            response = self.generate_response(instruction)
            
            # Calculate similarity (simple word overlap for now)
            similarity = self._calculate_similarity(response, expected_output)
            results.append(similarity)
        
        return {
            'avg_similarity': np.mean(results),
            'std_similarity': np.std(results),
            'min_similarity': np.min(results),
            'max_similarity': np.max(results)
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple word overlap similarity
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def evaluate_perplexity(self, test_texts: List[str]) -> Dict[str, float]:
        """Calculate perplexity on test texts."""
        if self.model is None:
            self.load_model()
        
        perplexities = []
        
        for text in tqdm(test_texts, desc="Calculating perplexity"):
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
        
        return {
            'avg_perplexity': np.mean(perplexities),
            'std_perplexity': np.std(perplexities),
            'min_perplexity': np.min(perplexities),
            'max_perplexity': np.max(perplexities)
        }
    
    def run_comprehensive_evaluation(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a comprehensive evaluation on the model."""
        logger.info("Starting comprehensive evaluation...")
        
        results = {}
        
        # Instruction following evaluation
        if 'instruction_cases' in test_data:
            logger.info("Evaluating instruction following...")
            results['instruction_following'] = self.evaluate_instruction_following(
                test_data['instruction_cases']
            )
        
        # Perplexity evaluation
        if 'test_texts' in test_data:
            logger.info("Calculating perplexity...")
            results['perplexity'] = self.evaluate_perplexity(test_data['test_texts'])
        
        # Save results
        output_path = Path("experiments/evaluation_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
        return results


def create_test_cases() -> Dict[str, Any]:
    """Create sample test cases for evaluation."""
    instruction_cases = [
        {
            "instruction": "What is the capital of France?",
            "expected_output": "The capital of France is Paris."
        },
        {
            "instruction": "Explain photosynthesis in simple terms.",
            "expected_output": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."
        },
        {
            "instruction": "Write a short poem about coding.",
            "expected_output": "A poem about programming and coding."
        }
    ]
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science."
    ]
    
    return {
        'instruction_cases': instruction_cases,
        'test_texts': test_texts
    }


def evaluate_model(base_model_path: str, lora_weights_path: str, 
                  test_data: Optional[Dict[str, Any]] = None):
    """Convenience function to evaluate a model."""
    if test_data is None:
        test_data = create_test_cases()
    
    evaluator = ModelEvaluator(base_model_path, lora_weights_path)
    results = evaluator.run_comprehensive_evaluation(test_data)
    
    return results