#!/usr/bin/env python3
"""
Inference script for using fine-tuned LLM models.
"""
import argparse
import logging
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class ModelInference:
    """Class for running inference with fine-tuned models."""
    
    def __init__(self, base_model_path: str, lora_weights_path: str):
        self.base_model_path = base_model_path
        self.lora_weights_path = lora_weights_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """Load the fine-tuned model."""
        logging.info("Loading fine-tuned model...")
        
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
        
        logging.info("Model loaded successfully")
    
    def generate_response(self, prompt: str, max_length: int = 512, 
                         temperature: float = 0.7, top_p: float = 0.9,
                         do_sample: bool = True) -> str:
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
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the response
        response = response[len(prompt):].strip()
        
        return response
    
    def interactive_mode(self):
        """Run interactive inference mode."""
        if self.model is None:
            self.load_model()
        
        print("Interactive inference mode. Type 'quit' to exit.")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Generate response
                response = self.generate_response(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned LLM")
    parser.add_argument(
        "--base-model", 
        type=str, 
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model path or HuggingFace model name"
    )
    parser.add_argument(
        "--lora-weights", 
        type=str, 
        default="experiments/outputs",
        help="Path to LoRA weights"
    )
    parser.add_argument(
        "--prompt", 
        type=str,
        help="Single prompt to generate response for"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=512,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Generation temperature"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.9,
        help="Top-p sampling parameter"
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
    
    # Initialize inference
    inference = ModelInference(args.base_model, args.lora_weights)
    
    if args.interactive:
        # Interactive mode
        inference.interactive_mode()
    elif args.prompt:
        # Single prompt mode
        logger.info(f"Generating response for: {args.prompt}")
        response = inference.generate_response(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}")
    else:
        # Default interactive mode
        print("No prompt provided. Starting interactive mode...")
        inference.interactive_mode()


if __name__ == "__main__":
    main()