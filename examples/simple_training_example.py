#!/usr/bin/env python3
"""
Simple training example demonstrating basic usage of the training pipeline.
"""
import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training_model.configs.training_config import TrainingConfig
from training_model.training.trainer import ModelTrainer


def create_sample_dataset():
    """Create a simple sample dataset for demonstration."""
    
    # Create datasets directory
    os.makedirs('datasets', exist_ok=True)
    
    # Sample data
    sample_data = [
        {
            "instruction": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
        },
        {
            "instruction": "Explain neural networks.",
            "response": "Neural networks are computing systems inspired by biological neural networks that consist of interconnected nodes processing information."
        },
        {
            "instruction": "What is deep learning?",
            "response": "Deep learning is a subset of machine learning using neural networks with multiple layers to learn complex patterns from data."
        },
        {
            "instruction": "How does gradient descent work?",
            "response": "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function by following gradients."
        }
    ]
    
    # Save training data
    with open('datasets/train.jsonl', 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    # Save eval data (first 2 examples)
    with open('datasets/eval.jsonl', 'w') as f:
        for item in sample_data[:2]:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created sample dataset with {len(sample_data)} training examples")


def main():
    """Main training function."""
    
    print("🚀 Simple Training Example")
    print("=" * 50)
    
    # Create sample dataset
    print("📊 Creating sample dataset...")
    create_sample_dataset()
    
    # Configure training for a small model
    print("⚙️  Setting up training configuration...")
    config = TrainingConfig(
        model_name="gpt2",  # Small model for demo
        dataset_path="datasets",
        train_file="train.jsonl",
        eval_file="eval.jsonl",
        output_dir="outputs/simple_example",
        num_train_epochs=1,  # Quick training
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_length=256,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        fp16=False,  # Compatibility
        evaluation_strategy="steps",
        eval_steps=2,
        save_steps=2,
        logging_steps=1,
    )
    
    print(f"📋 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Output: {config.output_dir}")
    print(f"   Epochs: {config.num_train_epochs}")
    print(f"   LoRA: {config.use_lora} (r={config.lora_r})")
    
    # Initialize trainer
    print("\n🔧 Initializing trainer...")
    trainer = ModelTrainer(config)
    
    # Run training pipeline
    print("\n🎯 Starting training pipeline...")
    try:
        trainer.run_full_training_pipeline()
        print("\n✅ Training completed successfully!")
        print(f"   Model saved to: {config.output_dir}")
        
        # Test the model
        print("\n🧪 Testing the fine-tuned model...")
        test_model(config.output_dir)
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise


def test_model(model_path):
    """Test the fine-tuned model with a simple generation."""
    try:
        from transformers import pipeline
        
        # Load the fine-tuned model
        generator = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            max_length=100,
            temperature=0.7,
        )
        
        # Test prompt
        test_prompt = "What is artificial intelligence?"
        print(f"   Prompt: {test_prompt}")
        
        # Generate response
        response = generator(test_prompt, max_new_tokens=50)
        print(f"   Response: {response[0]['generated_text']}")
        
    except Exception as e:
        print(f"   Could not test model: {str(e)}")


if __name__ == "__main__":
    main()