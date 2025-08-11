"""
Training module for LLM fine-tuning with LoRA.
"""
import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, DatasetDict, load_dataset
import wandb
from typing import Optional, Dict, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMTrainer:
    """Main training class for LLM fine-tuning."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Setup experiment tracking
        if "wandb" in config.training.report_to:
            self._setup_wandb()
    
    def _setup_wandb(self):
        """Setup Weights & Biases for experiment tracking."""
        try:
            wandb.init(
                project="llm-finetuning",
                name=self.config.training.run_name,
                config={
                    "model": self.config.model.__dict__,
                    "lora": self.config.lora.__dict__,
                    "training": self.config.training.__dict__,
                    "data": self.config.data.__dict__
                }
            )
            logger.info("WandB initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.config.model.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            torch_dtype=torch.float16 if self.config.training.fp16 else torch.float32,
            trust_remote_code=self.config.model.trust_remote_code,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=True if self.config.training.fp16 else False,
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora(self):
        """Setup LoRA configuration and apply to model."""
        logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied successfully")
    
    def load_dataset(self, train_file: str, validation_file: str) -> DatasetDict:
        """Load and prepare dataset for training."""
        logger.info("Loading dataset")
        
        # Load datasets
        train_dataset = load_dataset('json', data_files=train_file)['train']
        val_dataset = load_dataset('json', data_files=validation_file)['train']
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.config.training.max_seq_length,
                return_tensors="pt"
            )
        
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments."""
        return TrainingArguments(
            output_dir=self.config.training.output_dir,
            run_name=self.config.training.run_name,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=self.config.training.warmup_steps,
            max_steps=self.config.training.max_steps,
            optim=self.config.training.optim,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps,
            save_total_limit=self.config.training.save_total_limit,
            evaluation_strategy=self.config.training.evaluation_strategy,
            save_strategy=self.config.training.save_strategy,
            dataloader_pin_memory=self.config.training.dataloader_pin_memory,
            report_to=self.config.training.report_to,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
        )
    
    def setup_data_collator(self):
        """Setup data collator for training."""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
    
    def train(self, train_file: str, validation_file: str):
        """Main training function."""
        logger.info("Starting training process")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Setup LoRA
        self.setup_lora()
        
        # Load dataset
        dataset = self.load_dataset(train_file, validation_file)
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Setup data collator
        data_collator = self.setup_data_collator()
        
        # Setup trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        logger.info("Starting training...")
        self.trainer.train()
        
        # Save final model
        self.save_model()
        
        logger.info("Training completed successfully")
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save the trained model."""
        if output_dir is None:
            output_dir = self.config.training.output_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        self.trainer.save_model(str(output_path))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(output_path))
        
        # Save LoRA config
        lora_config_path = output_path / "lora_config.json"
        with open(lora_config_path, 'w') as f:
            json.dump(self.config.lora.__dict__, f, indent=2)
        
        # Save training config
        config_path = output_path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.training.__dict__, f, indent=2)
        
        logger.info(f"Model saved to {output_path}")
    
    def evaluate(self, test_file: Optional[str] = None):
        """Evaluate the trained model."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Run train() first.")
        
        if test_file:
            # Load test dataset
            test_dataset = load_dataset('json', data_files=test_file)['train']
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding=True,
                    max_length=self.config.training.max_seq_length,
                    return_tensors="pt"
                )
            
            test_dataset = test_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=test_dataset.column_names
            )
            
            # Evaluate on test set
            results = self.trainer.evaluate(test_dataset)
        else:
            # Evaluate on validation set
            results = self.trainer.evaluate()
        
        logger.info(f"Evaluation results: {results}")
        return results


def train_model(config_path: str, train_file: str, validation_file: str):
    """Convenience function to train a model."""
    from src.config.training_config import get_config
    
    # Load configuration
    config = get_config(config_path)
    
    # Initialize trainer
    trainer = LLMTrainer(config)
    
    # Train model
    trainer.train(train_file, validation_file)
    
    return trainer