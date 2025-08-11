"""
Main training module for fine-tuning language models with LoRA support.
"""
import os
import logging
import torch
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import wandb

from ..configs.training_config import TrainingConfig
from ..data.data_processor import DataProcessor
from ..utils.logging_utils import setup_logging
from ..evaluation.metrics import compute_metrics


class ModelTrainer:
    """Main trainer class for fine-tuning language models."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the trainer with configuration."""
        self.config = config
        self.logger = setup_logging()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        
        # Setup logging
        if config.report_to == "wandb":
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=config.to_dict()
            )
    
    def load_model_and_tokenizer(self) -> None:
        """Load the pretrained model and tokenizer."""
        self.logger.info(f"Loading model and tokenizer: {self.config.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True,
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "cache_dir": self.config.cache_dir,
                "trust_remote_code": True,
            }
            
            if not self.config.use_cpu:
                model_kwargs.update({
                    "torch_dtype": torch.float16 if self.config.fp16 else torch.float32,
                    "device_map": self.config.device_map,
                })
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Setup LoRA if enabled
            if self.config.use_lora:
                self._setup_lora()
            
            self.logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _setup_lora(self) -> None:
        """Setup LoRA configuration for the model."""
        self.logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def load_datasets(self) -> Dict[str, Any]:
        """Load and preprocess training datasets."""
        self.logger.info("Loading datasets")
        
        processor = DataProcessor(self.tokenizer, self.config.max_length)
        
        # Load training dataset
        train_dataset = processor.load_and_process_dataset(
            os.path.join(self.config.dataset_path, self.config.train_file)
        )
        
        datasets = {"train": train_dataset}
        
        # Load evaluation dataset if specified
        if self.config.eval_file:
            eval_dataset = processor.load_and_process_dataset(
                os.path.join(self.config.dataset_path, self.config.eval_file)
            )
            datasets["eval"] = eval_dataset
        
        # Load test dataset if specified
        if self.config.test_file:
            test_dataset = processor.load_and_process_dataset(
                os.path.join(self.config.dataset_path, self.config.test_file)
            )
            datasets["test"] = test_dataset
        
        self.logger.info(f"Loaded datasets: {list(datasets.keys())}")
        return datasets
    
    def setup_trainer(self, datasets: Dict[str, Any]) -> None:
        """Setup the Hugging Face trainer."""
        self.logger.info("Setting up trainer")
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            report_to=self.config.report_to,
            run_name=self.config.run_name,
            seed=self.config.seed,
            save_total_limit=self.config.save_total_limit,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Callbacks
        callbacks = []
        if "eval" in datasets:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold,
                )
            )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("eval"),
            data_collator=data_collator,
            compute_metrics=compute_metrics if "eval" in datasets else None,
            callbacks=callbacks,
        )
    
    def train(self) -> None:
        """Run the training process."""
        self.logger.info("Starting training")
        
        try:
            # Resume from checkpoint if specified
            resume_from_checkpoint = self.config.resume_from_checkpoint
            
            # Train the model
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Save the trained model
            self.trainer.save_model()
            
            # Log training results
            self.logger.info("Training completed successfully")
            self.logger.info(f"Training results: {train_result}")
            
            # Save training metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def evaluate(self, dataset_name: str = "eval") -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_trainer first.")
        
        self.logger.info(f"Evaluating on {dataset_name} dataset")
        
        eval_result = self.trainer.evaluate()
        
        self.logger.info(f"Evaluation results: {eval_result}")
        self.trainer.log_metrics(dataset_name, eval_result)
        self.trainer.save_metrics(dataset_name, eval_result)
        
        return eval_result
    
    def save_model(self, save_path: Optional[str] = None) -> None:
        """Save the trained model."""
        if save_path is None:
            save_path = self.config.output_dir
        
        self.logger.info(f"Saving model to {save_path}")
        
        if self.config.use_lora:
            # Save LoRA adapters
            self.model.save_pretrained(save_path)
        else:
            # Save full model
            self.trainer.save_model(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save configuration
        self.config.save_config(os.path.join(save_path, "training_config.yaml"))
    
    def run_full_training_pipeline(self) -> None:
        """Run the complete training pipeline."""
        try:
            # Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Load datasets
            datasets = self.load_datasets()
            
            # Setup trainer
            self.setup_trainer(datasets)
            
            # Train model
            self.train()
            
            # Evaluate if eval dataset exists
            if "eval" in datasets:
                self.evaluate()
            
            # Save final model
            self.save_model()
            
            self.logger.info("Training pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise
        finally:
            # Clean up wandb if used
            if self.config.report_to == "wandb":
                wandb.finish()