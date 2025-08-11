"""
Training configuration classes for model fine-tuning.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml
import os


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    
    # Model configuration
    model_name: str = "microsoft/DialoGPT-medium"
    model_path: Optional[str] = None
    cache_dir: Optional[str] = None
    
    # Dataset configuration
    dataset_path: str = "data/processed"
    train_file: str = "train.jsonl"
    eval_file: Optional[str] = "eval.jsonl"
    test_file: Optional[str] = "test.jsonl"
    max_length: int = 512
    
    # Training parameters
    output_dir: str = "outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Precision and optimization
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # Evaluation and logging
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Logging and monitoring
    report_to: Optional[str] = None  # "wandb", "tensorboard", or None
    run_name: Optional[str] = None
    project_name: str = "llm-fine-tuning"
    
    # Hardware configuration
    use_cpu: bool = False
    device_map: str = "auto"
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Miscellaneous
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 3
    
    def save_config(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.__dict__.copy()


@dataclass
class ModelConfig:
    """Configuration for different model types."""
    
    # Mistral-7B configuration
    MISTRAL_7B = {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "max_length": 8192,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_r": 16,
        "lora_alpha": 32,
    }
    
    # Llama2 configuration
    LLAMA2_7B = {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "max_length": 4096,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_r": 16,
        "lora_alpha": 32,
    }
    
    # Falcon configuration
    FALCON_7B = {
        "model_name": "tiiuae/falcon-7b-instruct",
        "max_length": 2048,
        "lora_target_modules": ["query_key_value"],
        "lora_r": 8,
        "lora_alpha": 16,
    }
    
    # GPT-2 configuration (for testing)
    GPT2_MEDIUM = {
        "model_name": "gpt2-medium",
        "max_length": 1024,
        "lora_target_modules": ["c_attn"],
        "lora_r": 8,
        "lora_alpha": 16,
    }