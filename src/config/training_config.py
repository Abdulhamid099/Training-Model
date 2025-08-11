"""
Training configuration for LLM fine-tuning experiments.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import yaml
import os


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    model_type: str = "mistral"
    max_length: int = 2048
    use_flash_attention: bool = True
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Basic training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: Optional[int] = None
    
    # Optimization
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    bf16: bool = False
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    
    # Data
    train_file: str = "data/processed/train.jsonl"
    validation_file: str = "data/processed/validation.jsonl"
    max_seq_length: int = 2048
    dataloader_pin_memory: bool = False
    
    # Output
    output_dir: str = "experiments/outputs"
    run_name: str = "mistral-finetune"
    
    # Experiment tracking
    report_to: List[str] = None
    
    def __post_init__(self):
        if self.report_to is None:
            self.report_to = ["tensorboard"]


@dataclass
class DataConfig:
    """Configuration for data processing."""
    train_split: float = 0.9
    validation_split: float = 0.1
    test_split: float = 0.0
    random_seed: int = 42
    max_samples: Optional[int] = None
    preprocessing_num_workers: int = 4


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = None
    lora: LoRAConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create nested config objects
        model_config = ModelConfig(**config_dict.get('model', {}))
        lora_config = LoRAConfig(**config_dict.get('lora', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        return cls(
            model=model_config,
            lora=lora_config,
            training=training_config,
            data=data_config
        )
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'lora': self.lora.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Default configuration
DEFAULT_CONFIG = ExperimentConfig()


def get_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """Get configuration from file or return default."""
    if config_path and os.path.exists(config_path):
        return ExperimentConfig.from_yaml(config_path)
    return DEFAULT_CONFIG