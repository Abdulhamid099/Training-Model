from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.data.build_dataset import build_hf_dataset


console = Console()


@dataclass
class Config:
    base_model_name_or_path: str
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"

    train_jsonl_path: str = "data/sample.jsonl"
    text_template: Optional[str] = None

    output_dir: str = "outputs/mistral7b-lora-demo"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 1
    bf16: bool = False
    fp16: bool = True
    seed: int = 42

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None


DTYPE_MAP = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def maybe_get_bnb_config(cfg: Config) -> Optional[BitsAndBytesConfig]:
    if not cfg.use_4bit:
        return None
    try:
        from transformers import BitsAndBytesConfig  # noqa: F401
    except Exception as e:  # pragma: no cover - informative path
        console.print("[yellow]bitsandbytes not available; falling back to full precision.[/yellow]")
        return None

    compute_dtype = getattr(__import__("torch"), "float16") if cfg.bnb_4bit_compute_dtype == "float16" else getattr(__import__("torch"), "bfloat16")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def main(config_path: str = "configs/mistral_lora_example.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    console.rule("Config")
    console.print(cfg)

    bnb_config = maybe_get_bnb_config(cfg)

    console.rule("Model & Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if bnb_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name_or_path,
            trust_remote_code=True,
        )

    lora_targets = cfg.target_modules or [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=lora_targets,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)

    console.rule("Dataset")
    dataset = build_hf_dataset(cfg.train_jsonl_path, cfg.text_template)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        report_to=[],
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    console.rule("Training")
    trainer.train()

    console.rule("Saving")
    trainer.save_model()
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mistral_lora_example.yaml")
    args = parser.parse_args()
    main(args.config)