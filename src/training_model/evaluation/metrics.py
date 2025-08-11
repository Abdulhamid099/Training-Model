"""
Evaluation metrics for model performance assessment.
"""
import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics for the model."""
    predictions, labels = eval_pred
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # For causal LM, we need to compute perplexity
    # Remove -100 labels (padding tokens)
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    if len(valid_labels) == 0:
        return {"perplexity": float("inf")}
    
    # Calculate perplexity
    shift_logits = valid_predictions[..., :-1, :].reshape(-1, valid_predictions.shape[-1])
    shift_labels = valid_labels[..., 1:].reshape(-1)
    
    # Compute cross entropy loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(torch.tensor(shift_logits), torch.tensor(shift_labels))
    perplexity = torch.exp(loss).item()
    
    # Calculate accuracy
    predicted_tokens = np.argmax(shift_logits, axis=-1)
    accuracy = accuracy_score(shift_labels, predicted_tokens)
    
    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
    }


def calculate_bleu_score(predictions: list, references: list) -> float:
    """Calculate BLEU score for text generation."""
    try:
        from nltk.translate.bleu_score import sentence_bleu
        scores = []
        for pred, ref in zip(predictions, references):
            score = sentence_bleu([ref.split()], pred.split())
            scores.append(score)
        return np.mean(scores)
    except ImportError:
        print("NLTK not installed. Skipping BLEU score calculation.")
        return 0.0


def calculate_rouge_score(predictions: list, references: list) -> Dict[str, float]:
    """Calculate ROUGE scores for text generation."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            "rouge1": np.mean(rouge1_scores),
            "rouge2": np.mean(rouge2_scores),
            "rougeL": np.mean(rougeL_scores),
        }
    except ImportError:
        print("rouge-score not installed. Skipping ROUGE score calculation.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}