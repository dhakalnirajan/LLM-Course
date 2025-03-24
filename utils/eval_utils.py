import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge
import numpy as np
from utils.data_utils import decode_text #Import decode_text


def calculate_perplexity(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """Calculates the perplexity of a language model."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            total_loss += loss.item() * xb.size(0) * xb.size(1) #loss is mean, so multiply by batch size and block size
            total_tokens += xb.size(0) * xb.size(1)

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def calculate_accuracy(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """Calculates the accuracy of a model on a classification task."""
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)  # Assuming the model returns logits
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == yb).sum().item()
            total_predictions += yb.numel()

    return correct_predictions / total_predictions


def generate_text(model: torch.nn.Module, stoi:Dict, itos: Dict, start_tokens: Optional[List[int]] = None,
                  max_length: int = 100, device: torch.device = torch.device('cpu'),
                  temperature: float = 1.0, top_k: Optional[int] = None,
                  top_p: Optional[float] = None) -> str:
    """
    Generates text from a language model using various decoding strategies.

    Args:
        model: The PyTorch language model.
        stoi: String to int
        itos: int to string.
        start_tokens: Optional list of starting token indices.  If None, starts with 0.
        max_length: The maximum length of the generated sequence.
        device: The device to use (CPU or GPU).
        temperature:  Controls the randomness of the sampling (higher = more random).
        top_k:  If set, only considers the top k most likely tokens at each step.
        top_p: If set, only considers tokens whose cumulative probability exceeds p.

    Returns:
        The generated text as a string.
    """
    model.eval()
    model.to(device) #Ensure model is on the correct device.

    if start_tokens is None:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with a zero token
    else:
        start_tokens_tensor = torch.tensor(start_tokens, dtype=torch.long, device=device)
        idx = start_tokens_tensor.unsqueeze(0)  # Add a batch dimension


    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(idx)
            logits = logits[:, -1, :] / temperature  # Scale by temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')  # Filter out tokens
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

    generated_text = decode_text(idx[0].tolist(), itos) # Decode
    return generated_text



def compute_bleu_score(predictions: List[str], references: List[List[str]]) -> float:
    """
    Computes the BLEU score.

    Args:
        predictions:  A list of predicted sentences (strings).
        references: A list of lists of reference sentences.  Each inner list
            represents the possible reference translations for a single
            predicted sentence.

    Returns:
        The BLEU score (a float between 0 and 1).
    """
    # Tokenize predictions and references
    tokenized_predictions = [pred.split() for pred in predictions]
    tokenized_references = [[ref.split() for ref in refs] for refs in references]
    return corpus_bleu(tokenized_references, tokenized_predictions)


def compute_rouge_score(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    """Computes ROUGE scores using the 'rouge' library."""
    rouge = Rouge()
    # Ensure that predictions and references are not empty, and handle cases with single strings
    if not predictions or not references:
        return {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}

    # The rouge library expects strings, not lists of strings
    # Join lists into single strings if necessary
    if isinstance(predictions, list):
        predictions = [' '.join(pred.split()) for pred in predictions]  # Remove extra spaces
    if isinstance(references, list):
        references = [' '.join(ref.split()) for ref in references]      # Remove extra spaces

    scores = rouge.get_scores(predictions, references, avg=True) # Get average scores
    return scores