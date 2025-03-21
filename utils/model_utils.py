import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import os

def count_parameters(model: nn.Module) -> int:
    """Counts the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module, init_type: str = 'xavier_uniform') -> None:
    """
    Initializes the weights of a PyTorch model.

    Args:
        model: The PyTorch model.
        init_type: The initialization method ('xavier_uniform', 'xavier_normal',
                   'kaiming_uniform', 'kaiming_normal', 'normal').
    """
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            if init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight)
            elif init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, 0.02)  # Customize mean and std
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            if init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight)
            elif init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.LayerNorm): #for layer normalization
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    model.apply(_init_weights)



def get_device() -> torch.device:
    """Checks if a GPU is available and returns the appropriate device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def move_to_device(data: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor], Tuple[torch.Tensor]], device: torch.device) -> Union[torch.Tensor, Dict[str, torch.Tensor],List[torch.Tensor], Tuple[torch.Tensor]]:
    """
    Recursively moves tensors in a nested structure to the specified device.  Handles tensors,
    lists, tuples, and dictionaries.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        return data  # Return unchanged if not a tensor, dict, list or tuple

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, filepath: str, **kwargs: Any) -> None:
    """
    Saves a model checkpoint.  Includes model state, optimizer state, epoch, and
    any other keyword arguments (e.g., loss, vocabulary).
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        **kwargs  # Add any other information
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model_class: type[nn.Module] = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Loads a model checkpoint.  Handles loading just the model weights or the
    entire checkpoint (including optimizer state).

    Args:
        filepath: Path to the checkpoint file.
        model_class: (Optional) The class of the model to be loaded. Required if
            you want to load the model weights into a new model instance.
        *args: Positional arguments to pass to the model constructor.
        **kwargs: Keyword arguments to pass to the model constructor.  Also used
            to update the loaded checkpoint (e.g., to override loaded values).

    Returns:
        A dictionary containing the loaded checkpoint data.  If `model_class` is
        provided, the dictionary will also contain 'model' (the loaded model)
        and potentially 'optimizer' (if an optimizer state was saved).
    """

    checkpoint = torch.load(filepath, map_location='cpu') #Load to CPU first

    # Update checkpoint with any provided kwargs (e.g., to override loaded values)
    checkpoint.update(kwargs)

    if model_class is not None:
        # Create a new model instance
        model = model_class(*args, **checkpoint)  # Pass checkpoint as kwargs
        model.load_state_dict(checkpoint['model_state_dict'])
        checkpoint['model'] = model

        if 'optimizer_state_dict' in checkpoint:
            # Create a dummy optimizer and load the state (for resuming training)
            optimizer = torch.optim.AdamW(model.parameters())  # Use a default optimizer
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            checkpoint['optimizer'] = optimizer
    return checkpoint

def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Gets the current learning rate from the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']