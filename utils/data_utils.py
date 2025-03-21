import torch
from torch.utils.data import DataLoader, Dataset
import os
import requests
import pandas as pd
import json
from typing import List, Dict, Tuple, Union, Any

class TextDataset(Dataset):
    """
    A PyTorch Dataset for handling text data with a fixed block size.
    """
    def __init__(self, data: List[int], block_size: int):
        """
        Args:
            data: A list of integer-encoded tokens.
            block_size: The context window size.
        """
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_text_data(filepath: str, encoding: str = 'utf-8') -> str:
    """
    Loads text data from a file.  Handles different file types.

    Args:
        filepath: Path to the text file.
        encoding: File encoding (default: utf-8).

    Returns:
        The text content as a single string.

    Raises:
        ValueError: If the file type is not supported.
    """
    _, ext = os.path.splitext(filepath)
    try:
        if ext == '.txt':
            with open(filepath, 'r', encoding=encoding) as f:
                text = f.read()
        elif ext == '.csv':
            df = pd.read_csv(filepath)
            text = ' '.join(df.iloc[:, 0].tolist()) # Assumes text is in first column
        elif ext == '.jsonl':
            data = []
            with open(filepath, 'r', encoding=encoding) as f:
                for line in f:
                    data.append(json.loads(line))
            text = ' '.join([item['text'] for item in data]) # Assumes a 'text' key
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return text
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise  # Re-raise the exception to halt execution
    except Exception as e:
        print(f"An error occurred while loading data from {filepath}: {e}")
        raise

def create_vocabulary(text: str, min_freq: int = 1) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Creates a vocabulary (character-level) from text.

    Args:
        text: The input text.
        min_freq: Minimum frequency for a token to be included.

    Returns:
        A tuple containing:
            stoi: A dictionary mapping characters to integer indices.
            itos: A dictionary mapping integer indices to characters.
    """
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos

def encode_text(text: str, stoi: Dict[str, int]) -> List[int]:
    """Encodes text using a string-to-integer mapping."""
    return [stoi.get(c, 0) for c in text]  # Handle unknown chars with index 0


def decode_text(encoded_text: List[int], itos: Dict[int, str]) -> str:
    """Decodes a sequence of integers back to text."""
    return ''.join([itos.get(i, '<UNK>') for i in encoded_text]) #Handles unknown indices


def split_data(data: List[Any], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[List[Any], List[Any], List[Any]]:
    """Splits data into train, validation, and test sets."""
    assert train_ratio + val_ratio + test_ratio == 1.0
    n = len(data)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    return train_data, val_data, test_data


def get_data_loader(data: List[int], batch_size: int, shuffle: bool = True, block_size: int = None) -> DataLoader:
    """Creates a DataLoader for the given data."""
    if block_size:
        dataset = TextDataset(data, block_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def download_dataset(url: str, save_path: str) -> None:
    """Downloads a dataset from a URL and saves it locally."""
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)