#tests/test_bigram.py
import pytest
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

import bigram  # Import your bigram module
import data_utils


@pytest.fixture
def sample_text():
    return "this is a test"

@pytest.fixture
def vocab(sample_text):
    stoi, itos = data_utils.create_vocabulary(sample_text)
    return stoi, itos

def test_bigram_model_creation(vocab):
    stoi, itos = vocab
    vocab_size = len(stoi)
    model = bigram.BigramLanguageModel(vocab_size)
    assert model.token_embedding_table.weight.shape == (vocab_size, vocab_size)

def test_bigram_forward_pass(vocab):
    stoi, itos = vocab
    vocab_size = len(stoi)
    model = bigram.BigramLanguageModel(vocab_size)
    idx = torch.tensor([[0, 1, 2]], dtype=torch.long)  # Example input
    logits, loss = model(idx)
    assert logits.shape == (1 * 3, vocab_size)  # (B*T, C)
    assert loss is None  # No targets provided

    targets = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logits, loss = model(idx, targets=targets)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)

def test_bigram_generate(vocab):
    stoi, itos = vocab
    vocab_size = len(stoi)
    model = bigram.BigramLanguageModel(vocab_size)
    # Initialize weights for deterministic output (for testing purposes)
    torch.nn.init.constant_(model.token_embedding_table.weight, 0.1)
    idx = torch.zeros((1, 1), dtype=torch.long) # Start with token 0
    generated_ids = model.generate(idx, max_new_tokens=3)
    assert generated_ids.shape == (1, 4)  # Initial token + 3 new tokens
    #With constant initialization all tokens should have the same logit value
    #and the selection will depend on the implementation.
    #Here we just verify the size.

def test_train_bigram(sample_text, tmp_path): #tmp_path is a built-in fixture
    #Test training and saving
    model, _, _ = bigram.train_bigram(sample_text, epochs=2, save_path=tmp_path / "model.pth")
    assert os.path.exists(tmp_path / "model.pth")