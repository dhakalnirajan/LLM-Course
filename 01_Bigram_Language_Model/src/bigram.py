import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def train_bigram(text, epochs=100, learning_rate=1e-2, batch_size = 32, block_size = 8, save_path='model.pth'):
    """
    Trains the Bigram Language Model and saves the trained model.

    Args:
        text: The training text.
        epochs: The number of training epochs.
        learning_rate: The learning rate for the optimizer.
        save_path (str): Path to save the trained model.  Defaults to 'model.pth' in current dir.
    """

    # Create a vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # Create mapping from characters to integers and vice versa
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode the entire text
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
    data = torch.tensor(encode(text), dtype=torch.long)

    # --- Train and test splits ---
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    # --- Data Loader ---
    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data)-block_size, (batch_size, ))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    # --- Instantiate Model and Optimizer ---
    model = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    for epoch in range(epochs):
          # --- Get Batch ---
        xb, yb = get_batch("train")

        # --- Forward Pass & Loss ---
        logits, loss = model(xb, yb)

        # --- Backward Pass and Optimization ---
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (epoch+1) % (epochs//10) == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # --- Save the Model ---
    torch.save({
        'model_state_dict': model.state_dict(),
        'stoi': stoi,
        'itos': itos,
        'vocab_size': vocab_size,
        'block_size' : block_size,
        'batch_size': batch_size

    }, save_path)
    print(f"Model saved to {save_path}")

    return model, encode, decode # Return the trained model, encode, and decode functions


def load_bigram_model(model_path):
    """Loads a BigramLanguageModel from a saved checkpoint."""
    checkpoint = torch.load(model_path)
    model = BigramLanguageModel(checkpoint['vocab_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    block_size = checkpoint['block_size']
    # Create encode/decode functions based on loaded stoi/itos
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return model, encode, decode, block_size


def generate_text(model, encode, decode, start_str="", max_new_tokens=100):
    """Generates text using the trained bigram model."""

     # Start with an optional seed string
    if start_str:
        start_ids = encode(start_str)
        idx = torch.tensor(start_ids, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    else:
        # Start with a newline character if no seed string is provided.
        idx = torch.zeros((1, 1), dtype=torch.long) # Use 0 for newline character
    # Generate text
    generated_ids = model.generate(idx, max_new_tokens=max_new_tokens)
    generated_text = decode(generated_ids[0].tolist()) # Decode the generated indices
    return generated_text