# Chapter 1: Bigram Language Model

This chapter introduces the concept of language modeling and implements a simple bigram language model.

## What is a Language Model?

A language model is a probability distribution over sequences of words (or characters, or other tokens).  Given a sequence of words, it assigns a probability to the entire sequence.  This probability reflects how "likely" that sequence is to occur in the language the model was trained on. Language models are used in many natural language processing (NLP) tasks, such as:

* **Text generation:**  Generating new text that resembles the training data.
* **Speech recognition:**  Predicting the most likely sequence of words given an audio signal.
* **Machine translation:**  Predicting the most likely translation of a sentence in one language to another.
* **Autocomplete/Autosuggest:**  Suggesting the next word or phrase as a user types.

## The Bigram Model

The bigram model is one of the simplest language models. It makes the *Markov assumption* that the probability of a word depends only on the *previous* word.  Mathematically:

P(w<sub>i</sub> | w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>i-1</sub>) ≈ P(w<sub>i</sub> | w<sub>i-1</sub>)

The probability of the entire sequence is then approximated as the product of these bigram probabilities:

P(w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>n</sub>) ≈ P(w<sub>1</sub>) \* P(w<sub>2</sub> | w<sub>1</sub>) \* P(w<sub>3</sub> | w<sub>2</sub>) \* ... \* P(w<sub>n</sub> | w<sub>n-1</sub>)

We estimate these probabilities from a training corpus by counting how often each word follows another:

P(w<sub>i</sub> | w<sub>i-1</sub>) ≈ count(w<sub>i-1</sub>, w<sub>i</sub>) / count(w<sub>i-1</sub>)

## Code Implementation (`src/bigram.py`)

The `bigram.py` file contains the implementation of the bigram model using PyTorch.

### `BigramLanguageModel` Class

This class defines the model itself:

* **`__init__(self, vocab_size)`:**  The constructor takes the vocabulary size as input.  It initializes an embedding table (`token_embedding_table`) of size `vocab_size` x `vocab_size`.  In a bigram model, this table *directly* stores the logits (unnormalized log probabilities) for the next token given the current token.  There are no hidden layers.
* **`forward(self, idx, targets=None)`:**  The forward pass takes a tensor `idx` of shape (Batch, Time) representing the input sequence.  It looks up the logits for the next token for each token in the input sequence using the embedding table. If `targets` are provided (during training), it calculates the cross-entropy loss between the predicted logits and the actual next tokens.
* **`generate(self, idx, max_new_tokens)`:**  This function generates new text from the model.  It starts with an input sequence `idx` and iteratively samples the next token based on the model's predicted probabilities.  It uses `torch.multinomial` to sample from the probability distribution, ensuring that the generated text is diverse.

### `train_bigram(text, epochs=100, learning_rate=1e-2, batch_size = 32, block_size = 8, save_path='model.pth')`

* Takes `text` as input string, and trains the bigram model.
* **Vocabulary Creation:** Creates a vocabulary of unique characters in the input text and mappings between characters and integer indices (`stoi` and `itos`).
* **Encoding and Decoding:** Creates `encode` and `decode` to translate between chars and integer indices.
* **Data Split**: Create Training and Validation sets.
* **Data Loader**: `get_batch` is created to create batches of size `batch_size`.
* **Model Initialization**: Creates `BigramLanguageModel` and `AdamW` optimizer.
* **Training Loop:** Iterates for the specified number of epochs, gets a batch of data, performs the forward and backward passes, and updates the model's parameters.
* **Model Saving:** Saves the trained model's state dictionary, `stoi`, `itos`, and `vocab_size` to a file (`model.pth` by default).  This allows us to load the model later without retraining.
* **Returns:** Returns the trained `model`, the encoding function `encode` and decoding function `decode`.

### `load_bigram_model(model_path)`

This function loads a previously saved bigram model from the specified file path. It restores the model's state dictionary, vocabulary mappings and other parameters and constructs `encode` and `decode` functions.

### `generate_text(model, encode, decode, start_str="", max_new_tokens=100)`

This function takes a trained model, a starting context string, and a maximum number of tokens to generate. It uses the model's `generate` method to produce new text, starting from the provided context (or a newline character if no context is given).

## Jupyter Notebook Example (`notebooks/bigram_example.ipynb`)

The `bigram_example.ipynb` notebook demonstrates how to use the `bigram.py` code:

1. **Loads the data:** Reads the example text from `data/input.txt`.
2. **Trains the Model (or loads it):** It trains the `BigramLanguageModel` using the `train_bigram` and saves it. If a saved model exists, the trained model is loaded.
3. **Generates Text:**  Uses the `generate_text` function to generate text from the model, both with and without a starting context.
4. **Visualizes the Embedding Table (Optional):**  Creates a heatmap of the embedding table.  This visualization shows the logit values for each possible next token given the current token.

## Running the Code

1. **Navigate to the `01_Bigram_Language_Model` directory.**
2. **Run the training script:** `bash scripts/train.sh`.  This will train the model and save it to `model.pth`.
3. **Open and run the Jupyter Notebook:** Launch Jupyter Notebook or Jupyter Lab and open `notebooks/bigram_example.ipynb`. Execute the cells to see the model in action.

## Key Concepts

* **Language Modeling:** Predicting the probability of a sequence of words.
* **Bigram Model:** A simple language model that conditions the probability of a word only on the preceding word.
* **Markov Assumption:** The assumption that the future depends only on the present, not the entire past.
* **Embedding Table:** In this case, a direct lookup table of logits for the next token.
* **Cross-Entropy Loss:**  A loss function used to train language models.
* **Tokenization:**  The process of splitting text into individual units (characters, words, subwords).
* **Vocabulary:**  The set of unique tokens in the training data.
* **torch.multinomial:** For sampling from discrete distributions.

This chapter provides a foundational understanding of language models and sets the stage for more complex models in later chapters.
