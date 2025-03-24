# Chapter 2: Micrograd - A Tiny Autograd Engine

This chapter introduces the core concepts of automatic differentiation by building a minimalist autograd engine called "Micrograd." This engine, inspired by Andrej Karpathy's micrograd, allows us to perform backpropagation on scalar values and build a small neural network library.

## Core Concepts

* **Automatic Differentiation (Autograd):**  A technique for automatically computing derivatives of functions defined by computer programs.  It's the foundation of modern deep learning frameworks.
* **Computational Graph:**  A directed graph where nodes represent variables (tensors) and edges represent operations.  Autograd builds this graph implicitly as you perform calculations.
* **Backpropagation:**  The algorithm for computing gradients by traversing the computational graph in reverse, applying the chain rule of calculus at each node.
* **Scalar Values:**  Our Micrograd engine operates on scalar values (single numbers) rather than tensors (multi-dimensional arrays). This simplification makes the code easier to understand.

## `engine.py`

This file contains the `Value` class, which is the heart of our autograd engine.

* **`Value` Class:**
  * `data`: Stores the scalar value.
  * `grad`: Stores the gradient of the output with respect to this value (initialized to 0).
  * `_backward`:  A function that performs the backward pass (local gradient computation) for this value. This function implements the chain rule.
  * `_prev`: A set of `Value` objects that were used as inputs to create this `Value`.  This defines the connections in the computational graph.
  * `_op`:  A string representing the operation that created this `Value` (for debugging and visualization).
  * `label`: For naming the Values.

  * **Methods:**
    * `__add__`, `__mul__`, `__pow__`, `tanh`, `exp`:  These methods implement basic mathematical operations and their corresponding backward passes.  Each operation creates a new `Value` object and defines its `_backward` function.
    * `backward()`:  This method performs the full backpropagation algorithm.  It uses a topological sort to ensure that gradients are computed in the correct order.
    * Operator overloads are implemented.

## `nn.py`

This file builds a simple neural network library on top of the `engine.py`.

* **`Module` Class**: Base class, has `zero_grad` and `parameters` methods.
* **`Neuron` Class:**
  * Represents a single neuron with weights (`w`), a bias (`b`), and an optional nonlinearity (`tanh` by default).

* **`Layer` Class:**
  * Represents a layer of neurons.

* **`MLP` Class:**
  * Represents a multi-layer perceptron (a sequence of layers).

* **Methods (`Neuron`, `Layer`, `MLP`):**
  * `__call__(self, x)`:  Performs the forward pass.
  * `parameters()`:  Returns a list of all the parameters (`Value` objects) of the neuron/layer/MLP.
  * `zero_grad()`: Set all the gradients to zero.

## `notebooks/micrograd_example.ipynb`

The Jupyter Notebook demonstrates the usage of `engine.py` and `nn.py`:

1. **Simple Expression:** Shows how to create `Value` objects, perform basic operations, and compute gradients using `backward()`.
2. **Manual Neuron:**  Demonstrates creating a neuron manually using `Value` objects and performing a forward and backward pass.
3. **Neuron with Exp and division:** To test the implementation.
4. **Using the `nn` module:**  Shows how to create and train a simple `MLP` using the `nn.py` module.
5. **Cross Entropy:** An implementation of cross entropy is given.

**Visualization (Optional):** The notebook includes a `draw_dot` function that uses the `graphviz` library to visualize the computational graph. This is *extremely* helpful for understanding how backpropagation works.  You'll need to install Graphviz separately (see instructions below).

## Running the Code

1. **Navigate to the `02_Micrograd` directory.**
2. **Run the training script:** `bash scripts/train.sh`.  This will train the model.
3. **Open and run the Jupyter Notebook:** Launch Jupyter Notebook or Jupyter Lab and open `notebooks/micrograd_example.ipynb`. Execute the cells to see Micrograd in action.

## Graphviz Installation (Optional, but Highly Recommended)

The `draw_dot` function in the notebook uses the `graphviz` library to create visualizations of the computational graphs.  You need to install both the Python library and the Graphviz software itself.

**1. Install the Python library:**

```bash
pip install graphviz
```

**2. Install Graphviz:**

* **Linux (Ubuntu):** `sudo apt-get install graphviz`
* **Mac (Homebrew):** `brew install graphviz`
* **Windows:** Download and install from [Graphviz website](https://graphviz.gitlab.io/download/)

**3. Restart Jupyter:** If you installed Graphviz after starting Jupyter, you may need to restart the Jupyter server.

After installing Graphviz, you should be able to run the notebook cells that use draw_dot and see the visualizations.

## Key Takeaways

* Understanding how automatic differentiation works is fundamental to understanding deep learning.
* Micrograd provides a simplified but powerful way to learn these concepts.
* The chain rule is the core of backpropagation.
* Building the computational graph dynamically allows for flexible and efficient gradient computation.
* The `nn.py` module demonstrates how to build higher-level abstractions on top of the basic autograd engine.

This chapter lays the groundwork for understanding more complex deep learning frameworks like PyTorch. The next chapters will build upon these concepts.
