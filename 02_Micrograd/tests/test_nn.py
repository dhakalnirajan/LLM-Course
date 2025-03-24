import pytest
import sys
import os
import torch

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from nn import Neuron, Layer, MLP
from engine import Value


def test_neuron_creation():
    n = Neuron(3)
    assert len(n.w) == 3
    assert isinstance(n.b, Value)

def test_neuron_call():
    n = Neuron(3)
    x = [Value(1.0), Value(2.0), Value(3.0)]
    output = n(x)
    assert isinstance(output, Value)

def test_neuron_parameters():
    n = Neuron(3)
    params = n.parameters()
    assert len(params) == 4  # 3 weights + 1 bias
    assert all(isinstance(p, Value) for p in params)

def test_layer_creation():
    layer = Layer(3, 4)  # 3 inputs, 4 neurons
    assert len(layer.neurons) == 4
    for neuron in layer.neurons:
        assert isinstance(neuron, Neuron)

def test_layer_call():
    layer = Layer(3, 2)
    x = [Value(1.0), Value(2.0), Value(3.0)]
    output = layer(x)
    assert isinstance(output, list)
    assert len(output) == 2
    assert all(isinstance(o, Value) for o in output)
    
    
def test_layer_parameters():
    layer = Layer(3, 2)
    params = layer.parameters()
    assert len(params) == 10  # 3*2 weights + 2 biases
    assert all(isinstance(p, Value) for p in params)
    
def test_mlp_creation():
    mlp = MLP(2, [4, 3, 2])
    assert len(mlp.layers) == 3
    assert all(isinstance(layer, Layer) for layer in mlp.layers)
    
def test_mlp_call():
    mlp = MLP(2, [4, 3, 2])
    x = [Value(1.0), Value(2.0)]
    output = mlp(x)
    assert isinstance(output, list)
    assert len(output) == 2
    assert all(isinstance(o, Value) for o in output)
    
def test_mlp_parameters():
    mlp = MLP(2, [4, 3, 2])
    params = mlp.parameters()
    assert len(params) == 26  # 2*4 + 4*3 + 3*2 + 4 + 3 + 2
    assert all(isinstance(p, Value) for p in params)
    
def test_mlp_repr():
    mlp = MLP(2, [4, 3, 2])
    assert str(mlp) == "MLP of [Layer of [LinearNeuron(2), LinearNeuron(2), LinearNeuron(2), LinearNeuron(2)], Layer of [LinearNeuron(4), LinearNeuron(4), LinearNeuron(4)], Layer of [LinearNeuron(3), LinearNeuron(3)]]"

def test_layer_repr():
    layer = Layer(3, 2)
    assert str(layer) == "Layer of [LinearNeuron(3), LinearNeuron(3)]"
    
def test_neuron_repr():
    n = Neuron(3)
    assert str(n) == "LinearNeuron(3)"
    
def test_neuron_nonlin():
    n = Neuron(3, nonlin=True)
    x = [Value(1.0), Value(2.0), Value(3.0)]
    output = n(x)
    assert isinstance(output, Value)
    assert output.data <= 1.0
    assert output.data >= -1.0
    
