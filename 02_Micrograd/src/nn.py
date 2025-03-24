import random
from micrograd.engine import Value  # Use the engine we just created


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    """A single neuron with weights, bias, and a nonlinearity."""

    def __init__(self, nin, nonlin=True):
        """
        Args:
            nin: Number of inputs.
            nonlin: Whether to apply a nonlinearity (tanh).
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """A layer of neurons."""

    def __init__(self, nin, nout, **kwargs):
        """
        Args:
            nin: Number of inputs.
            nout: Number of outputs (neurons in the layer).
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """A Multi-Layer Perceptron."""

    def __init__(self, nin, nouts):
        """
        Args:
            nin: Number of inputs.
            nouts: A list of integers, where each integer represents the
                   number of neurons in a layer.
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"