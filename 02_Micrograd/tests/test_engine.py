
import pytest
import sys
import os
import math

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from engine import Value

#Test Cases
def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

def test_value_add():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0

def test_value_mul():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0
    c.backward()
    assert a.grad == 3.0
    assert b.grad == 2.0

def test_value_pow():
    a = Value(2.0)
    b = a ** 3
    assert b.data == 8.0
    b.backward()
    assert a.grad == 12.0

def test_value_tanh():
    a = Value(0.5)
    b = a.tanh()
    assert abs(b.data - math.tanh(0.5)) < 1e-6  # Use abs and tolerance for float comparison
    b.backward()
    assert abs(a.grad - (1 - math.tanh(0.5)**2)) < 1e-6

def test_value_exp():
    a = Value(2.0)
    b = a.exp()
    assert abs(b.data - math.exp(2.0)) < 1e-6
    b.backward()
    assert abs(a.grad - math.exp(2.0)) < 1e-6

def test_value_neg():
    a = Value(2.0)
    b = -a
    assert b.data == -2.0
    b.backward()
    assert a.grad == -1.0

def test_value_radd():
    a = Value(2.0)
    b = 3.0 + a  # Using radd
    assert b.data == 5.0
    b.backward()
    assert a.grad == 1.0

def test_value_sub():
    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    assert c.data == 2.0
    c.backward()
    assert a.grad == 1.0
    assert b.grad == -1.0

def test_value_rsub():
    a = Value(2.0)
    b = 3.0 - a  # Using rsub
    assert b.data == 1.0
    b.backward()
    assert a.grad == -1.0

def test_value_rmul():
    a = Value(2.0)
    b = 3.0 * a
    assert b.data == 6.0
    b.backward()
    assert a.grad == 3.0

def test_value_truediv():
    a = Value(6.0)
    b = Value(2.0)
    c = a / b
    assert c.data == 3.0
    c.backward()
    assert a.grad == 0.5
    assert b.grad == -1.5

def test_value_rtruediv():
    a = Value(2.0)
    b = 6.0 / a  # Using rtruediv
    assert b.data == 3.0
    b.backward()
    assert a.grad == -1.5

def test_value_backward():
    # Test a more complex expression with multiple operations
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label='e'
    d = e + c; d.label='d'
    f = Value(-2.0, label='f')
    L = d * f; L.label='L'
    L.backward()

    assert a.grad == 6.0  # dL/da = dL/dd * dd/de * de/da = -2 * 1 * -3 = 6
    assert b.grad == -4.0 # dL/db = dL/dd * dd/de * de/db = -2 * 1 * 2 = -4
    assert c.grad == -2.0 # dL/dc = dL/dd * dd/dc = -2 * 1 = -2
    assert d.grad == -2.0 # dL/dd = f = -2
    assert e.grad == -2.0 # dL/de = dL/dd * dd/de = -2 * 1 = -2
    assert f.grad == 4.0 # dL/df = d = 4