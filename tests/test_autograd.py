"""Assignment 4, Part 3 (Questions 6-9): the autograd behaviour described in the answers."""
import pytest
import torch


def test_q6_gradient_of_quadratic():
    x = torch.tensor([2.0], requires_grad=True)
    y = x**2 + 3 * x
    y.backward()
    assert x.grad.item() == 7.0                      # dy/dx = 2x + 3 = 7


def test_q6_no_grad_raises():
    x = torch.tensor([2.0], requires_grad=False)
    y = x**2 + 3 * x
    with pytest.raises(RuntimeError):
        y.backward()


def test_q6c_default_requires_grad_is_false():
    assert torch.tensor([2.0]).requires_grad is False


def test_q7_weights_without_requires_grad():
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([1.0, 3.0])
    y = w[0] * x**2 + w[1] * x
    y.backward()
    assert x.grad.item() == 7.0 and w.grad is None


def test_q7_weights_with_requires_grad():
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([1.0, 3.0], requires_grad=True)
    y = w[0] * x**2 + w[1] * x
    y.backward()
    assert w.grad.tolist() == [4.0, 2.0]             # [x^2, x] at x = 2
    assert x.grad.item() == 7.0                      # 2*w0*x + w1


def test_q8_detach_breaks_graph_and_fix():
    x = torch.tensor([1.0], requires_grad=True)
    z = (x * 3).detach()
    with pytest.raises(RuntimeError):
        (z * 2).backward()
    x = torch.tensor([1.0], requires_grad=True)
    z = (x * 3).clone()                              # fix: keep z in the graph
    (z * 2).backward()
    assert x.grad.item() == 6.0


def test_q9_gradients_accumulate_and_reset():
    x = torch.tensor([1.0], requires_grad=True)
    (x * 2).backward()
    assert x.grad.item() == 2.0
    (x * 3).backward()
    assert x.grad.item() == 5.0                      # 2 + 3: accumulated
    x.grad.zero_()                                   # fix: reset between backward passes
    (x * 3).backward()
    assert x.grad.item() == 3.0
