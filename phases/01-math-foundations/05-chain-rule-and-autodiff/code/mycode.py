class Value:
    def __init__(self, data, children=(), op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data**n, (self,), f"**{n}")

        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return (
            self * (other**-1)
            if isinstance(other, Value)
            else self * (Value(other) ** -1)
        )

    def exp(self):
        import math

        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        import math

        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        import math

        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out


import random


class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# random.seed(42)
# model = MLP([2, 4, 1])  # 2 inputs, 4 hidden neurons, 1 output

# xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
# ys = [-1, 1, 1, -1]  # XOR pattern (using -1/1 for tanh)

# for step in range(100):
#     preds = [model(x) for x in xs]
#     loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

#     for p in model.parameters():
#         p.grad = 0.0
#     loss.backward()

#     lr = 0.05
#     for p in model.parameters():
#         p.data -= lr * p.grad

#     if step % 20 == 0:
#         print(f"step {step:3d}  loss = {loss.data:.4f}")

# print("\nPredictions after training:")
# for x, y in zip(xs, ys):
#     print(f"  input={x}  target={y:2d}  pred={model(x).data:6.3f}")


def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff


def expr(x):
    return (x**3 + x * 2 + 1).tanh()


# ad, num, diff = gradient_check(expr, 0.5)
# print(f"Autodiff:  {ad:.8f}")
# print(f"Numerical: {num:.8f}")
# print(f"Difference: {diff:.2e}")
# # Difference should be < 1e-5

# x1 = Value(2.0)
# x2 = Value(3.0)
# a = x1 * x2  # a = 6.0
# b = a + Value(1.0)  # b = 7.0
# y = b.relu()  # y = 7.0

# y.backward()

# print(f"y = {y.data}")  # 7.0
# print(f"dy/dx1 = {x1.grad}")  # 3.0 (= x2)
# print(f"dy/dx2 = {x2.grad}")  # 2.0 (= x1)


# x1 = Value(2.0)
# n = 3
# b = x1**n
# y = b.relu()

# y.backward()

# print(f"y = {y.data}")  # 8.0
# print(f"dy/dx1 = {x1.grad}")  # 12.0 (= n * x1^(n-1))

# x1 = Value(0.0)
# x2 = Value(2.0)
# tanh_x1 = x1.tanh()
# tanh_x2 = x2.tanh()

# tanh_x1.backward()
# tanh_x2.backward()

# print(f"tanh_x1 = {tanh_x1.data}, dtanh_x1/dx1 = {x1.grad}")
# print(f"tanh_x2 = {tanh_x2.data}, dtanh_x2/dx2 = {x2.grad}")

# x1 = Value(2.0)
# x2 = Value(3.0)
# w1 = Value(0.3)
# w2 = Value(0.7)
# b = Value(1.0)
# f = (w1 * x1 + w2 * x2 + b).relu()
# f.backward()

# print(f"f = {f.data}")
# print(f"df/dx1 = {x1.grad}")
# print(f"df/dx2 = {x2.grad}")
# print(f"df/dw1 = {w1.grad}")
# print(f"df/dw2 = {w2.grad}")
# print(f"df/db = {b.grad}")


# import torch

# x1 = torch.tensor(2.0, requires_grad=True)
# x2 = torch.tensor(3.0, requires_grad=True)
# w1 = torch.tensor(0.3, requires_grad=True)
# w2 = torch.tensor(0.7, requires_grad=True)
# b = torch.tensor(1.0, requires_grad=True)
# f = torch.relu(w1 * x1 + w2 * x2 + b)
# f.backward()

# print(f"f = {f.item()}")
# print(f"df/dx1 = {x1.grad.item()}")
# print(f"df/dx2 = {x2.grad.item()}")
# print(f"df/dw1 = {w1.grad.item()}")
# print(f"df/dw2 = {w2.grad.item()}")
# print(f"df/db = {b.grad.item()}")


class Dual:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0.0)
        return Dual(self.real + other.real, self.dual + other.dual)

    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0.0)
        return Dual(
            self.real * other.real, self.real * other.dual + self.dual * other.real
        )


x = Dual(2.0, 1.0)
three = Dual(3.0, 0.0)
two = Dual(2.0, 0.0)
y = x * x + x * three + two
print(y.real)  # 12.0
print(y.dual)  # 7.0

x = Value(2.0)
y = x * x + x * Value(3.0) + Value(2.0)
print(y.data)  # 12.0
y.backward()
print(x.grad)  # 7.0
