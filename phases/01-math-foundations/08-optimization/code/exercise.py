def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def rosenbrock_gradient(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x**2) * (-2 * x)
    df_dy = 200 * (y - x**2)
    return [df_dx, df_dy]


class GradientDescent:
    def __init__(self, lr=0.001, withDecay=False, decay_rate=0.999):
        self.lr_0 = lr
        self.withDecay = withDecay
        self.decay_rate = decay_rate
        self.t = 0

    def step(self, params, grads):
        # Use the base lr on the first step, then decay after each update.
        # This matches lr = lr_0 * decay_rate^step with step starting at 0.
        if self.withDecay:
            lr = self.lr_0 * (self.decay_rate**self.t)
        else:
            lr = self.lr_0

        self.t += 1
        return [p - lr * g for p, g in zip(params, grads)]


class SGDMomentum:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        self.velocity = [self.momentum * v + g for v, g in zip(self.velocity, grads)]
        return [p - self.lr * v for p, v in zip(params, self.velocity)]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        self.m = [self.beta1 * m + (1 - self.beta1) * g for m, g in zip(self.m, grads)]
        self.v = [
            self.beta2 * v + (1 - self.beta2) * g**2 for v, g in zip(self.v, grads)
        ]

        m_hat = [m / (1 - self.beta1**self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2**self.t) for v in self.v]

        return [
            p - self.lr * mh / (vh**0.5 + self.epsilon)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]


def optimize(optimizer, func, grad_func, start, steps=5000):
    params = list(start)
    history = [params[:]]
    for _ in range(steps):
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history


start = [-1.0, 1.0]
# ex1
# lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01]
# for lr in lrs:
#     optimizer = GradientDescent(lr=lr)
#     history = optimize(optimizer, rosenbrock, rosenbrock_gradient, start)
#     final = history[-1]
#     loss = rosenbrock(final)
#     print(f"GD lr={lr:.4f} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")
# result: works until 0.001. [0.005, 0.01] diverge

# ex2
momentums = [0.0, 0.5, 0.9, 0.99]
for momentum in momentums:
    optimizer = SGDMomentum(lr=0.0001, momentum=momentum)
    history = optimize(optimizer, rosenbrock, rosenbrock_gradient, start, steps=10000)
    final = history[-1]
    loss = rosenbrock(final)
    print(
        f"SGD+M momentum={momentum:.2f} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}"
    )
# result 0.99 converges fastest, but for overshoot, i have not find it.


# ex3
def customBenchFunc(params):
    x, y = params
    return x**2 - y**2


def customBenchFuncGradient(params):
    x, y = params
    df_dx = 2 * x
    df_dy = -2 * y
    return [df_dx, df_dy]


start = [0.01, 0.01]
step = 5000
gd_history = optimize(
    GradientDescent(lr=0.0005),
    customBenchFunc,
    customBenchFuncGradient,
    start,
    steps=step,
)
sgd_history = optimize(
    SGDMomentum(lr=0.0001, momentum=0.9),
    customBenchFunc,
    customBenchFuncGradient,
    start,
    steps=step,
)
adam_history = optimize(
    Adam(lr=0.01), customBenchFunc, customBenchFuncGradient, start, steps=step
)

for name, history in [
    ("GD", gd_history),
    ("SGD+M", sgd_history),
    ("Adam", adam_history),
]:
    final = history[-1]
    loss = customBenchFunc(final)
    print(f"{name:6s} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")

# result: speed from slow to fast: GD < Adam < SGD+M.

# ex4
start = [-1.0, 1.0]

optimizer_with_decay = GradientDescent(lr=0.0005, withDecay=True)
history = optimize(optimizer_with_decay, rosenbrock, rosenbrock_gradient, start)
final = history[-1]
loss = rosenbrock(final)
print(f"GD with decay -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")

optimizer_without_decay = GradientDescent(lr=0.0005, withDecay=False)
history = optimize(optimizer_without_decay, rosenbrock, rosenbrock_gradient, start)
final = history[-1]
loss = rosenbrock(final)
print(f"GD without decay -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")

# result: without decay converges faster than with decay.
