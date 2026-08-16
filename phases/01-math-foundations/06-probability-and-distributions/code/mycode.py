import math
import random


def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b


p_king_given_face = conditional_probability(4 / 52, 12 / 52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")


def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)


def categorical_pmf(k, probs):
    return probs[k]


def poisson_pmf(k, lam):
    return (lam**k) * math.exp(-lam) / factorial(k)


def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)


def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))


def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))


die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1 / 6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")


def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]


def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples


def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples


def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]


def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]


def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]


def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages


# import matplotlib.pyplot as plt

# xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
# ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
# plt.plot(xs, ys)
import matplotlib.pyplot as plt
import numpy as np


def sample_exponential(n, lam):
    u = np.random.rand(n)
    return -np.log(u) / lam


samples = sample_exponential(10_000, 2.0)

x = np.linspace(0, samples.max(), 500)
pdf = 2.0 * np.exp(-2.0 * x)

plt.hist(samples, bins=50, density=True, alpha=0.6, label="samples")
plt.plot(x, pdf, label="true PDF")
plt.legend()
plt.show()

# exercise 3
output_logits = [2.0, 0.5, -1.0, 3.0, 0.1]
loss = cross_entropy_loss(output_logits, target_index=3)
print(f"Cross-entropy loss for target index 3: {loss:.4f}")

import torch
import torch.nn as nn

# Batch of 3 samples, 4 classes each
logits = torch.tensor([output_logits])

# Target class indices, not one-hot vectors
targets = torch.tensor([3])

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)

print(loss.item())
