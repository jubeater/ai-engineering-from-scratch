import math


def information_content(p, base=2):
    if p <= 0 or p > 1:
        return float("inf") if p <= 0 else 0.0
    return -math.log(p) / math.log(base)


def entropy(probs, base=2):
    return sum(p * information_content(p, base) for p in probs if p > 0)


fair_coin = [0.5, 0.5]
biased_coin = [0.99, 0.01]
fair_die = [1 / 6] * 6

print(f"Fair coin entropy:   {entropy(fair_coin):.4f} bits")
print(f"Biased coin entropy: {entropy(biased_coin):.4f} bits")
print(f"Fair die entropy:    {entropy(fair_die):.4f} bits")


def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float("inf")
            total += pi * (-math.log(qi) / math.log(base))
    return total


def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)


true_dist = [0.7, 0.2, 0.1]
good_model = [0.6, 0.25, 0.15]
bad_model = [0.1, 0.1, 0.8]

print(f"Entropy of true dist:     {entropy(true_dist):.4f} bits")
print(f"CE (good model):          {cross_entropy(true_dist, good_model):.4f} bits")
print(f"CE (bad model):           {cross_entropy(true_dist, bad_model):.4f} bits")
print(f"KL divergence (good):     {kl_divergence(true_dist, good_model):.4f} bits")
print(f"KL divergence (bad):      {kl_divergence(true_dist, bad_model):.4f} bits")


def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])


logits = [2.0, 1.0, 0.1]
true_class = 0

probs = softmax(logits)
loss = cross_entropy_loss(true_class, logits)

print(f"Logits:      {logits}")
print(f"Softmax:     {[f'{p:.4f}' for p in probs]}")
print(f"True class:  {true_class}")
print(f"Loss:        {loss:.4f} nats")
print(f"Perplexity:  {math.exp(loss):.2f}")

import random

random.seed(42)

n_samples = 1000
n_classes = 3
true_labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
model_logits = [
    [random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)
]

ce_loss = (
    sum(
        cross_entropy_loss(label, logits)
        for label, logits in zip(true_labels, model_logits)
    )
    / n_samples
)

nll = (
    -sum(
        math.log(softmax(logits)[label])
        for label, logits in zip(true_labels, model_logits)
    )
    / n_samples
)

print(f"Cross-entropy loss:      {ce_loss:.6f}")
print(f"Negative log-likelihood: {nll:.6f}")
print(f"Difference:              {abs(ce_loss - nll):.2e}")


def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi


independent = [[0.25, 0.25], [0.25, 0.25]]
dependent = [[0.45, 0.05], [0.05, 0.45]]

print(f"MI (independent): {mutual_information(independent):.4f} bits")
print(f"MI (dependent):   {mutual_information(dependent):.4f} bits")

import numpy as np


def np_entropy(p):
    p = np.asarray(p, dtype=float)
    mask = p > 0
    result = np.zeros_like(p)
    result[mask] = p[mask] * np.log(p[mask])
    return -result.sum()


def np_cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return -(p[mask] * np.log(q[mask])).sum()


def np_kl_divergence(p, q):
    return np_cross_entropy(p, q) - np_entropy(p)


true = np.array([0.7, 0.2, 0.1])
pred = np.array([0.6, 0.25, 0.15])
print(f"Entropy:    {np_entropy(true):.4f} nats")
print(f"Cross-ent:  {np_cross_entropy(true, pred):.4f} nats")
print(f"KL div:     {np_kl_divergence(true, pred):.4f} nats")

# ex1
pred = [1 / 26 for x in range(26)]
print(f"Entropy:    {entropy(pred):.4f} bits")
# result: uniform have higher entropy

# ex2
pred = [5.0, 2.0, 0.5]
true_class = 1

loss = cross_entropy_loss(true_class, pred)
print(f"Loss:        {loss:.4f} bits")
pred = [0, 1, 0]
loss = cross_entropy_loss(true_class, pred)
print(f"Loss:        {loss:.4f} bits")
# result: cannot get zero loss with finite number of logits

# ex3
p = [0.7, 0.2, 0.1]
q = [0.6, 0.25, 0.15]
print(f"KL divergence (p || q):     {kl_divergence(p, q):.4f} bits")
print(f"KL divergence (q || p):     {kl_divergence(q, p):.4f} bits")

# ex4
sequence_pair = [(1, [0.7, 0.2, 0.1]), (2, [0.6, 0.25, 0.15]), (0, [0.8, 0.1, 0.1])]
ce_index = []
for pair in sequence_pair:
    true_token_index = pair[0]
    predicted_logits = pair[1]
    cur_probs = softmax(predicted_logits)
    cur_ce = -math.log(cur_probs[true_token_index])
    ce_index.append(cur_ce)
total_ce = sum(ce_index)

print(f"plex sequence is : {math.exp(total_ce/len(ce_index))}")
