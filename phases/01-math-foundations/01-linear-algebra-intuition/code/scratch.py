import math


class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def angle_between(self, other):
        cos_theta = self.cosine_similarity(other)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.degrees(math.acos(cos_theta))

    def __repr__(self):
        return f"Vector({self.components})"


# a = Vector([1, 2, 3])
# b = Vector([4, 5, 6])

# print(f"a + b = {a + b}")
# print(f"a · b = {a.dot(b)}")
# print(f"|a| = {a.magnitude():.4f}")
# print(f"cosine similarity = {a.cosine_similarity(b):.4f}")


class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector(
                [
                    sum(
                        self.rows[i][j] * other.components[j]
                        for j in range(self.shape[1])
                    )
                    for i in range(self.shape[0])
                ]
            )
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(
                    sum(
                        self.rows[i][k] * other.rows[k][j] for k in range(self.shape[1])
                    )
                )
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix(
            [
                [self.rows[j][i] for j in range(self.shape[0])]
                for i in range(self.shape[1])
            ]
        )

    def __repr__(self):
        return f"Matrix({self.rows})"


# rotation_90 = Matrix([[0, -1], [1, 0]])
# point = Vector([3, 1])

# rotated = rotation_90 @ point
# print(f"Original: {point}")
# print(f"Rotated 90°: {rotated}")

# import random

# random.seed(42)
# weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
# input_vector = Vector([1.0, 0.5, -0.3])

# output = weights @ input_vector
# print(f"Input (3D): {input_vector}")
# print(f"Output (2D): {output}")
# print("This is what a neural network layer does -- matrix multiplication.")


def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


# v1 = Vector([1, 0, 0])
# v2 = Vector([1, 1, 0])
# v3 = Vector([1, 1, 1])
# basis = gram_schmidt([v1, v2, v3])
# for i, u in enumerate(basis):
#     print(f"u{i+1} = {u}")
#     print(f"  |u{i+1}| = {u.magnitude():.6f}")

# print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
# print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
# print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")

# import numpy as np

# a = np.array([1, 2, 3], dtype=float)
# b = np.array([4, 5, 6], dtype=float)

# print(f"a + b = {a + b}")
# print(f"a · b = {np.dot(a, b)}")
# print(f"|a| = {np.linalg.norm(a):.4f}")
# print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

# W = np.random.randn(2, 3) * 0.1
# x = np.array([1.0, 0.5, -0.3])
# print(f"Wx = {W @ x}")

vec1 = Matrix([[2, 0], [0, 3]])
vec2 = Vector([1, 1])
res = vec1 @ vec2
print(f"vec1 @ vec2 = {res}")
