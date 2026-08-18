import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, : self.n_components].T
        self.eigenvalues = eigenvalues[: self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        return Z @ self.components + self.mean


np.random.seed(42)
n_samples = 500

t = np.random.uniform(0, 2 * np.pi, n_samples)
x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)

X_synthetic = np.column_stack([x1, x2, x3])

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_synthetic)

print(f"Original shape: {X_synthetic.shape}")
print(f"Reduced shape:  {X_reduced.shape}")
print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
print(f"Total variance captured: {sum(pca.explained_variance_ratio_):.4f}")

from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_mnist = mnist.data[:5000].astype(float)
y_mnist = mnist.target[:5000].astype(int)

pca_mnist = PCA(n_components=50)
X_pca50 = pca_mnist.fit_transform(X_mnist)
print(
    f"50 components capture {sum(pca_mnist.explained_variance_ratio_):.2%} of variance"
)

pca_2d = PCA(n_components=2)
X_pca2d = pca_2d.fit_transform(X_mnist)
print(f"2 components capture {sum(pca_2d.explained_variance_ratio_):.2%} of variance")

from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE

sklearn_pca = SklearnPCA(n_components=2)
X_sklearn_pca = sklearn_pca.fit_transform(X_mnist)

print(f"\nOur PCA explained variance:     {pca_2d.explained_variance_ratio_}")
print(f"Sklearn PCA explained variance: {sklearn_pca.explained_variance_ratio_}")

diff = np.abs(np.abs(X_pca2d) - np.abs(X_sklearn_pca))
print(f"Max absolute difference: {diff.max():.10f}")
diff_tole = np.allclose(np.abs(X_pca2d), np.abs(X_sklearn_pca), atol=1e-6)
print(f"weather the two matrix are close enough: {diff_tole}")

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_mnist)
print(f"\nt-SNE output shape: {X_tsne.shape}")

try:
    from umap import UMAP

    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_mnist)
    print(f"UMAP output shape: {X_umap.shape}")
except ImportError:
    print("Install umap-learn: pip install umap-learn")


from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_mnist, y_mnist, test_size=0.2, random_state=42
)

results = {}
for k in [10, 30, 50, 100, 200]:
    pca_k = SklearnPCA(n_components=k)
    X_tr = pca_k.fit_transform(X_train)
    X_te = pca_k.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te))
    var_captured = sum(pca_k.explained_variance_ratio_)
    results[k] = (acc, var_captured)
    print(f"k={k:>3d}  accuracy={acc:.4f}  variance={var_captured:.4f}")

# ex1
components = [10, 50, 200]
for n_comp in components:
    pca_mnist = PCA(n_components=n_comp)
    cur_pca = pca_mnist.fit_transform(X_mnist)
    cur_pac_inversed = pca_mnist.inverse_transform(cur_pca)
    print(
        f"for top {n_comp} components, the mse is {np.mean((X_mnist - cur_pac_inversed) ** 2)}"
    )
    print(f"Total variance captured: {sum(pca_mnist.explained_variance_ratio_):.4f}")
# more components, less MSE

# ex2
import matplotlib.pyplot as plt

perplexitys = [5, 30, 100]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, perplexity in zip(axes, perplexitys):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_mnist)
    print(f"\n perplexity: {perplexity} t-SNE output shape: {X_tsne.shape}")
    scatter = ax.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=y_mnist,
        cmap="tab10",
        s=8,
        alpha=0.7,
    )
    ax.set_title(f"t-SNE perplexity = {perplexity}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
# higher the perplexity, more tightness for each cluster


# ex3
# using custom PCA
from sklearn.datasets import make_classification

X_sample, y_sample = make_classification(
    n_samples=5000,
    n_features=50,
    n_informative=5,
    n_redundant=0,
    n_repeated=0,
    n_clusters_per_class=1,
    class_sep=2.0,
    flip_y=0.0,
    shuffle=False,
    random_state=42,
)
components = [50]
for n_comp in components:
    pca_sample = PCA(n_components=n_comp)
    cur_pca = pca_sample.fit_transform(X_sample)
    cur_pac_inversed = pca_sample.inverse_transform(cur_pca)
    print(
        f"for top {n_comp} components, the mse is {np.mean((X_sample - cur_pac_inversed) ** 2)}"
    )
    print(f"Total variance captured: {sum(pca_sample.explained_variance_ratio_):.4f}")
    cum_var = np.cumsum(pca_sample.explained_variance_ratio_)
    print(f"list of the cum ratio: {cum_var}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
    plt.axhline(0.95, color="red", linestyle="--", label="95% variance")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Cumulative Explained Variance")
    plt.grid(True)
    plt.legend()
    plt.show()

# using sklearn PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

X_sample, y_sample = make_classification(
    n_samples=5000,
    n_features=50,
    n_informative=5,
    n_redundant=0,
    n_repeated=0,
    n_clusters_per_class=1,
    class_sep=10.0,
    flip_y=0.0,
    shuffle=False,
    random_state=42,
)

pca_sample = PCA(n_components=50)
pca_sample.fit(X_sample)

cum_ratio = np.cumsum(pca_sample.explained_variance_ratio_)
print(f"list of the cum ratio: {cum_ratio}")
plt.figure(figsize=(8, 5))
plt.plot(range(1, 51), cum_ratio, marker="o")
plt.axhline(0.95, color="red", linestyle="--", label="95% variance")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Elbow Plot")
plt.grid(True)
plt.legend()
plt.show()

# The elbow is not sharp, even with sklearn PCA,
# because n_informative=5 only means 5 features are useful for the labels,
# not that the dataset has only 5 high-variance directions.
# PCA measures variance, so the cumulative explained variance increases gradually instead of flattening after 5 components.
