import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.mixture import GaussianMixture

# Load embeddings
embeddings = np.load("models/embeddings.npy")

# Train clustering model
gmm = GaussianMixture(n_components=10, random_state=42)
labels = gmm.fit_predict(embeddings)

# Reduce dimensions using UMAP
reducer = umap.UMAP(random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

# Plot clusters
plt.figure(figsize=(8,6))

plt.scatter(
    embedding_2d[:,0],
    embedding_2d[:,1],
    c=labels,
    cmap="tab20",
    s=5
)

plt.title("Semantic Clusters of 20 Newsgroups Dataset")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

plt.show()