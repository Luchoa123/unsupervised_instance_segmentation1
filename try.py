import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate data
cluster_1 = np.random.normal(loc=0.0, scale=1.0, size=(1000, 100))
cluster_2 = np.random.normal(loc=0.9, scale=1.0, size=(50, 100))

# Combine data
data = np.vstack((cluster_1, cluster_2))

# Labels for visualization
labels = np.array([0]*1000 + [1]*50)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data)

# Plot the result
plt.figure(figsize=(8, 6))
plt.scatter(data_tsne[labels == 0, 0], data_tsne[labels == 0, 1], label='Cluster 1', alpha=0.5)
plt.scatter(data_tsne[labels == 1, 0], data_tsne[labels == 1, 1], label='Cluster 2', alpha=0.5, color='red')
plt.legend()
plt.title("t-SNE visualization of clusters")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()
