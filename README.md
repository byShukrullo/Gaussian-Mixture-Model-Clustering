# Gaussian-Mixture-Model-Clustering
Distribution-Based Clustering Description: Models clusters based on probability distributions. Examples: Gaussian Mixture Models (GMMs). Advantages: Flexible, suitable for probabilistic data. Disadvantages: Sensitive to initialization and assumes the data follows a known distribution.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
def generate_data(n_samples=300, n_features=2, n_clusters=4, random_state=42):
    data, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    return data
    def gmm_clustering(data, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(data)
    labels = gmm.predict(data)
    return gmm.means_, labels
    def visualize(data, labels, centroids):
    plt.figure(figsize=(8, 6))
    for i, centroid in enumerate(centroids):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i+1}')
        plt.scatter(centroid[0], centroid[1], s=200, c='black', marker='x')

    plt.title('Gaussian Mixture Model Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    if __name__ == "__main__":
    data = generate_data(n_samples=300, n_clusters=4)

    n_clusters = 4
    centroids, labels = gmm_clustering(data, n_clusters)

    visualize(data, labels, centroids)
