import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.decomposition import PCA

from kmeans_from_scratch import KMeansFromScratch

def generate_sample_data():
    """Generate sample data for clustering demonstration."""
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, 
                     cluster_std=1.5, random_state=42)
    X += np.random.normal(0, 0.2, X.shape)
    return X, y

def compare_kmeans_vs_pca():
    """Compare K-Means clustering with PCA dimensionality reduction."""
    # Generate data
    X, y = generate_sample_data()
    
    # Apply K-Means
    kmeans_scratch = KMeansFromScratch(n_clusters=3, random_state=42)
    kmeans_scratch.fit(X)
    
    # Apply sklearn K-Means for comparison
    kmeans_sklearn = SklearnKMeans(n_clusters=3, random_state=42)
    kmeans_sklearn.fit(X)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original data
    axes[0].scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=50)
    axes[0].set_title('Original Data', fontsize=14)
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: K-Means from scratch
    scatter = axes[1].scatter(X[:, 0], X[:, 1], c=kmeans_scratch.labels, 
                             cmap='viridis', alpha=0.6, s=50)
    axes[1].scatter(kmeans_scratch.centroids[:, 0], kmeans_scratch.centroids[:, 1],
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[1].set_title('K-Means from Scratch', fontsize=14)
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: PCA visualization
    scatter = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                             cmap='viridis', alpha=0.6, s=50)
    axes[2].set_title('PCA Visualization', fontsize=14)
    axes[2].set_xlabel('Principal Component 1')
    axes[2].set_ylabel('Principal Component 2')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

def demo_high_dimensional():
    """Demonstrate K-Means on high-dimensional data."""
    # Generate high-dimensional data
    X, y = make_blobs(n_samples=200, centers=3, n_features=10, 
                     cluster_std=2.0, random_state=42)
    
    # Apply K-Means
    kmeans = KMeansFromScratch(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels, 
                         cmap='viridis', alpha=0.6, s=50)
    plt.title('K-Means on High-Dimensional Data (PCA Projection)', fontsize=14)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    
    return plt

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    # Compare K-Means with PCA
    plt = compare_kmeans_vs_pca()
    plt.savefig('output/kmeans_vs_pca_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # High-dimensional demonstration
    plt = demo_high_dimensional()
    plt.savefig('output/high_dimensional_kmeans.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("K-Means vs PCA comparison completed successfully!")
    print("Visualizations saved to output/")