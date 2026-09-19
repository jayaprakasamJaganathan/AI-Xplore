import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

class KMeansFromScratch:
    """
    K-Means clustering algorithm implemented from scratch.
    
    Parameters:
    -----------
    n_clusters : int
        Number of clusters to form
    max_iter : int
        Maximum number of iterations
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        
    def fit(self, X):
        """
        Fit K-Means clustering to the data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features)
        """
        np.random.seed(self.random_state)
        
        # Initialize centroids randomly from data points
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        # Iterate until convergence or max iterations
        for iteration in range(self.max_iter):
            # Step 1: Assign each point to nearest centroid
            distances = self._compute_distances(X)
            new_labels = np.argmin(distances, axis=1)
            
            # Step 2: Update centroids
            new_centroids = self._update_centroids(X, new_labels)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
            self.labels = new_labels
            
        # Calculate inertia (within-cluster sum of squared distances)
        self.inertia = self._compute_inertia(X)
        
    def _compute_distances(self, X):
        """Compute distances from each point to each centroid."""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)
            
        return distances
    
    def _update_centroids(self, X, labels):
        """Update centroids based on current cluster assignments."""
        new_centroids = np.zeros_like(self.centroids)
        
        for cluster in range(self.n_clusters):
            cluster_points = X[labels == cluster]
            if len(cluster_points) > 0:
                new_centroids[cluster] = np.mean(cluster_points, axis=0)
            else:
                # Reinitialize empty cluster
                random_idx = np.random.randint(0, X.shape[0])
                new_centroids[cluster] = X[random_idx]
                
        return new_centroids
    
    def _compute_inertia(self, X):
        """Calculate within-cluster sum of squared distances."""
        inertia = 0
        for cluster in range(self.n_clusters):
            cluster_points = X[self.labels == cluster]
            if len(cluster_points) > 0:
                distances = np.linalg.norm(cluster_points - self.centroids[cluster], axis=1)
                inertia += np.sum(distances ** 2)
        return inertia
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns:
        --------
        labels : numpy.ndarray
            Predicted cluster labels
        """
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        """Fit the model and return cluster labels."""
        self.fit(X)
        return self.labels

def generate_sample_data():
    """Generate sample data for clustering demonstration."""
    # Create 3 clusters with different centers
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, 
                     cluster_std=1.5, random_state=42)
    
    # Add some noise
    X += np.random.normal(0, 0.2, X.shape)
    
    return X, y

def visualize_clusters(X, labels, centroids, title):
    """Visualize clustering results."""
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                         alpha=0.6, s=50)
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', 
               marker='x', s=200, linewidths=3, label='Centroids')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    
    return plt

def elbow_method(X, max_k=10):
    """Apply elbow method to determine optimal number of clusters."""
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeansFromScratch(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.title('Elbow Method for Optimal K', fontsize=14)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squared Distances)')
    plt.grid(True, alpha=0.3)
    
    return plt, k_range, inertias

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    # Generate sample data
    X, true_labels = generate_sample_data()
    
    # Apply K-Means with different numbers of clusters
    for k in [2, 3, 4]:
        kmeans = KMeansFromScratch(n_clusters=k, random_state=42)
        predicted_labels = kmeans.fit_predict(X)
        
        # Visualize results
        plt = visualize_clusters(X, predicted_labels, kmeans.centroids, 
                               f'K-Means Clustering (K={k})')
        plt.savefig(f'output/kmeans_from_scratch_k{k}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Apply elbow method
    plt, k_range, inertias = elbow_method(X)
    plt.savefig('output/elbow_method_from_scratch.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("K-Means from scratch implementation completed successfully!")
    print(f"Final inertia for K=3: {inertias[2]:.2f}")
    print("Visualizations saved to output/")