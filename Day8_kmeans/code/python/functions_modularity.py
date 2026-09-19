"""
Functions & Modularity - Practical Solutions (A6)
=================================================
Solutions for all practical exercises from A6 Python Programming Task
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# 1. K-Means Clustering from Scratch
# ============================================================
class KMeansFromScratch:
    """K-Means clustering algorithm implemented from scratch."""
    
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        
    def fit(self, X):
        """Fit K-Means clustering to the data."""
        np.random.seed(self.random_state)
        
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for iteration in range(self.max_iter):
            distances = self._compute_distances(X)
            new_labels = np.argmin(distances, axis=1)
            new_centroids = self._update_centroids(X, new_labels)
            
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
            self.labels = new_labels
            
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
        """Predict cluster labels for new data."""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        """Fit the model and return cluster labels."""
        self.fit(X)
        return self.labels


# ============================================================
# 2. Linear Discriminant Analysis (LDA) from Scratch
# ============================================================
class LDAFromScratch:
    """Linear Discriminant Analysis implemented from scratch."""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.means_ = None
        self.scalings_ = None
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit LDA model to training data."""
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        else:
            self.n_components = min(self.n_components, n_classes - 1, n_features)
        
        # 1. Compute class-wise means and overall mean
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        overall_mean = X.mean(axis=0)
        
        # 2. Construct Within-Class Scatter Matrix (Sw)
        Sw = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            diff = X_c - self.means_[i]
            Sw += diff.T @ diff
        
        # 3. Construct Between-Class Scatter Matrix (Sb)
        Sb = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            n_c = X[y == c].shape[0]
            diff = self.means_[i] - overall_mean
            Sb += n_c * np.outer(diff, diff)
        
        # 4. Solve generalized eigenvalue problem: Sw^-1 Sb
        # Use pseudo-inverse for numerical stability
        Sw_inv = np.linalg.pinv(Sw)
        eigvals, eigvecs = np.linalg.eig(Sw_inv @ Sb)
        
        # 5. Sort eigenvectors by eigenvalues (descending)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        
        # 6. Select top n_components
        self.scalings_ = eigvecs[:, :self.n_components]
        
    def transform(self, X):
        """Project data onto discriminant components."""
        return X @ self.scalings_
    
    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)


# ============================================================
# 3. Utility Functions
# ============================================================
def generate_sample_data():
    """Generate sample data for clustering demonstration."""
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, 
                     cluster_std=1.5, random_state=42)
    X += np.random.normal(0, 0.2, X.shape)
    return X, y

def generate_classification_data():
    """Generate sample data for LDA demonstration."""
    X, y = make_blobs(n_samples=200, centers=3, n_features=4, 
                     cluster_std=1.5, random_state=42)
    return X, y

def visualize_kmeans(X, labels, centroids, title, save_path=None):
    """Visualize K-Means clustering results."""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                         alpha=0.6, s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', 
               marker='x', s=200, linewidths=3, label='Centroids')
    plt.title(title, fontsize=14)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return plt

def visualize_lda(X_lda, y, title, save_path=None):
    """Visualize LDA projection results."""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', 
                         alpha=0.6, s=50)
    plt.title(title, fontsize=14)
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return plt

def elbow_method(X, max_k=10, save_path=None):
    """Apply elbow method to determine optimal number of clusters."""
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeansFromScratch(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.title('Elbow Method for Optimal K', fontsize=14)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squared Distances)')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return plt, k_range, inertias


# ============================================================
# 4. Main Execution
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("A6: Python Programming Task - K-Means & LDA from Scratch")
    print("=" * 60)
    
    # Generate sample data
    X_kmeans, _ = generate_sample_data()
    X_lda, y_lda = generate_classification_data()
    
    # Standardize data for LDA
    scaler = StandardScaler()
    X_lda_scaled = scaler.fit_transform(X_lda)
    
    # --- K-Means Clustering ---
    print("\n1. K-Means Clustering from Scratch")
    print("-" * 40)
    
    # Test with different K values
    for k in [2, 3, 4]:
        kmeans = KMeansFromScratch(n_clusters=k, random_state=42)
        predicted_labels = kmeans.fit_predict(X_kmeans)
        print(f"  K={k}: Inertia = {kmeans.inertia:.2f}")
        
        # Visualize
        plt = visualize_kmeans(X_kmeans, predicted_labels, kmeans.centroids,
                              f'K-Means Clustering (K={k})',
                              f'D:/AI_Pravartak/Day8_kmeans/output/kmeans_k{k}.png')
        plt.close()
    
    # Elbow method
    plt, k_range, inertias = elbow_method(X_kmeans, max_k=10,
                                          save_path='D:/AI_Pravartak/Day8_kmeans/output/elbow_method.png')
    plt.close()
    print(f"  Elbow method completed. Inertias: {[f'{i:.1f}' for i in inertias]}")
    
    # --- LDA from Scratch ---
    print("\n2. Linear Discriminant Analysis from Scratch")
    print("-" * 40)
    
    lda = LDAFromScratch(n_components=2)
    X_lda_transformed = lda.fit_transform(X_lda_scaled, y_lda)
    print(f"  Original shape: {X_lda.shape}")
    print(f"  Transformed shape: {X_lda_transformed.shape}")
    print(f"  Explained variance ratio (eigenvalues): {lda.scalings_[:, 0] if lda.scalings_ is not None else 'N/A'}")
    
    # Visualize LDA
    plt = visualize_lda(X_lda_transformed, y_lda,
                       'LDA Projection (from Scratch)',
                       'D:/AI_Pravartak/Day8_kmeans/output/lda_projection.png')
    plt.close()
    
    # --- Comparison with sklearn ---
    print("\n3. Comparison with sklearn")
    print("-" * 40)
    
    from sklearn.cluster import KMeans as SklearnKMeans
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
    
    # K-Means comparison
    kmeans_sklearn = SklearnKMeans(n_clusters=3, random_state=42)
    kmeans_sklearn.fit(X_kmeans)
    print(f"  Sklearn K-Means inertia: {kmeans_sklearn.inertia_:.2f}")
    
    kmeans_scratch = KMeansFromScratch(n_clusters=3, random_state=42)
    kmeans_scratch.fit(X_kmeans)
    print(f"  Scratch K-Means inertia: {kmeans_scratch.inertia:.2f}")
    
    # LDA comparison
    lda_sklearn = SklearnLDA(n_components=2)
    X_lda_sklearn = lda_sklearn.fit_transform(X_lda_scaled, y_lda)
    print(f"  Sklearn LDA shape: {X_lda_sklearn.shape}")
    print(f"  Scratch LDA shape: {X_lda_transformed.shape}")
    
    print("\n" + "=" * 60)
    print("All tasks completed successfully!")
    print("Visualizations saved to output/")
    print("=" * 60)