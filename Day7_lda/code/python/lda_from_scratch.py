"""
LDA Implementation from Scratch using NumPy
============================================
Demonstrates A1-A4 concepts: Manual LDA implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
        
        # Limit n_components to C-1
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
        
        # 4. Solve generalized eigenvalue problem: Sw^(-1) Sb v = λ v
        # Use pseudo-inverse for numerical stability
        Sw_inv = np.linalg.pinv(Sw)
        eigvals, eigvecs = np.linalg.eig(Sw_inv @ Sb)
        
        # 5. Sort eigenvectors by eigenvalues descending
        idx = np.argsort(eigvals.real)[::-1]
        eigvecs = eigvecs[:, idx].real
        
        # 6. Select top k discriminant vectors
        self.scalings_ = eigvecs[:, :self.n_components]
        
        # Store centroids for prediction
        X_lda = X @ self.scalings_
        self.centroids_ = np.array([X_lda[y == c].mean(axis=0) for c in self.classes_])
        
        return self
    
    def transform(self, X):
        """Project data onto discriminant vectors."""
        return X @ self.scalings_
    
    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def predict(self, X):
        """Predict class labels using nearest centroid in LDA space."""
        X_lda = self.transform(X)
        # Compute centroids from training data (stored during fit)
        if not hasattr(self, 'centroids_'):
            raise ValueError("Model must be fitted before prediction")
        distances = np.linalg.norm(X_lda[:, np.newaxis] - self.centroids_, axis=2)
        return self.classes_[np.argmin(distances, axis=1)]


def demo_lda():
    """Demonstrate LDA on synthetic data."""
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=300, n_features=4, n_informative=3,
        n_redundant=1, n_classes=3, n_clusters_per_class=1,
        random_state=42
    )
    
    # Standardize features (important for LDA!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # Our implementation
    lda_scratch = LDAFromScratch(n_components=2)
    X_train_lda = lda_scratch.fit_transform(X_train, y_train)
    X_test_lda = lda_scratch.transform(X_test)
    y_pred = lda_scratch.predict(X_test)
    
    # Compare with scikit-learn
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda_sklearn = LinearDiscriminantAnalysis(n_components=2)
    X_train_sk = lda_sklearn.fit_transform(X_train, y_train)
    X_test_sk = lda_sklearn.transform(X_test)
    y_pred_sk = lda_sklearn.predict(X_test)
    
    # Accuracy
    acc_scratch = np.mean(y_pred == y_test)
    acc_sklearn = np.mean(y_pred_sk == y_test)
    
    print(f"From Scratch Accuracy: {acc_scratch:.4f}")
    print(f"Scikit-learn Accuracy: {acc_sklearn:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # From scratch
    scatter = axes[0].scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_test, cmap='viridis', alpha=0.7)
    axes[0].set_title(f'LDA From Scratch (Acc: {acc_scratch:.3f})')
    axes[0].set_xlabel('LD1')
    axes[0].set_ylabel('LD2')
    plt.colorbar(scatter, ax=axes[0])
    
    # Scikit-learn
    scatter = axes[1].scatter(X_test_sk[:, 0], X_test_sk[:, 1], c=y_test, cmap='viridis', alpha=0.7)
    axes[1].set_title(f'Scikit-learn LDA (Acc: {acc_sklearn:.3f})')
    axes[1].set_xlabel('LD1')
    axes[1].set_ylabel('LD2')
    plt.colorbar(scatter, ax=axes[1])
    
    plt.tight_layout()
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', '..', 'output', 'lda_comparison.png')
    plt.savefig(output_path, dpi=150)
    plt.show()
    
    return lda_scratch, lda_sklearn


if __name__ == "__main__":
    demo_lda()