"""
LDA vs PCA Comparison
=====================
Demonstrates A5 concepts: When to use LDA vs PCA
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def generate_data():
    """Generate dataset where PCA and LDA give different results."""
    # Create data where max variance direction ≠ class separation direction
    np.random.seed(42)
    
    # Class 0: centered at (-2, 0), elongated along x-axis
    n = 200
    X0 = np.random.randn(n, 2)
    X0[:, 0] *= 3  # High variance along x
    X0 += [-2, 0]
    
    # Class 1: centered at (2, 0), elongated along x-axis  
    X1 = np.random.randn(n, 2)
    X1[:, 0] *= 3  # High variance along x
    X1 += [2, 0]
    
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)])
    
    return X, y


def demo_pca_vs_lda():
    """Compare PCA and LDA on the same data."""
    X, y = generate_data()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA (unsupervised - ignores labels)
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_scaled)
    
    # LDA (supervised - uses labels)
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X_scaled, y)
    
    # Classification on reduced data
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.3, random_state=42
    )
    X_train_lda, X_test_lda, _, _ = train_test_split(
        X_lda, y, test_size=0.3, random_state=42
    )
    
    # Train classifiers
    clf_pca = LogisticRegression(random_state=42)
    clf_lda = LogisticRegression(random_state=42)
    
    clf_pca.fit(X_train_pca, y_train)
    clf_lda.fit(X_train_lda, y_train)
    
    acc_pca = accuracy_score(y_test, clf_pca.predict(X_test_pca))
    acc_lda = accuracy_score(y_test, clf_lda.predict(X_test_lda))
    
    print(f"PCA + LogisticRegression Accuracy: {acc_pca:.4f}")
    print(f"LDA + LogisticRegression Accuracy: {acc_lda:.4f}")
    print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"LDA uses class labels: Yes")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data
    axes[0, 0].scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], alpha=0.5, label='Class 0')
    axes[0, 0].scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], alpha=0.5, label='Class 1')
    axes[0, 0].set_title('Original Data (Standardized)')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    axes[0, 0].legend()
    axes[0, 0].axis('equal')
    
    # PCA projection
    axes[0, 1].hist(X_pca[y==0], alpha=0.5, bins=30, label='Class 0', density=True)
    axes[0, 1].hist(X_pca[y==1], alpha=0.5, bins=30, label='Class 1', density=True)
    axes[0, 1].set_title(f'PCA Projection (1D)\nAcc: {acc_pca:.3f}')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    
    # LDA projection
    axes[1, 0].hist(X_lda[y==0], alpha=0.5, bins=30, label='Class 0', density=True)
    axes[1, 0].hist(X_lda[y==1], alpha=0.5, bins=30, label='Class 1', density=True)
    axes[1, 0].set_title(f'LDA Projection (1D)\nAcc: {acc_lda:.3f}')
    axes[1, 0].set_xlabel('LD1')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    
    # Decision boundaries in original space
    xx, yy = np.meshgrid(
        np.linspace(X_scaled[:, 0].min()-1, X_scaled[:, 0].max()+1, 100),
        np.linspace(X_scaled[:, 1].min()-1, X_scaled[:, 1].max()+1, 100)
    )
    
    # PCA boundary
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_pca = pca.transform(grid)
    Z_pca = clf_pca.predict(grid_pca).reshape(xx.shape)
    
    axes[1, 1].contourf(xx, yy, Z_pca, alpha=0.3, cmap='coolwarm')
    axes[1, 1].scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], alpha=0.5, label='Class 0')
    axes[1, 1].scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], alpha=0.5, label='Class 1')
    axes[1, 1].set_title('PCA Decision Boundary')
    axes[1, 1].set_xlabel('Feature 1')
    axes[1, 1].set_ylabel('Feature 2')
    axes[1, 1].legend()
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', '..', 'output', 'pca_vs_lda_comparison.png')
    plt.savefig(output_path, dpi=150)
    plt.show()
    
    return acc_pca, acc_lda


def demo_high_dimensional():
    """Show PCA before LDA for high-dimensional data."""
    # High-dimensional data with few samples
    X, y = make_classification(
        n_samples=50, n_features=100, n_informative=10,
        n_redundant=20, n_classes=3, random_state=42
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Direct LDA fails (singular Sw)
    try:
        lda_direct = LinearDiscriminantAnalysis(n_components=2)
        X_lda_direct = lda_direct.fit_transform(X_scaled, y)
        print("Direct LDA succeeded")
    except Exception as e:
        print(f"Direct LDA failed: {type(e).__name__}")
    
    # PCA first, then LDA
    pca = PCA(n_components=30)  # Reduce to n_samples - 1
    X_pca = pca.fit_transform(X_scaled)
    
    lda_after_pca = LinearDiscriminantAnalysis(n_components=2)
    X_lda_pca = lda_after_pca.fit_transform(X_pca, y)
    print(f"PCA + LDA succeeded: shape {X_lda_pca.shape}")
    
    return X_lda_pca


if __name__ == "__main__":
    print("=" * 60)
    print("PCA vs LDA Comparison")
    print("=" * 60)
    demo_pca_vs_lda()
    
    print("\n" + "=" * 60)
    print("High-Dimensional: PCA before LDA")
    print("=" * 60)
    demo_high_dimensional()