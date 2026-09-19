# K-Means Clustering & Related Topics Implementation

## Algorithms
- **K-Means Clustering** - Unsupervised clustering algorithm
- **Bayesian Learning** - Probabilistic classification using Bayes' theorem
- **K-Nearest Neighbors (KNN)** - Supervised lazy learning algorithm
- **Linear Discriminant Analysis (LDA)** - Supervised dimensionality reduction & classification

## Overview
This project implements K-Means clustering, LDA, and related concepts with practical Python exercises.

## Datasets
- **Synthetic K-Means Data**: `sklearn.datasets.make_blobs` - 300 samples, 3 centers, 2 features
- **Synthetic LDA Data**: `sklearn.datasets.make_blobs` - 200 samples, 3 centers, 4 features
- **Sample Dataset**: `data/sample_dataset.csv` - 15 samples with 4 features and 3 classes

## Implementation Details

### 1. Theory & Conceptual Understanding
- K-Means fundamentals: objective, K parameter, steps, distance metrics, centroids, elbow method, limitations
- Unsupervised learning: clustering vs classification, SSE interpretation, elbow method, feature scaling, outliers, hard/soft clustering
- K-Means numerical problems: manual iterations, centroid updates, SSE calculation, elbow method interpretation
- Bayes Learning: Bayes' theorem, prior/posterior/likelihood, MAP estimation, Naive Bayes, diagnostic testing
- KNN: lazy learning, distance metrics, K selection, feature scaling, overfitting, curse of dimensionality

### 2. Python Implementation from Scratch

**K-Means Clustering (`KMeansFromScratch` class):**
1. Centroid initialization (random selection from data points)
2. Distance calculation (Euclidean distance to all centroids)
3. Cluster assignment (nearest centroid)
4. Centroid update (mean of assigned points)
5. Iteration until convergence
6. Inertia calculation (within-cluster SSE)

**Linear Discriminant Analysis (`LDAFromScratch` class):**
1. Compute class-wise means and overall mean
2. Construct Within-Class Scatter Matrix (Sw)
3. Construct Between-Class Scatter Matrix (Sb)
4. Solve generalized eigenvalue problem: Sw⁻¹Sb
5. Select top discriminant components (max C-1)
6. Project data onto discriminant subspace

### 2. Comparison with Scikit-learn
- **K-Means**: Scratch inertia matches sklearn inertia exactly
- **LDA**: Both produce (200, 2) projections from 4D data

### 3. Visualizations Generated
- `kmeans_k2.png` - K-Means with K=2
- `kmeans_k3.png` - K-Means with K=3
- `kmeans_k4.png` - K-Means with K=4
- `elbow_method.png` - Elbow method for optimal K selection
- `lda_projection.png` - LDA 2D projection of 4D data

## Output/Results

### K-Means Inertia Values:
| K | Inertia |
|---|---------|
| 1 | 21288.2 |
| 2 | 6584.7 |
| 3 | 1297.0 |
| 4 | 1134.2 |
| 5 | 981.6 |
| 6 | 856.3 |
| 7 | 769.6 |
| 8 | 732.1 |
| 9 | 700.6 |
| 10 | 657.0 |

**Elbow Point**: K=3 (significant drop from K=2 to K=3, diminishing returns after)

### LDA Results:
- Original: 200 samples × 4 features
- Transformed: 200 samples × 2 discriminant components
- Explained variance captured in eigenvalues

## Project Structure
```
Day8_kmeans/
├── code/
│   ├── notebooks/          # Jupyter notebooks (assignment deliverables)
│   │   ├── 01_kmeans_exploration.ipynb      # A1, A2, A3 - K-Means concepts & problems
│   │   └── 02_lda_exploration.ipynb         # A6 - LDA from scratch
│   └── python/             # Python package (reusable modules)
│       ├── __init__.py
│       ├── kmeans_from_scratch.py
│       ├── kmeans_vs_pca_comparison.py
│       ├── functions_modularity.py
│       └── terminal_calculator.py
├── data/
├── output/
├── report/
└── README.md
```


## How to Run

### Jupyter Notebooks
```bash
# From workspace root
cd Day8_kmeans/code/notebooks

# Start Jupyter
jupyter notebook

# Run notebooks
python -m nbconvert --to notebook --execute 01_kmeans_exploration.ipynb
python -m nbconvert --to notebook --execute 02_lda_exploration.ipynb
```

### Python Scripts
```bash
# From workspace root
cd Day8_kmeans

# Run K-Means standalone demo
python code/python/kmeans_from_scratch.py

# Run K-Means vs PCA comparison
python code/python/kmeans_vs_pca_comparison.py

# Run combined K-Means & LDA with all practical exercises
python code/python/functions_modularity.py

# Run terminal calculator (interactive)
python code/python/terminal_calculator.py
```

### Import from Python Package
```python
import sys
sys.path.insert(0, 'code/python')

from kmeans_from_scratch import KMeansFromScratch
from functions_modularity import LDAFromScratch
```

## Key Takeaways
1. K-Means is unsupervised; LDA is supervised - fundamental difference
2. K-Means minimizes within-cluster SSE; LDA maximizes Fisher's criterion
3. Elbow method helps determine optimal K
4. Feature scaling is essential for both K-Means and LDA
5. Small sample size problem: use PCA first, then LDA
6. Modular code improves reusability, testing, and maintenance
7. Terminal execution enables automation and scripting