# Linear Discriminant Analysis (LDA) Implementation

## Overview
This project implements Linear Discriminant Analysis (LDA) for dimensionality reduction and classification, with comparisons to PCA and practical Python programming exercises.

## Algorithm
**Linear Discriminant Analysis (LDA)** - Supervised Dimensionality Reduction & Classification

## Datasets
- **Synthetic Dataset**: Generated using `sklearn.datasets.make_classification` with 3 classes, 4 features
- **Sample Dataset**: `data/sample_dataset.csv` - 15 samples with 4 features and 3 classes
- **High-dimensional Test**: 50 samples, 100 features (for PCA+LDA pipeline demonstration)

## Implementation Details

### 1. Theory & Conceptual Understanding
- LDA fundamentals: Fisher's criterion, scatter matrices (Sw, Sb)
- Assumptions: Gaussian classes, equal covariance matrices
- Limitations: C-1 components max, small sample size problem, sensitivity to outliers

### 2. Scikit-learn Implementation
- Used `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
- Demonstrated `fit()`, `transform()`, `fit_transform()`, `predict()` methods
- Explored parameters: `n_components`, `solver`, `shrinkage`
- Applied feature scaling before LDA (critical for performance)

### 3. From-Scratch Implementation
Implemented complete LDA pipeline:
1. Compute class-wise means and overall mean
2. Construct Within-Class Scatter Matrix (Sw)
3. Construct Between-Class Scatter Matrix (Sb)
4. Solve generalized eigenvalue problem: Sw⁻¹Sb
5. Select top k discriminant vectors (largest eigenvalues)
6. Project data onto new subspace
7. Visualize transformed data
- Handled singular Sw using pseudo-inverse (`np.linalg.pinv`)
- Compared results with scikit-learn implementation

### 4. LDA vs PCA Comparison
- PCA: Unsupervised, maximizes variance
- LDA: Supervised, maximizes class separation
- When max variance direction ≠ class separation direction, LDA outperforms PCA
- For high-dimensional data: PCA before LDA addresses singularity

### 5. Functions & Modularity
Implemented 9 practical exercises:
1. `calculate_average()` - list average without NumPy
2. `find_maximum()` - max without built-in max()
3. Modular program: read → mean → max → display
4. `count_even()` - count even numbers
5. `calculate_accuracy()` + `display_result()` - classification metrics
6. `mean_and_std()` - returns both statistics
7. `normalize()` - min-max normalization
8. ML Pipeline: load → remove_missing → normalize → display
9. `euclidean_distance()` - without built-in functions

### 6. Terminal Execution
- Created terminal calculator with separate functions: `add()`, `subtract()`, `multiply()`, `divide()`
- User input handling with menu-driven interface
- Executed via: `python terminal_calculator.py`

## Results

### LDA Implementation Comparison
- **From Scratch Accuracy**: ~0.9333
- **Scikit-learn Accuracy**: ~0.9333
- Both implementations produce equivalent results
- Visualization saved to `output/lda_comparison.png`

### PCA vs LDA Comparison
- **PCA + LogisticRegression Accuracy**: ~0.5000 (fails when variance ≠ separation)
- **LDA + LogisticRegression Accuracy**: ~1.0000 (uses label information)
- Visualization saved to `output/pca_vs_lda_comparison.png`

### High-Dimensional Pipeline
- Direct LDA fails on 100 features / 50 samples (singular Sw)
- PCA (30 components) + LDA succeeds
- Demonstrates practical preprocessing pipeline

### Function Outputs
All modular functions tested and verified:
- Average, maximum, even count, accuracy, normalization, distance calculations
- ML preprocessing pipeline executes successfully

### Terminal Calculator
- Interactive menu-driven calculator
- Handles division by zero
- Accepts user input for two numbers
- Performs all four basic operations

## How to Run

### Jupyter Notebooks
```bash
# From workspace root
cd Day7_lda/code/notebooks

# Start Jupyter
jupyter notebook

# Run notebooks
python -m nbconvert --to notebook --execute 03_functions_modularity.ipynb
python -m nbconvert --to notebook --execute 04_terminal_calculator.ipynb
```

### Python Scripts
```bash
# From workspace root
cd Day7_lda

# Run LDA from scratch demo
python code/python/lda_from_scratch.py

# Run PCA vs LDA comparison
python code/python/lda_vs_pca_comparison.py

# Run functions & modularity demos
python code/python/functions_modularity.py

# Run terminal calculator (interactive)
python code/python/terminal_calculator.py
```

### Import from Python Package
```python
import sys
sys.path.insert(0, 'code/python')

from lda_from_scratch import LDAFromScratch
from functions_modularity import calculate_average, find_maximum
```

## Key Takeaways
1. LDA is supervised; PCA is unsupervised - fundamental difference
2. LDA maximizes Fisher's criterion: between-class / within-class scatter
3. Maximum discriminant components = C - 1 (where C = number of classes)
4. Feature scaling is essential before LDA
5. Small sample size problem: use PCA first, then LDA
6. Modular code improves reusability, testing, and maintenance
7. Terminal execution enables automation and scripting