# Customer Churn Classification

## Overview

This project implements customer churn classification using multiple machine learning models to identify customers likely to cancel their subscription.

## Algorithm

The project compares five classification algorithms:
- **K-Nearest Neighbors (KNN)** - Distance-based classification with feature scaling
- **Gaussian Naive Bayes** - Probabilistic classifier assuming Gaussian feature distributions
- **Decision Tree** - Rule-based classifier with interpretable splits
- **Random Forest** - Ensemble of decision trees with feature importance
- **Logistic Regression** - Linear model with probability outputs

## Dataset

**Source**: `customer_churn_classification.xlsx` (Customer data sheet)

| Column | Description | Values/Unit |
|--------|-------------|-------------|
| customer_id | Unique identifier (not used as predictor) | - |
| age_years | Customer age | Years |
| tenure_months | Months as customer | Months |
| monthly_fee_inr | Monthly subscription fee | Indian Rupees |
| weekly_usage_hours | Weekly product usage | Hours |
| support_tickets_6m | Support tickets in last 6 months | Count |
| late_payments_6m | Late payments in last 6 months | Count |
| discount_percent | Discount percentage | Percent |
| auto_pay | Auto-payment enabled | Binary (0/1) |
| satisfaction_score | Customer satisfaction score | 1-10 scale |
| competitor_offer | Competitor offer received | Binary (0/1) |
| churned | **Target variable** - Customer canceled | Binary (0/1) |

**Data Shape**: 500 samples × 12 columns (10 features + 1 target + 1 ID)

## Implementation Details

### Concepts Covered
- Stratified train/test splitting
- Feature scaling with StandardScaler (pipelines)
- Model training and evaluation
- Hyperparameter tuning (K for KNN)
- Model interpretability (feature importance, coefficients, tree visualization)
- Business metric optimization (precision vs recall trade-off)

### Key Implementation Details
- **Pipelines**: KNN and Logistic Regression use StandardScaler fitted only on training data
- **Stratification**: Maintains class proportions in train/test splits
- **Random State**: Fixed (42) for reproducibility
- **Cross-validation**: Used for K selection in KNN

### Models Implemented

**1. K-Nearest Neighbors (KNN)**
- Pipeline with StandardScaler
- Hyperparameter tuning for K (1-20)
- Distance-weighted voting option

**2. Gaussian Naive Bayes**
- No scaling required
- Assumes Gaussian distribution for continuous features

**3. Decision Tree**
- Default parameters with random_state=42
- Visualization with max_depth=3 for interpretability
- Feature importance analysis

**4. Random Forest**
- 100 estimators, random_state=42
- Feature importance ranking
- Bootstrap sampling with out-of-bag evaluation

**5. Logistic Regression**
- Pipeline with StandardScaler
- L2 regularization (default)
- max_iter=1000 for convergence
- Coefficient interpretation for business insights

## Results

### Evaluation Metrics
| Metric | Formula | Business Meaning |
|--------|---------|------------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted churners, how many actually churn |
| **Recall** | TP / (TP + FN) | Of actual churners, how many we caught |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |

### Model Comparison
- **Highest Recall** (catches most churners): Random Forest (0.75)
- **Highest Precision** (fewest false alarms): Random Forest (0.75)
- **Trade-off**: Different models optimize for different business objectives

### Key Features Driving Churn (from Random Forest)
1. `weekly_usage_hours` (0.186) - Low usage indicates churn risk
2. `satisfaction_score` (0.155) - Strong negative correlation with churn
3. `tenure_months` (0.127) - Newer customers more likely to churn
4. `monthly_fee_inr` (0.118) - Higher fees correlate with churn
5. `age_years` (0.111) - Age contributes to churn prediction

## Recommendation for Retention Campaign

**Primary Goal**: Catch customers who will cancel (maximize Recall)

**Recommended Model**: **Random Forest** (Recall=0.75, Precision=0.75, F1=0.75)

**Rationale**:
- Retention campaigns prioritize **recall** - missing a churner (false negative) means losing a customer
- False alarms (false positives) mean wasted retention budget on loyal customers - acceptable cost
- Model provides interpretable feature importance for targeted interventions

## How to Run

```bash
# Navigate to notebook directory
cd code/notebooks

# Start Jupyter
jupyter notebook customer_churn_implementation.ipynb
```

### Dependencies
- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- openpyxl (for Excel reading)

## Key Takeaways

- **For retention campaigns**: Prioritize recall (catch churners) over precision
- **Random Forest** provides best balance of performance and interpretability
- **Feature importance** reveals actionable business drivers (usage, satisfaction, tenure)
- **Pipeline approach** prevents data leakage during scaling
- **Stratified splitting** ensures representative class distribution in both sets