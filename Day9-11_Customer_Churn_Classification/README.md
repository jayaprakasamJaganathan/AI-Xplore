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

## Datasets

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

## Results

### Evaluation Metrics
| Metric | Formula | Business Meaning |
|--------|---------|------------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted churners, how many actually churn |
| **Recall** | TP / (TP + FN) | Of actual churners, how many we caught |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |

### Model Comparison
- **Highest Recall** (catches most churners): Typically Random Forest or Logistic Regression
- **Highest Precision** (fewest false alarms): Typically Logistic Regression or Random Forest
- **Trade-off**: Different models optimize for different business objectives

### Key Features Driving Churn
1. `satisfaction_score` - Strong negative correlation with churn
2. `weekly_usage_hours` - Low usage indicates churn risk
3. `support_tickets_6m` - High support tickets indicate dissatisfaction
4. `competitor_offer` - Receiving competitor offers increases churn risk
5. `tenure_months` - Newer customers more likely to churn

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
- **Random Forest** typically provides best balance of performance and interpretability
- **Feature importance** reveals actionable business drivers (satisfaction, usage, support tickets)
- **Pipeline approach** prevents data leakage during scaling
- **Stratified splitting** ensures representative class distribution in both sets

## Dataset Description

**Source**: `customer_churn_classification.xlsx` (Customer data sheet)

| Column | Description | Values/Unit |
|--------|-------------|-------------|
| customer_id | Unique identifier (do not use as predictor) | - |
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

## Models Implemented

### 1. K-Nearest Neighbors (KNN)
- Pipeline with StandardScaler
- Hyperparameter tuning for K (1-20)
- Distance-weighted voting option

### 2. Gaussian Naive Bayes
- No scaling required
- Assumes Gaussian distribution for continuous features

### 3. Decision Tree
- Default parameters with random_state=42
- Visualization with max_depth=3 for interpretability
- Feature importance analysis

### 4. Random Forest
- 100 estimators, random_state=42
- Feature importance ranking
- Bootstrap sampling with out-of-bag evaluation

### 5. Logistic Regression
- Pipeline with StandardScaler
- L2 regularization (default)
- max_iter=1000 for convergence
- Coefficient interpretation for business insights

## Evaluation Metrics

| Metric | Formula | Business Meaning |
|--------|---------|------------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted churners, how many actually churn |
| **Recall** | TP / (TP + FN) | Of actual churners, how many we caught |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |

## Key Findings

### Model Comparison (Expected Results)
- **Highest Recall** (catches most churners): Typically Random Forest or Logistic Regression
- **Highest Precision** (fewest false alarms): Typically Logistic Regression or Random Forest
- **Trade-off**: Different models optimize for different business objectives

### Feature Importance (Typical Top Features)
1. `satisfaction_score` - Strong negative correlation with churn
2. `weekly_usage_hours` - Low usage indicates churn risk
3. `support_tickets_6m` - High support tickets indicate dissatisfaction
4. `competitor_offer` - Receiving competitor offers increases churn risk
5. `tenure_months` - Newer customers more likely to churn

## Recommendation for Retention Campaign

**Primary Goal**: Catch customers who will cancel (maximize Recall)

**Recommended Model**: Based on evaluation results (typically Random Forest or Logistic Regression)

**Rationale**:
- Retention campaigns prioritize **recall** - missing a churner (false negative) means losing a customer
- False alarms (false positives) mean wasted retention budget on loyal customers - acceptable cost
- Model provides interpretable feature importance for targeted interventions

## Running the Notebook

```bash
# Activate environment
d:\AI_Pravartak\ai_env\Scripts\Activate.ps1

# Install dependencies (if needed)
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl

# Run notebook
jupyter notebook code/notebooks/customer_churn_implementation.ipynb
```

## Dependencies

- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- openpyxl (for Excel reading)

## Source Files

All answers and implementation are based on the following source files in the assignment directory:
- `Foundational MCQ.pdf` - 50 MCQs on KNN, Naive Bayes, Decision Trees, Random Forest, Logistic Regression
- `Foundational MCQ 2.pdf` - 30 scenario-based MCQs
- `Scenario Based Questions.pdf` - 30 scenario-based questions (duplicate of MCQ 2)
- `Python assignment.pdf` - Practical coding assignment (implemented in notebook)
- `customer_churn_classification.xlsx` - Dataset with Customer data and Data dictionary sheets

## Notes

- All MCQ answers are extracted directly from source PDFs using pymupdf (fitz)
- No external knowledge was used - answers strictly follow source material
- Implementation follows the exact steps specified in `Python assignment.pdf`
- The notebook uses pymupdf for PDF text extraction (available as `fitz` module)