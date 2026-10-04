# Day 4: Pandas Data Analysis

This project contains the complete implementation for Day 4 assignments covering Pandas data analysis and MCQ assessments.

## Project Structure

```
Day4_Pandas_Analysis/
├── code/
│   └── notebooks/
│       ├── assignment1_patient_clinical_analysis.ipynb
│       ├── assignment2_ecommerce_sales_analysis.ipynb
│       ├── assignment3_iot_sensor_analysis.ipynb
│       └── assignment4_customer_product_order_analysis.ipynb
├── data/
│   ├── patient_clinical_data_raw.csv
│   ├── ecommerce_orders_raw.csv
│   ├── iot_sensor_data_raw.csv
│   ├── customers.csv
│   ├── products.csv
│   └── orders.csv
├── report/
│   ├── A1_Patient_Clinical_Analysis_answers.txt
│   ├── A2_Ecommerce_Sales_Analysis_answers.txt
│   ├── A3_IoT_Sensor_Analysis_answers.txt
│   ├── A4_Customer_Product_Order_Analysis_answers.txt
│   ├── A5_Linear_Algebra_MCQ_answers.txt
│   ├── A6_Probability_MCQ_answers.txt
│   ├── A7_Statistics_MCQ_answers.txt
│   └── A8_Probability_vs_Statistics_MCQ_answers.txt
└── README.md
```

## Assignments Overview

### Practical Assignments (Notebooks)

| Assignment | Notebook | Dataset | Focus |
|------------|----------|---------|-------|
| **1** | `assignment1_patient_clinical_analysis.ipynb` | `patient_clinical_data_raw.csv` | Clinical data cleaning, transformation, feature engineering |
| **2** | `assignment2_ecommerce_sales_analysis.ipynb` | `ecommerce_orders_raw.csv` | E-commerce data cleaning, feature engineering |
| **3** | `assignment3_iot_sensor_analysis.ipynb` | `iot_sensor_data_raw.csv` | IoT sensor time-series analysis, anomaly detection |
| **4** | `assignment4_customer_product_order_analysis.ipynb` | `customers.csv`, `products.csv`, `orders.csv` | Multi-table merging, business analysis |

### MCQ Assignments (Answer Sheets)

| Assignment | Topic | Questions |
|------------|-------|-----------|
| **5** | Linear Algebra | 25 MCQs |
| **6** | Probability | 25 MCQs |
| **7** | Statistics | 25 MCQs |
| **8** | Probability vs Statistics | 23 MCQs |

## How to Run

### Prerequisites
- Python 3.10+
- Pandas, NumPy
- Jupyter Notebook

### Running Notebooks

```bash
cd code/notebooks
jupyter notebook
```

Open any of the four assignment notebooks and run cells sequentially.

### Data
All required CSV files are in the `data/` folder. No external downloads needed.

## Assignment Details

### Assignment 1: Patient Clinical Data Analysis
- **Dataset**: `patient_clinical_data_raw.csv`
- **Parts**: Understanding (6 tasks), Cleaning (8 tasks), Transformation (4 tasks)
- **Key Operations**: Missing value handling, duplicate removal, age validation, categorical standardization, age grouping, blood pressure parsing, risk scoring, cost calculation

### Assignment 2: E-Commerce Sales Analysis
- **Dataset**: `ecommerce_orders_raw.csv`
- **Parts**: Understanding (6 tasks), Cleaning (5 tasks), Feature Engineering (4 tasks)
- **Key Operations**: Rating imputation, city standardization, quantity validation, date parsing, gross/discount/net amount calculation, rating categorization

### Assignment 3: IoT Sensor Data Analysis
- **Dataset**: `iot_sensor_data_raw.csv`
- **Parts**: Preparation (6 tasks), Cleaning (5 tasks), Time-Series Analysis (7 tasks), New Variables (1 task)
- **Key Operations**: Timestamp parsing, sorting, missing value interpolation, 3-sigma anomaly detection, hourly/device/factory aggregations, battery status classification

### Assignment 4: Customer, Product & Order Analysis
- **Datasets**: `customers.csv`, `products.csv`, `orders.csv`
- **Parts**: Understanding (5 tasks), Cleaning (6 tasks), Combining (3 tasks)
- **Key Operations**: Multi-table merging (left joins), referential integrity checks, derived metrics (gross/net amounts), business insights (top customers/products/cities)

### MCQ Assignments (5-8)
- **Assignment 5**: Linear Algebra (vectors, matrices, eigenvalues, linear transformations)
- **Assignment 6**: Probability (basic, conditional, random variables, rules)
- **Assignment 7**: Statistics (central tendency, dispersion, distributions, inference)
- **Assignment 8**: Probability vs Statistics (conceptual differentiation)

## Verification Checklist

- [x] All 4 practical notebooks created and executable
- [x] All 6 CSV data files present in `data/`
- [x] 8 answer sheets created (4 practical + 4 MCQ)
- [x] No `code/python/` folder (notebooks only)
- [x] No external dataset downloads required
- [x] All notebooks generate synthetic data in-code where needed
- [x] Notebooks run offline (no internet required)

## Key Principles Followed

- **Conditional creation**: Only folders/files required by assignments created
- **No external dependencies**: All data provided locally
- **Self-contained notebooks**: No imports from `code/python/`
- **Synthetic data in-code**: No external downloads for standard datasets
- **Offline capability**: All notebooks run without internet
- **Relative paths**: `../data/` from notebook location