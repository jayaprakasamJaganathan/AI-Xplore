# Day 2: Python Basics - 3D Arrays and Dictionaries

## Overview

This project implements solutions for Python basics covering 3D array manipulation with NumPy and dictionary operations.

## Topics Covered

### Part A: 3D Arrays (NumPy)
- Creating 3D arrays
- Array indexing and slicing
- Reversing layers, rows, and elements
- Complete array reversal

### Part B: Dictionaries
- Basic dictionary creation and manipulation
- Updating values and adding keys
- Dictionary methods (keys, values, items)
- Nested dictionaries
- Product inventory management
- Student marks analysis

## Dataset

No external dataset required. All data is generated in-code using NumPy and Python dictionaries.

## Implementation Details

### Concepts Covered
- NumPy 3D array creation and shape manipulation
- Advanced slicing with `[::-1]` for reversal
- Dictionary CRUD operations
- Dictionary methods: `.keys()`, `.values()`, `.items()`
- Nested dictionary access
- Lambda functions with `max()`/`min()` for finding extremes
- Aggregation operations (sum, average)

### Key Implementation Details
- **NumPy arrays** used for efficient 3D array operations
- **Slicing syntax** `arr[::-1]` reverses along an axis
- **Dictionary comprehension** not needed - direct methods used
- **Lambda functions** for custom sorting/finding

## Results

### Question 1: 3D Array Basics
- Array shape: (2, 2, 3) - 2 layers, 2 rows, 3 columns
- First layer access: `arr[0]`
- Element access: `arr[layer, row, column]`

### Question 2: Reverse Layers
- `arr[::-1]` reverses layer order

### Question 3: Reverse Rows
- `arr[:, :, ::-1]` reverses elements in each row

### Question 4: Complete Reverse
- `arr[::-1, ::-1, ::-1]` reverses all three axes

### Question 5-7: Basic Dictionary Operations
- Created student dictionary with 5 fields
- Updated CGPA, added City
- Used `.keys()`, `.values()`, `.items()`

### Question 8: Nested Dictionary
- Access nested values: `students['Student2']['Name']`
- Loop through nested structure

### Question 9: Product Inventory
- Highest price: Laptop (₹50,000)
- Total inventory value: ₹985,000

### Question 10: Student Marks Analysis
- Highest: Emma (95)
- Lowest: Charlie (78)
- Average: 87.60
- Above 85: Bob (92), David (88), Emma (95)

## How to Run

```bash
# Navigate to notebook directory
cd code/notebooks

# Start Jupyter
jupyter notebook assignment_2_basics_solution.ipynb
```

### Dependencies
- Python 3.10+
- numpy

## Key Takeaways

- **3D array indexing**: `array[layer, row, column]` for intuitive access
- **Slicing power**: `[::-1]` reverses any axis; combine for multi-axis reversal
- **Dictionary methods**: `.keys()`, `.values()`, `.items()` for different views
- **Nested access**: Chain keys `dict['outer']['inner']`
- **Lambda with max/min**: `max(dict.items(), key=lambda x: x[1])` finds max by value
- **Aggregation**: `sum()` and `len()` for averages