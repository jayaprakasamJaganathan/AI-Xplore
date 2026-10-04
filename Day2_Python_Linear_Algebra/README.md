# Day 2: Python Basics & Linear Algebra

## Overview

This project contains solutions for two Day 2 assignments:
1. **Python Basics** - 3D array manipulation with NumPy and dictionary operations
2. **Linear Algebra** - 23 problems covering scalars, vectors, matrices, vector operations, vector spaces, and orthogonality

## Topics Covered

### Assignment 1: Python Basics (Slicing, Dictionary)

#### Part A: 3D Arrays (NumPy)
- Creating 3D arrays
- Array indexing and slicing
- Reversing layers, rows, and elements
- Complete array reversal

#### Part B: Dictionaries
- Basic dictionary creation and manipulation
- Updating values and adding keys
- Dictionary methods (keys, values, items)
- Nested dictionaries
- Product inventory management
- Student marks analysis

### Assignment 2: Linear Algebra

#### Part A: Scalars, Vectors and Matrices
- Identifying scalars, vectors, matrices, tensors
- Matrix dimensions
- Matrix addition, subtraction, scalar multiplication
- Matrix transpose

#### Part B: Matrix Multiplication
- 2×2 matrix multiplication
- Multiplication compatibility rules
- Non-commutativity of matrix multiplication

#### Part C: Vector Operations
- Vector magnitude (norm)
- Dot product
- Orthogonality check
- Vector normalization
- Euclidean distance

#### Part D: Vector Spaces and Span
- Linear independence
- Span membership
- Basis for ℝ²
- Matrix rank

#### Part E: Orthogonality
- Orthogonal vector pairs
- Cosine similarity
- Conceptual differences: orthogonal vs orthonormal, basis, vector space

## Dataset

No external dataset required. All data is generated in-code using NumPy and Python dictionaries.

## Implementation Details

### Assignment 1: Python Basics

#### Concepts Covered
- NumPy 3D array creation and shape manipulation
- Advanced slicing with `[::-1]` for reversal
- Dictionary CRUD operations
- Dictionary methods: `.keys()`, `.values()`, `.items()`
- Nested dictionary access
- Lambda functions with `max()`/`min()` for finding extremes
- Aggregation operations (sum, average)

#### Key Implementation Details
- **NumPy arrays** used for efficient 3D array operations
- **Slicing syntax** `arr[::-1]` reverses along an axis
- **Dictionary methods** for different views
- **Lambda functions** for custom sorting/finding

### Assignment 2: Linear Algebra

#### Concepts Covered
- Matrix arithmetic (addition, subtraction, scalar multiplication, transpose)
- Matrix multiplication rules and computation
- Vector operations (magnitude, dot product, normalization)
- Linear algebra concepts (independence, span, basis, rank)
- Orthogonality and cosine similarity

#### Key Implementation Details
- All calculations shown with intermediate steps
- Numerical answers rounded to 2 decimal places where needed
- Conceptual explanations for theoretical questions

## Results

### Assignment 1: Python Basics

#### Question 1: 3D Array Basics
- Array shape: (2, 2, 3) - 2 layers, 2 rows, 3 columns
- First layer access: `arr[0]`
- Element access: `arr[layer, row, column]`

#### Question 2: Reverse Layers
- `arr[::-1]` reverses layer order

#### Question 3: Reverse Rows
- `arr[:, :, ::-1]` reverses elements in each row

#### Question 4: Complete Reverse
- `arr[::-1, ::-1, ::-1]` reverses all three axes

#### Question 5-7: Basic Dictionary Operations
- Created student dictionary with 5 fields
- Updated CGPA, added City
- Used `.keys()`, `.values()`, `.items()`

#### Question 8: Nested Dictionary
- Access nested values: `students['Student2']['Name']`
- Loop through nested structure

#### Question 9: Product Inventory
- Highest price: Laptop (₹50,000)
- Total inventory value: ₹985,000

#### Question 10: Student Marks Analysis
- Highest: Emma (95)
- Lowest: Charlie (78)
- Average: 87.60
- Above 85: Bob (92), David (88), Emma (95)

### Assignment 2: Linear Algebra (Key Results)

| Question | Topic | Answer |
|----------|-------|--------|
| 1 | Scalar/Vector/Matrix/Tensor | a) Scalar, b) Vector, c) Vector, d) Tensor |
| 2 | Matrix Dimensions | a) 1×6, b) 4×1, c) 1×6 |
| 3 | Matrix Arithmetic | A+B=[[6,6],[9,15]], A-B=[[-2,4],[-3,-1]], 2A=[[4,10],[6,14]] |
| 4 | Transpose (2×3) | 3×2 matrix |
| 5 | Transpose (3×2) | 2×3 matrix |
| 6 | Matrix Multiply | [[19,22],[43,50]] |
| 7 | Multiplication Compatibility | a) 2×4 ✓, b) ✗, c) 4×5 ✓, d) 5×2 ✓ |
| 8 | Matrix Multiply | [[17,20],[41,48]] |
| 9 | Commutativity | AB=[[19,22],[43,50]], BA=[[23,34],[31,46]] → Not commutative |
| 10 | Vector Magnitude | 5 |
| 11 | Vector Magnitude | √65 ≈ 8.06 |
| 12 | Dot Product | 21 |
| 13 | Orthogonality | Yes (dot product = 0) |
| 14 | Normalization | [0.6, 0.8]ᵀ |
| 15 | Euclidean Distance | 10 |
| 16 | Linear Independence | Dependent (v₂ = 2v₁) |
| 17 | Span Membership | Yes (a=9, b=-2) |
| 18 | Basis for ℝ² | {[1,0]ᵀ, [0,1]ᵀ} |
| 19 | Matrix Rank | 1 |
| 20 | Matrix Rank | 2 |
| 21 | Orthogonal Pairs | a) Yes, b) Yes, c) No |
| 22 | Cosine Similarity | ≈0.984 (very similar direction) |
| 23 | Concept Definitions | See answer sheet |

## How to Run

```bash
# Navigate to notebook directory
cd code/notebooks

# Start Jupyter for Python Basics
jupyter notebook assignment_2_basics_solution.ipynb
```

### Dependencies
- Python 3.10+
- numpy

## Key Takeaways

### Python Basics
- **3D array indexing**: `array[layer, row, column]` for intuitive access
- **Slicing power**: `[::-1]` reverses any axis; combine for multi-axis reversal
- **Dictionary methods**: `.keys()`, `.values()`, `.items()` for different views
- **Nested access**: Chain keys `dict['outer']['inner']`
- **Lambda with max/min**: `max(dict.items(), key=lambda x: x[1])` finds max by value
- **Aggregation**: `sum()` and `len()` for averages

### Linear Algebra
- **Matrix multiplication**: Not commutative (AB ≠ BA in general)
- **Compatibility**: Inner dimensions must match (m×n)(n×p) → m×p
- **Orthogonality**: Dot product = 0 means perpendicular
- **Normalization**: Divide by magnitude to get unit vector
- **Linear dependence**: One vector is scalar multiple of another
- **Rank**: Number of linearly independent rows/columns
- **Cosine similarity**: Measures directional similarity (1 = same, 0 = orthogonal, -1 = opposite)