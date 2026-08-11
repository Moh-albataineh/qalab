# QALab

## What is QALab?

QALab is a learning-oriented quantum computing laboratory that I am building
incrementally while studying the mathematical and programming foundations of
quantum computing.

At its current stage, QALab provides small linear algebra utilities and
verification tools for state vectors and matrices. The project is intended to
grow naturally as new quantum computing concepts are learned and implemented.

## Current capabilities

### Linear algebra
- `inner_product`
- `normalize`
- `tensor_product`
- `expectation_value`

### State verification
- `is_normalized`
- `global_phase_equivalent`

### Matrix verification
- `is_hermitian`
- `is_unitary`

## Installation

Install QALab in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]" 
```

## Running tests

Run the test suite with:

```bash
pytest -q
```