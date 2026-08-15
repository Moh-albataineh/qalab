# QALab

QALab is a learning-oriented quantum computing laboratory built incrementally while studying the mathematical, physical, and programming foundations of quantum computing.

The project currently focuses on mathematical verification and single-qubit quantum information. New capabilities are added as the underlying concepts are studied, implemented, tested, and explored experimentally.

## Current capabilities

### Linear algebra

* `inner_product`
* `normalize`
* `tensor_product`
* `expectation_value`

### State-vector tools

* Computational-basis probabilities from the Born rule
* Bloch-vector calculation
* Single-qubit state analysis

### Single-qubit operators

* Pauli gates: X, Y, Z
* Hadamard gate
* Phase gates: S, T
* Rotation gates: Rx, Ry, Rz
* General parameterized single-qubit U gate

### Measurement

* Computational-basis sampling
* X, Y, and Z basis measurements through basis changes
* Finite-shot Pauli expectation estimation
* Reproducible sampling with configurable random seeds

### Verification

* State normalization
* Global-phase equivalence
* Hermitian matrix checks
* Unitary matrix checks

## Example

```python
import numpy as np

from qalab.states.state_vector import probabilities, bloch_vector
from qalab.operators.gates import h_gate

ket0 = np.array([1, 0], dtype=complex)

state = h_gate() @ ket0

print(probabilities(state))
print(bloch_vector(state))
```

The Hadamard gate transforms `|0>` into `|+>`, giving computational-basis probabilities close to:

```text
[0.5, 0.5]
```

and a Bloch vector close to:

```text
[1, 0, 0]
```

## Experiments

Current experiments include finite-shot single-qubit measurement convergence, comparing sampled Bloch-vector estimates with their exact theoretical values as the number of measurement shots increases.

Experiments live under:

```text
experiments/
```

They are kept separate from the test suite:

* `tests/` checks software correctness.
* `experiments/` investigates quantitative behavior.

## Installation

Install QALab in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Running tests

Run the full test suite with:

```bash
pytest -q
```

## Project structure

```text
qalab/
├── src/
│   └── qalab/
│       ├── math/
│       ├── states/
│       ├── operators/
│       ├── measurement/
│       └── verification/
├── tests/
├── experiments/
├── README.md
└── pyproject.toml
```

QALab will continue to grow incrementally as new quantum-computing concepts are studied and implemented.
