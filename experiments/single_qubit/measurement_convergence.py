import numpy as np
from qalab.states.state_vector import bloch_vector
from qalab.measurement.projective import sample_pauli_basis, pauli_expectation_from_samples
theta = 1.1
phi = 0.7

state = np.array([
    np.cos(theta / 2),
    np.exp(1j * phi) * np.sin(theta / 2),
])

exact = np.real(bloch_vector(state))

shot_counts = [10, 50, 100, 500, 1000, 5000]

for shots in shot_counts:
    x = pauli_expectation_from_samples(sample_pauli_basis(state, "X", shots, 1))
    y = pauli_expectation_from_samples(sample_pauli_basis(state, "Y", shots, 1))
    z = pauli_expectation_from_samples(sample_pauli_basis(state, "Z", shots, 1))
    estimated = np.array([x,y,z])
    exact = np.real(bloch_vector(state))
    result = np.linalg.norm(estimated - exact)
    print(result)