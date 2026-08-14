import numpy as np
from qalab.states.state_vector import probabilities

def sample_computational_basis(state , shots , seed=None):
    """
    Samples the computational basis (0 or 1) for a given single-qubit state.

    This function calculates the probabilities of measuring 0 or 1 from the 
    state vector and then simulates the measurement process by randomly 
    sampling outcomes based on those probabilities.

    Args:
        state (np.ndarray): The quantum state vector (amplitudes).
        shots (int): The number of measurement shots (samples) to take.
        seed (int, optional): A seed value for the random number generator 
                              to ensure reproducible results. Defaults to None.

    Returns:
        np.ndarray: An array of size `shots` containing the sampled outcomes (0s and 1s).
    """
    if shots <= 0:
        raise ValueError("shots need to be >= 1 ")
    rng = np.random.default_rng(seed) 
    # Local RNG allows reproducible sampling with a seed.
    probs = probabilities(state)
    return rng.choice([0,1], shots , p=probs) 

def z_expectation_from_samples(samples: np.ndarray) -> float:
    """
    Estimates the expectation value of the Z observable from measurement samples.

    In the computational basis, a measurement outcome of 0 corresponds to the
    +1 eigenvalue of the Pauli-Z operator, and an outcome of 1 corresponds 
    to the -1 eigenvalue. This function maps the binary samples (0, 1) to the 
    Z eigenvalues (+1, -1) using the transformation (1 - 2 * samples), and 
    then calculates the average (mean) to estimate the expectation value <Z>.

    Args:
        samples (np.ndarray): An array of binary measurement outcomes (0s and 1s).

    Returns:
        float: The estimated expectation value <Z>, ranging from -1.0 to +1.0.

    Example:
        >>> samples = np.array([0, 0, 1, 0, 1])
        >>> z_expectation_from_samples(samples)
        0.2
    """
    return np.mean((1-2*samples))
