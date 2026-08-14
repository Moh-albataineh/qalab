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

