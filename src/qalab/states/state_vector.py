import numpy as np 

def probabilities(state : np.ndarray) -> np.ndarray:
    return np.abs(state) ** 2     