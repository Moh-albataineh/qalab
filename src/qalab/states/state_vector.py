import numpy as np 

def probabilities(state : np.ndarray) -> np.ndarray:
    """
    Calculates the measurement probabilities for a given quantum state.
    
    According to Born's rule, the probability of measuring a basis state 
    is the absolute square of its probability amplitude.
    
    Args:
        state (numpy.ndarray): The quantum state vector (amplitudes).
        
    Returns:
        numpy.ndarray: An array of probabilities corresponding to each basis state.
        
    Example:
        >>> import numpy as np
        >>> state = np.array([1/np.sqrt(2), 1/np.sqrt(2)]) # Plus state |+>
        >>> probabilities(state)
        array([0.5, 0.5])
    """
    return np.abs(state) ** 2     