import numpy as np
from qalab.math.linear_algebra import inner_product

def is_normalized(state):
    """
    Checks if a quantum state vector is normalized (its norm equals 1).
    
    Args:
        state (numpy.ndarray): The input state vector to check.
        
    Returns:
        bool: True if the state is normalized, False otherwise.
        
    Example:
        >>> import numpy as np
        >>> state = np.array([1, 0])
        >>> is_normalized(state)
        True
    """
    is_norm = inner_product(state,state)
    return  np.isclose(is_norm , 1.0)
    