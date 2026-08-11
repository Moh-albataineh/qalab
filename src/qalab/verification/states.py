import numpy as np
from qalab.math.linear_algebra import inner_product

def is_normalized(state: np.ndarray) -> bool:
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

#global_phase_equivalent
def global_phase_equivalent(a: np.ndarray , b: np.ndarray) -> bool:
    """
    Checks if two quantum states are equivalent up to a global phase.
    
    Args:
        a (numpy.ndarray): The first state vector.
        b (numpy.ndarray): The second state vector.
        
    Returns:
        bool: True if they differ only by a global phase, False otherwise.
        
    Example:
        >>> import numpy as np
        >>> state1 = np.array([1, 0])
        >>> state2 = np.array([-1, 0])
        >>> global_phase_equivalent(state1, state2)
        True
    """
    if a.shape != b.shape :
        return False
    if not is_normalized(a) or not is_normalized(b):
        return False 
    return np.isclose (np.abs(inner_product(a,b)),1.0)
