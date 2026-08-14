import numpy as np 
from qalab.math.linear_algebra import expectation_value
from qalab.operators.gates import x_gate , y_gate , z_gate

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

def bloch_vector(state : np.ndarray) -> np.ndarray:
    """
    Calculates the 3D Bloch vector coordinates for a given single-qubit quantum state.

    The Bloch vector represents the quantum state geometrically on the Bloch sphere,
    where each coordinate (x, y, z) corresponds to the expectation value of the
    respective Pauli operator (X, Y, Z).

    Args:
        state (numpy.ndarray): The quantum state vector of a single qubit.

    Returns:
        numpy.ndarray: A 3-element array [x, y, z] containing the Bloch vector coordinates.

    Example:
        >>> import numpy as np
        >>> state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Plus state |+>
        >>> bloch_vector(state)
        array([1., 0., 0.])
    """
    x = expectation_value(state , x_gate())
    y = expectation_value(state , y_gate())
    z = expectation_value(state , z_gate())
    return np.array([x,y,z])
