import numpy as np 

def x_gate():
    """
    Generates the Pauli-X quantum gate (Quantum NOT gate).
    
    The Pauli-X gate is a single-qubit operation that flips the state 
    |0> to |1> and vice versa. It is represented by a 2x2 unitary matrix.
    
    Returns:
        numpy.ndarray: A 2x2 complex matrix representing the Pauli-X gate.
        
    Example:
        >>> x_gate()
        array([[0.+0.j, 1.+0.j],
               [1.+0.j, 0.+0.j]])
    """
    X = np.array([[0,1],
                  [1,0]],
                 dtype=complex)
    return X