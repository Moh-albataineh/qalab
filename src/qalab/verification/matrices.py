import numpy as np 

def is_hermitian(matrix):
    """
    Checks if a matrix is Hermitian (equal to its conjugate transpose).
    
    Args:
        matrix (numpy.ndarray): The input matrix to check.
        
    Returns:
        bool: True if the matrix is Hermitian, False otherwise.
        
    Example:
        >>> import numpy as np
        >>> pauli_y = np.array([[0, -1j], 
        ...                     [1j, 0]])
        >>> is_hermitian(pauli_y)
        True
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    conj = np.conj(matrix)
    dagger = conj.T
    return np.allclose(dagger , matrix)

def is_unitary(matrix):
    """
    Checks if a matrix is unitary (U * U^dagger = Identity matrix).
    
    Args:
        matrix (numpy.ndarray): The input matrix to check.
        
    Returns:
        bool: True if the matrix is unitary, False otherwise.
        
    Example:
        >>> import numpy as np
        >>> pauli_x = np.array([[0, 1], 
        ...                     [1, 0]])
        >>> is_unitary(pauli_x)
        True
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False 
    conj = np.conj(matrix)
    dagger = conj.T
    Unit = dagger @ matrix
    return np.allclose(Unit , np.eye(matrix.shape[0]))