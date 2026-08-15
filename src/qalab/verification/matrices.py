import numpy as np 

def is_hermitian(matrix : np.ndarray) -> bool:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    conj = np.conj(matrix)
    dagger = conj.T
    return np.allclose(dagger , matrix)

def is_unitary(matrix : np.ndarray) -> bool:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False 
    conj = np.conj(matrix)
    dagger = conj.T
    Unit = dagger @ matrix
    return np.allclose(Unit , np.eye(matrix.shape[0]))