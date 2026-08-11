import numpy as np 
from qalab.verification.matrices import (
    is_hermitian,
    is_unitary
    )

#is_hermitian 
def test_is_hermitian_True():
    matrix = np.array([[1,0],
                       [0,-1]])
    assert is_hermitian(matrix)
    
def test_is_hermitian_False():
    matrix = np.array([[0,1],
                       [0,0]])
    assert not is_hermitian(matrix)
    
def test_is_hermitian_complex():
    matrix = np.array([[0,-1j],
                       [1j,0]])
    assert is_hermitian(matrix)
    
def test_is_hermitian_not_square():
    matrix = np.array([[1,2,3],
                       [4,5,6]])
    assert not is_hermitian(matrix)

#is_unitary
def test_is_unitary_True():
    matrix = np.array([[1,1],[1,-1]]/np.sqrt(2))
    assert is_unitary(matrix)

def test_is_unitary_False():
    matrix = np.array([[1,0],[0,2]])
    assert not is_unitary(matrix)
    
def test_is_unitary_complex():
    matrix = np.array([[0,-1j],
                       [1j,0]])
    assert is_unitary(matrix)
    
def test_is_unitary_not_square():
    matrix = np.array([[1,2,3],
                       [4,5,6]])
    assert not is_unitary(matrix)

def test_is_unitary_not_2x2():
    matrix = np.array([[0, 1, 0], 
                       [0, 0, 1], 
                       [1, 0, 0]])
    assert is_unitary(matrix)
    
    
    