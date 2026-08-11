import numpy as np 
from qalab.verification.matrices import is_hermitian

#is hermitian 
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
    