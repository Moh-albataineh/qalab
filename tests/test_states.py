import numpy as np 
from qalab.verification.states import is_normalized

#is_normalized
def test_is_normalized_True():
    state = np.array([1,1]/np.sqrt(2))
    assert is_normalized(state)
    
def test_is_normalized_False():
    state = np.array([1,1])
    assert not is_normalized(state)
    
def test_is_normalized_complex():
    state = np.array([1j,0])
    assert is_normalized(state)