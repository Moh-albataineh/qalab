import numpy as np 
from qalab.states.state_vector import probabilities

#probabilities
def test_probabilities_basis_state():
    state = np.array([1,0])
    probs = probabilities(state)
    assert np.allclose(probs , np.array([1,0]))
    
def test_probabilities_complex_state():
    state = np.array([1, 1j]) / np.sqrt(2)
    probs = probabilities(state)
    assert np.allclose(probs , np.array([0.5,0.5]))
    
def test_probabilities_property():
    state = np.array([np.sqrt(0.8), np.sqrt(0.2)])
    probs = probabilities(state)
    assert np.isclose(np.sum(probs) , 1.0)