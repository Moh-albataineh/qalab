import numpy as np 
import pytest
from qalab.verification.states import is_normalized
from qalab.states.state_vector import probabilities , bloch_vector

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
    
#bloch_vector  
@pytest.mark.parametrize("state , expected" , [
    (np.array([1,0]),np.array([0,0,1])),
    (np.array([0,1]),np.array([0,0,-1])),
    (np.array([1,1]/np.sqrt(2)),np.array([1,0,0])),
    (np.array([1,-1]/np.sqrt(2)),np.array([-1,0,0])),
    (np.array([1,1j]/np.sqrt(2)),np.array([0,1,0])),
    (np.array([1,-1j]/np.sqrt(2)),np.array([0,-1,0]))
])
def test_bloch_vector(state , expected):
    assert np.allclose(bloch_vector(state),expected)
    
def test_bloch_vector_normalized():
    theta = 0.7
    phi = 1.2
    state = np.array([
       np.cos(theta / 2),
       np.exp(1j * phi) * np.sin(theta / 2)
    ])
    assert np.isclose(np.linalg.norm(bloch_vector(state)) , 1.0)
    
