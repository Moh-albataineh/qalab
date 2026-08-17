import numpy as np 
import pytest
from qalab.states.state_vector import computational_basis_state
from qalab.verification.states import  (
    is_normalized,
    global_phase_equivalent,
    is_product_state
    )

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
    
#global_phase_equivalent
def test_global_phase_equivalent_True():
    stateA = np.array([1,0])
    stateB = np.array([-1,0])
    assert global_phase_equivalent(stateA,stateB)
    
def test_global_phase_equivalent_False():
    stateA = np.array([1,0])
    stateB = np.array([0,1])
    assert not global_phase_equivalent(stateA,stateB)
        
def test_global_phase_equivalent_complex():
    stateA = np.array([1,0])
    stateB = np.array([1j,0])
    assert global_phase_equivalent(stateA,stateB)
    
def test_global_phase_equivalent_shape():
    stateA = np.array([1,0])
    stateB = np.array([-1,0,0,0])
    assert not global_phase_equivalent(stateA,stateB)
    
def test_global_phase_equivalent_normalized():
    stateA = np.array([1,0])
    stateB = np.array([2,0])
    assert not global_phase_equivalent(stateA,stateB)
    
#is_product_state
def test_is_product_state_basis():
    state = computational_basis_state("01")
    assert is_product_state(state)
    
def test_is_product_state_superposition():
    state = np.array([1,0,1,0]/np.sqrt(2))
    assert is_product_state(state)
    
def test_is_product_state_Bell():
    state = np.array([1,0,0,1]/np.sqrt(2))
    assert not is_product_state(state)
    
def test_is_product_state_ValueError():
    state = np.array([1,0])
    with pytest.raises(ValueError):
        is_product_state(state)