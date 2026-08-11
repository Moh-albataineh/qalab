import numpy as np 
from qalab.verification.states import  (
    is_normalized,
    global_phase_equivalent
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
        