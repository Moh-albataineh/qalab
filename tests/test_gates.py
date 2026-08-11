import numpy as np 
from qalab.operators.gates import x_gate
from qalab.verification.matrices import is_unitary

#x_gate
def test_x_gate():
    ket0 = np.array([1,0])
    assert np.allclose(x_gate() @ ket0, np.array([0,1]))
    
def test_x_gate_superposition():
    state = np.array([1,1]/np.sqrt(2))
    assert np.allclose(x_gate() @ state, state )

def test_x_gate_is_unitary():
    assert is_unitary(x_gate())