import numpy as np 
from qalab.operators.gates import x_gate , z_gate
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
    
#z_gate
def test_z_gate_ket0():
    ket0 = np.array([1,0])
    assert np.allclose(z_gate() @ ket0, np.array([1,0]))
    
def test_z_gate_ket1():
    ket1 = np.array([0,1])
    assert np.allclose(z_gate() @ ket1, np.array([0,-1]))
    
def test_z_gate_superposition():
    state = np.array([1,1]/np.sqrt(2))
    assert np.allclose(z_gate() @ state, np.array([1,-1]/np.sqrt(2)) )

def test_z_gate_is_unitary():
    assert is_unitary(z_gate())
