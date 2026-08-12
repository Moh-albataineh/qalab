import numpy as np 
import pytest
from qalab.verification.matrices import is_unitary
from qalab.operators.gates import (
    x_gate ,
    z_gate ,
    y_gate ,
    h_gate ,
    s_gate ,
    t_gate ,
    rx_gate,
    ry_gate,
    rz_gate,
    )

#is_unitary
@pytest.mark.parametrize(
    "gate",
    [
        x_gate,
        y_gate,
        z_gate,
        h_gate,
        s_gate,
        t_gate,
    ],
)
def test_single_qubit_gates_are_unitary(gate):
    assert is_unitary(gate())

#x_gate
def test_x_gate():
    ket0 = np.array([1,0])
    assert np.allclose(x_gate() @ ket0, np.array([0,1]))
    
def test_x_gate_ket1():
    ket1 = np.array([0,1])
    assert np.allclose(x_gate() @ ket1, np.array([1,0]))
    
def test_x_gate_superposition():
    state = np.array([1,1]/np.sqrt(2))
    assert np.allclose(x_gate() @ state, state)
   
#z_gate
def test_z_gate_ket0():
    ket0 = np.array([1,0])
    assert np.allclose(z_gate() @ ket0, np.array([1,0]))
    
def test_z_gate_ket1():
    ket1 = np.array([0,1])
    assert np.allclose(z_gate() @ ket1, np.array([0,-1]))
    
def test_z_gate_superposition():
    state = np.array([1,1]/np.sqrt(2))
    assert np.allclose(z_gate() @ state, np.array([1,-1]/np.sqrt(2)))

#y_gate
def test_y_gate_ket0():
    ket0 = np.array([1,0])
    assert np.allclose(y_gate() @ ket0, np.array([0,1j]))

def test_y_gate_ket1():
    ket1 = np.array([0,1])
    assert np.allclose(y_gate() @ ket1, np.array([-1j,0]))
    
def test_y_gate_superposition():
    state = np.array([1,1]/np.sqrt(2))
    assert np.allclose(y_gate() @ state, np.array([-1j,1j]/np.sqrt(2)))
    
#h_gate
def test_h_gate_ket0():
    ket0 = np.array([1,0])
    assert np.allclose(h_gate() @ ket0, np.array([1,1]/np.sqrt(2)))
    
def test_h_gate_ket1():
    ket1 = np.array([0,1])
    assert np.allclose(h_gate() @ ket1, np.array([1,-1]/np.sqrt(2)))
    
def test_h_gate_superposition():
    state = np.array([1,1]/np.sqrt(2))
    assert np.allclose(h_gate() @ state, np.array([1,0]))
    
#s_gate
def test_s_gate_ket0():
    ket0 = np.array([1,0])
    assert np.allclose(s_gate() @ ket0, np.array([1,0]))
    
def test_s_gate_ket1():
    ket1 = np.array([0,1])
    assert np.allclose(s_gate() @ ket1, np.array([0,1j]))
    
def test_s_gate_superposition():
    state = np.array([1,1]/np.sqrt(2))
    assert np.allclose(s_gate() @ state, np.array([1,1j]/np.sqrt(2)))
    
#t_gate
def test_t_gate_ket0():
    ket0 = np.array([1,0])
    assert np.allclose(t_gate() @ ket0, np.array([1,0]))
    
def test_t_gate_ket1():
    ket1 = np.array([0,1])
    assert np.allclose(t_gate() @ ket1, np.array([0,np.exp((1j * np.pi)/4)]))
    
def test_t_gate_superposition():
    state = np.array([1,1]/np.sqrt(2))
    assert np.allclose(t_gate() @ state, np.array([1,np.exp((1j * np.pi)/4)]/np.sqrt(2)))

#rx_gate 
def test_rx_gate():     
    assert np.allclose(rx_gate(np.pi),-1j * x_gate())

#ry_gate 
def test_ry_gate():     
    assert np.allclose(ry_gate(np.pi),-1j * y_gate())
    
#rz_gate 
def test_rz_gate():     
    assert np.allclose(rz_gate(np.pi),-1j * z_gate())

#rotation_zero
@pytest.mark.parametrize(
    "rotation",
    [rx_gate, ry_gate, rz_gate],
)
def test_rotation_zero_is_identity(rotation):
    assert np.allclose(rotation(0), np.eye(2))
    
#rotations_are_unitary
@pytest.mark.parametrize("rotation", [rx_gate, ry_gate, rz_gate])
@pytest.mark.parametrize("theta", [0, np.pi / 3, np.pi, 2 * np.pi])
def test_rotations_are_unitary(rotation, theta):
    assert is_unitary(rotation(theta))