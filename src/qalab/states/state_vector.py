import numpy as np 
from qalab.math.linear_algebra import expectation_value
from qalab.operators.gates import x_gate , y_gate , z_gate

def probabilities(state : np.ndarray) -> np.ndarray:

    return np.abs(state) ** 2     

def bloch_vector(state : np.ndarray) -> np.ndarray:

    x = expectation_value(state , x_gate())
    y = expectation_value(state , y_gate())
    z = expectation_value(state , z_gate())
    return np.array([x,y,z])

def computational_basis_state(bits : str) -> np.ndarray:
    if bits == "":
        raise ValueError("bits cannot be empty")
    elif any (char not in ("1","0") for char in bits):
        raise ValueError("bits can only contain 0 or 1.")
    qubit_num = len(bits)
    x = int(bits,2)
    array_len = np.zeros(2**qubit_num , dtype=complex) 
    array_len[x] = 1
    return array_len



