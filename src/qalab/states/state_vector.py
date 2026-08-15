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
