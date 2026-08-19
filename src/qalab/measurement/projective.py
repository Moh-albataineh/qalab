import numpy as np
from qalab.states.state_vector import probabilities
from qalab.operators.gates import h_gate , s_gate

def sample_computational_basis(state , shots , seed=None):
    if shots <= 0: 
        raise ValueError("shots need to be >= 1 ")
    rng = np.random.default_rng(seed) #RNG not np.random.choice
    # Local RNG allows reproducible sampling with a seed.
    probs = probabilities(state)
    outcomes = np.arange(len(probs))
    return rng.choice(outcomes, shots , p=probs) 

def sample_pauli_basis(
    state : np.ndarray,
    basis : str,
    shots : int,
    seed = None
) :
    b = basis.upper()
    
    if b == "Z":
        pass
    elif b == "X":
        state = h_gate() @ state
    elif b == "Y":
        s_dagger = np.conj(s_gate()).T
        state = h_gate() @ s_dagger @ state 
    else :
        raise ValueError("basis should be 'X', 'Y', or 'Z'")
        
    return sample_computational_basis(state , shots , seed)

from qalab.states.state_vector import computational_basis_state
from qalab.math.linear_algebra import expectation_value , tensor_product
from  qalab.operators.gates import x_gate, y_gate,z_gate , h_gate , cx_gate

def pauli_expectation_from_samples(sampls):
    return np.mean((1-2*sampls))

ket00 = computational_basis_state("00")
HI = tensor_product(h_gate() , np.eye(2))
shots = 2000
state = HI @ ket00
state = cx_gate() @ state
samples = sample_computational_basis(state, shots, seed=1)
print(np.isin(samples, [0, 3]))
    