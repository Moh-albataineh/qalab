import numpy as np
from qalab.math.linear_algebra import inner_product

def is_normalized(state: np.ndarray) -> bool:
    is_norm = inner_product(state,state)
    return  np.isclose(is_norm , 1.0)

#global_phase_equivalent
def global_phase_equivalent(a: np.ndarray , b: np.ndarray) -> bool:

    if a.shape != b.shape :
        return False
    if not is_normalized(a) or not is_normalized(b):
        return False 
    return np.isclose (np.abs(inner_product(a,b)),1.0)
 
