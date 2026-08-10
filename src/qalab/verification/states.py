import numpy as np
from qalab.math.linear_algebra import inner_product

def is_normalized(state):
    is_norm = inner_product(state,state)
    return  np.isclose(is_norm , 1.0)
    