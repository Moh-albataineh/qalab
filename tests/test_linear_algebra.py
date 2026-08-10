import numpy as np 
from qalab.math.linear_algebra import inner_product

def test_inner_product():
    ket0 = np.array([1,0])
    assert inner_product(ket0,ket0) == 1