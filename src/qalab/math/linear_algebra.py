import numpy as np 

def inner_product(a, b):
    a = np.conj(a)
    x = a * b 
    return np.sum(x)
