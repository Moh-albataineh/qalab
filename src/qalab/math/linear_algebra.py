import numpy as np 

def inner_product(a, b):
    a = np.conj(a)
    x = a * b 
    return np.sum(x)

def normalize (vector):
    norm = inner_product(vector,vector)
    norm = np.sqrt(norm)
    if norm == 0 :
        raise ValueError("Cannot normalize the zero vector")
    return vector / norm