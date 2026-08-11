import numpy as np 

#inner_product
def inner_product(a, b):
    a = np.conj(a)
    x = a * b 
    return np.sum(x)

#normalize
def normalize (vector):
    norm = inner_product(vector,vector)
    norm = np.sqrt(norm)
    if norm == 0 :
        raise ValueError("Cannot normalize the zero vector")
    return vector / norm

#tensor_product
def tensor_product(a, b):
    return np.kron(a,b)

ket0 = np.array([1,0])
ket1 = np.array([0,1])

print(tensor_product(ket1,ket0))
    