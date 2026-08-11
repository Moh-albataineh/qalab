import numpy as np 

#inner_product
def inner_product(a, b):
    """
    Calculates the inner (dot) product of two complex vectors.
    
    Args:
        a (numpy.ndarray): The first input vector.
        b (numpy.ndarray): The second input vector.
        
    Returns:
        complex: The calculated inner product (scalar value).
        
    Example:
        >>> import numpy as np
        >>> vec1 = np.array([1, 0])
        >>> vec2 = np.array([0, 1])
        >>> inner_product(vec1, vec2)
        0
        
        >>> vec3 = np.array([1j, 0])
        >>> inner_product(vec3, vec3)
        1.0
    """
    a = np.conj(a)
    x = a * b 
    return np.sum(x)

#normalize
def normalize (vector):
    """
    Normalizes a vector to have a length (norm) of 1.
    
    Args:
        vector (numpy.ndarray): The input vector to be normalized.
        
    Returns:
        numpy.ndarray: The normalized unit vector.
        
    Raises:
        ValueError: If the input is a zero vector.
        
    Example:
        >>> import numpy as np
        >>> vec = np.array([3, 4])
        >>> normalize(vec)
        array([0.6, 0.8])
    """
    norm = inner_product(vector,vector)
    norm = np.sqrt(norm)
    if norm == 0 :
        raise ValueError("Cannot normalize the zero vector")
    return vector / norm

#tensor_product
def tensor_product(a, b):
    """
    Computes the tensor (Kronecker) product of two arrays.
    
    Args:
        a (numpy.ndarray): The first input array or vector.
        b (numpy.ndarray): The second input array or vector.
        
    Returns:
        numpy.ndarray: The resulting tensor product array.
        
    Example:
        >>> import numpy as np
        >>> state_0 = np.array([1, 0])
        >>> tensor_product(state_0, state_0)
        array([1, 0, 0, 0])
    """
    return np.kron(a,b)