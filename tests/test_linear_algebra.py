import pytest
import numpy as np 
from qalab.math.linear_algebra import inner_product , normalize , tensor_product

#inner_product
def test_inner_product():
    ket0 = np.array([1,0])
    assert inner_product(ket0,ket0) == 1
    
def test_inner_product_orthogonal():
    ket0 = np.array([1,0])
    ket1 = np.array([0,1])
    assert inner_product(ket0,ket1) == 0
    
def test_inner_product_complex():
    a = np.array([1j,0])
    assert inner_product(a,a) == 1
    
#normalize
def test_normalize():
    ket = np.array([3,4]) 
    norm_ket = normalize(ket)
    assert np.allclose(norm_ket , np.array([0.6,0.8]))

def test_normalize_produces_unit_norm():
    ket = np.array([3,4])    
    norm_ket = normalize(ket)
    assert np.isclose(inner_product(norm_ket,norm_ket),1.0)
    
def test_normalize_complex():
    ket = np.array([1j,1])    
    norm_ket = normalize(ket)
    assert np.isclose(inner_product(norm_ket,norm_ket),1.0)    
    
def test_normalize_zero_vector_raises_error():  
      ket = np.array([0,0])
      with pytest.raises(ValueError):
          normalize(ket)
          
#tensor_product
def test_tensor_product():
    ket0 = np.array([1,0])
    ket1 = np.array([0,1])
    assert np.allclose(tensor_product(ket1,ket0),np.array([0,0,1,0]))
   
def test_tensor_product_comlpex():
    ketA = np.array([1j,0])
    ketB = np.array([0,1])
    assert np.allclose(tensor_product(ketA,ketB),np.array([0,1j,0,0])) 