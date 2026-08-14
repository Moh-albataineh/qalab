import numpy as np 
import pytest
from qalab.measurement.projective import sample_computational_basis

#measurement sampling
def test_sample_computational_basis_ket0():
    ket0 = np.array([1,0])
    shots = 100
    samples = sample_computational_basis(ket0  , shots)
    assert np.all(samples == 0)
    
def test_sample_computational_basis_ket1():
    ket1 = np.array([0,1])
    shots = 100
    samples = sample_computational_basis(ket1  , shots)
    assert np.all(samples == 1)
    
def test_sample_computational_basis_seeds():
    state = np.array([1,1]/np.sqrt(2))
    shots = 1000
    seed = 1
    samples = sample_computational_basis(state , shots , seed)
    count_0 = np.sum(samples == 0)
    count_0 = count_0 / shots
    assert np.isclose(count_0 , 0.5 , atol=0.05)
    
def test_sample_computational_basis_raises_error():
    state = np.array([1,0])
    shots = -5 
    with pytest.raises(ValueError):
        sample_computational_basis(state , shots)
    