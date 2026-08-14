import numpy as np 
import pytest
from qalab.measurement.projective import sample_computational_basis , z_expectation_from_samples
from qalab.math.linear_algebra import expectation_value
from qalab.operators.gates import z_gate 

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
        
#test_z_expectation_from_samples
@pytest.mark.parametrize("samples , expected" , [
    (np.array([0,0,0,0]), 1),
    (np.array([1,1,1,1]), -1),
    (np.array([1,0,1,0]), 0)
])
def test_z_expectation_from_samples(samples , expected):
    assert np.isclose(z_expectation_from_samples(samples) , expected)
    
def test_z_expectation_from_samples_vs_expectation_value():
    state = np.array([np.sqrt(0.8),np.sqrt(0.2)])
    shots = 5000
    seed = 2
    exact = expectation_value(state , z_gate())
    samples = sample_computational_basis(state , shots , seed)
    estimated = z_expectation_from_samples(samples)
    assert np.isclose(exact , estimated , atol=0.05)