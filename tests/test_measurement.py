import numpy as np 
import pytest
from qalab.verification.states import is_product_state
from qalab.states.state_vector import computational_basis_state
from qalab.math.linear_algebra import expectation_value , tensor_product
from  qalab.operators.gates import x_gate, y_gate,z_gate , h_gate , cx_gate
from qalab.measurement.projective import (
    sample_computational_basis,
    pauli_expectation_from_samples,
    sample_pauli_basis
    )

#sample_computational_basis
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
    assert np.isclose(count_0, 0.5, atol=0.05)
    
def test_sample_computational_basis_raises_error():
    state = np.array([1,0])
    shots = -5 
    with pytest.raises(ValueError):
        sample_computational_basis(state, shots)
        
def test_sample_computational_basis_Ball():
    ket00 = computational_basis_state("00")
    HI = tensor_product(h_gate(), np.eye(2))
    shots = 2000
    state = HI @ ket00
    Ball = cx_gate() @ state
    samples = sample_computational_basis(Ball, shots, seed=1)
    count_00 = np.sum(samples == 0)
    count_11 = np.sum(samples == 3)
    assert np.all(np.isin(samples, [0, 3]))
    assert np.isclose(count_00/shots, 0.5 , atol=0.05)
    assert np.isclose(count_11/shots, 0.5 , atol=0.05)
    assert not is_product_state(Ball)
    
def test_sample_computational_basis_Ball_expectation():
    Ball = np.array([1,0,0,1]/np.sqrt(2))
    I = np.eye(2)
    ZI = tensor_product(z_gate(), I)
    IZ = tensor_product(I, z_gate())
    ZZ = tensor_product(z_gate(), z_gate())
    XX = tensor_product(x_gate(), x_gate())
    YY = tensor_product(y_gate(), y_gate())
    assert np.isclose(expectation_value(Ball, ZI), 0)
    assert np.isclose(expectation_value(Ball, IZ), 0)
    assert np.isclose(expectation_value(Ball, ZZ), 1)
    assert np.isclose(expectation_value(Ball, XX), 1)
    assert np.isclose(expectation_value(Ball, YY), -1)
        
#sample_pauli_basis
@pytest.mark.parametrize("state , basis , expected" , [
    (np.array([1,0]), "Z", 0),
    (np.array([0,1]), "Z", 1),
    (np.array([1,1]/np.sqrt(2)), "X", 0),
    (np.array([1,-1]/np.sqrt(2)), "X", 1),
    (np.array([1,1j]/np.sqrt(2)), "Y", 0),
    (np.array([1,-1j]/np.sqrt(2)), "Y", 1)
])
def test_sample_pauli_basis(state , basis , expected):
    sample = sample_pauli_basis(state, basis, 1)
    assert np.isclose(sample , expected)

#pauli_expectation_from_samples
@pytest.mark.parametrize("basis, operator", [
    ("X", x_gate()),
    ("Y", y_gate()),
    ("Z", z_gate())
])
def test_pauli_expectation_from_samples(basis , operator): 
    theta = 1.1
    phi = 0.7
    state = np.array([
       np.cos(theta / 2),
       np.exp(1j * phi) * np.sin(theta / 2),
    ])
    exact = expectation_value(state , operator)
    samples = sample_pauli_basis(
                state,
                basis,
                shots=5000,
                seed=2
                )
    expect = pauli_expectation_from_samples(samples)
    assert np.isclose(expect, exact, atol=0.05)
