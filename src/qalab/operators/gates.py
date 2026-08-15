import numpy as np 

#x_gate
def x_gate():
    """
    Generates the Pauli-X quantum gate (Quantum NOT gate).
    
    The Pauli-X gate is a single-qubit operation that flips the state 
    |0> to |1> and vice versa.
    
    Returns:
        numpy.ndarray: A 2x2 complex matrix representing the Pauli-X gate.
    """
    X = np.array([[0,1],
                  [1,0]],
                 dtype=complex)
    return X

#z_gate
def z_gate():  
    """
    Generates the Pauli-Z quantum gate (Phase-flip gate).
    
    The Pauli-Z gate leaves the |0> state unchanged and flips the sign 
    (applies a phase shift of pi) to the |1> state.
    
    Returns:
        numpy.ndarray: A 2x2 complex matrix representing the Pauli-Z gate.
    """
    Z = np.array([[1,0],
                  [0,-1]],
                 dtype=complex)
    return Z

#y_gate
def y_gate():
    """
    Generates the Pauli-Y quantum gate.

    The Pauli-Y gate performs a combined bit-flip and phase-flip operation
    on a single qubit.

    Returns:
        numpy.ndarray: A 2x2 complex matrix representing the Pauli-Y gate.
    """
    Y = np.array([[0,-1j],
                  [1j,0]],
                 dtype=complex)
    return Y

#h_gate
def h_gate():
    """
    Generates the Hadamard quantum gate.

    The Hadamard gate transforms basis states (|0> and |1>) into equal 
    superposition states.

    Returns:
        numpy.ndarray: A 2x2 complex matrix representing the Hadamard gate.
    """
    H = np.array([[1,1],
                  [1,-1]]/np.sqrt(2),
                 dtype=complex)
    return H

#s_gate
def s_gate():
    """
    Generates the Phase (S) quantum gate.

    The S gate applies a phase shift of pi/2 to the |1> state component.
    It is also known as the sqrt(Z) gate.

    Returns:
        numpy.ndarray: A 2x2 complex matrix representing the S gate.
    """
    S = np.array([[1,0],
                  [0,1j]],
                 dtype=complex)
    return S

#t_gate
def t_gate():
    """
    Generates the T (pi/8) quantum gate.

    The T gate applies a phase shift of pi/4 to the |1> state component.
    It is also known as the sqrt(S) gate.

    Returns:
        numpy.ndarray: A 2x2 complex matrix representing the T gate.
    """
    T = np.array([[1,0],
                  [0,np.exp((1j*np.pi)/4)]],
                 dtype=complex)
    return T

def rx_gate(theta : float) -> np.ndarray:
    """
    Generates the Rotation-X (Rx) quantum gate.
    
    Applies a rotation of angle theta around the X-axis on the Bloch sphere.
    
    Args:
        theta (float): The rotation angle in radians.
        
    Returns:
        numpy.ndarray: A 2x2 complex matrix representing the Rx gate.
    """
    RX = np.array([[np.cos(theta/2),-1j * np.sin(theta/2)],
                   [-1j * np.sin(theta/2),np.cos(theta/2)]])
    return RX

def ry_gate(theta : float) -> np.ndarray:
    """
    Generates the Rotation-Y (Ry) quantum gate.
    
    Applies a rotation of angle theta around the Y-axis on the Bloch sphere.
    
    Args:
        theta (float): The rotation angle in radians.
        
    Returns:
        numpy.ndarray: A 2x2 complex matrix representing the Ry gate.
    """
    RY = np.array([[np.cos(theta/2),-np.sin(theta/2)],
                   [np.sin(theta/2), np.cos(theta/2)]])
    return RY

def rz_gate(theta : float) ->np.ndarray:
    """
    Generates the Rotation-Z (Rz) quantum gate.
    
    Applies a rotation of angle theta around the Z-axis on the Bloch sphere.
    
    Args:
        theta (float): The rotation angle in radians.
        
    Returns:
        numpy.ndarray: A 2x2 complex matrix representing the Rz gate.
    """
    RZ = np.array([[np.exp((-1j * theta)/2),0],
                   [0,np.exp((1j * theta)/2)]])
    return RZ

def u_gate(
    theta: float,
    phi: float,
    lam: float,
) -> np.ndarray:
    return np.array([[np.cos(theta/2),           (-np.exp(1j*lam))*np.sin(theta/2)],
              [(np.exp(1j*phi))*np.sin(theta/2), (np.exp(1j *(phi+lam)))*np.cos(theta/2)]])