import numpy as np 

#x_gate
def x_gate():
    X = np.array([[0,1],
                  [1,0]],
                 dtype=complex)
    return X

#z_gate
def z_gate():  
    Z = np.array([[1,0],
                  [0,-1]],
                 dtype=complex)
    return Z

#y_gate
def y_gate():
    Y = np.array([[0,-1j],
                  [1j,0]],
                 dtype=complex)
    return Y

#h_gate
def h_gate():
    H = np.array([[1,1],
                  [1,-1]]/np.sqrt(2),
                 dtype=complex)
    return H

#s_gate
def s_gate():
    S = np.array([[1,0],
                  [0,1j]],
                 dtype=complex)
    return S

#t_gate
def t_gate():
    T = np.array([[1,0],
                  [0,np.exp((1j*np.pi)/4)]],
                 dtype=complex)
    return T

#rx_gate
def rx_gate(theta : float) -> np.ndarray:
    RX = np.array([[np.cos(theta/2),-1j * np.sin(theta/2)],
                   [-1j * np.sin(theta/2),np.cos(theta/2)]])
    return RX

#ry_gate
def ry_gate(theta : float) -> np.ndarray:
    RY = np.array([[np.cos(theta/2),-np.sin(theta/2)],
                   [np.sin(theta/2), np.cos(theta/2)]])
    return RY

#rz_gate
def rz_gate(theta : float) ->np.ndarray:
    RZ = np.array([[np.exp((-1j * theta)/2),0],
                   [0,np.exp((1j * theta)/2)]])
    return RZ

#cx_gate
def cx_gate() ->np.ndarray:
    CX = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,0,1],
                   [0,0,1,0]],
                  dtype=complex)
    return CX

#u_gate
def u_gate(
    theta: float,
    phi: float,
    lam: float,
) -> np.ndarray:
    return np.array([[np.cos(theta/2),           (-np.exp(1j*lam))*np.sin(theta/2)],
              [(np.exp(1j*phi))*np.sin(theta/2), (np.exp(1j *(phi+lam)))*np.cos(theta/2)]])