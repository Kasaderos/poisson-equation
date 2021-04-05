import numpy as np

def getC(N, a, b, c):
    C = np.array([[0] * i + [a, b, c] + [0] * (N-3-i+2) for i in range(N)])
    return C[:,1:N+1] 

def getA(C, I):
    I = np.zeros(C.shape) 
    A = np.array([[0] * i + [a, b, c] + [0] * (N-3-i+2) for i in range(N)])
    return C[:,1:N+1] 
    

N = 10 
h = h*h
C = getC()
I = 1/h2 * (-1) * np.eye(N-1)

A = 

def simple_iter(u, eps):
    while True:

        norm_old = norm(u)

        for i in range(1, N):
            for j in range(1, N):
                u[i, j] = 0

