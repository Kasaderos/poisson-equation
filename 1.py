import numpy as np
import sys
import time as t

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

def u0(x, y):
    return 0.0

def phi1(x):
    return 0.0

def phi2(x):
    return x + x*x 

def phi3(y):
    return 0.0

def phi4(y):
    return y + y*y

def ua(x, y):
    return x*x*y + y*y*x 

T = 1.0
L = 1.0
N = 32 
N1 = N+1
h = L/N
h2 = h*h
x = np.linspace(0.0, L, N1)
y = np.linspace(0.0, L, N1)

m = 1024
tau = T / m

U0 = np.zeros((N1, N1))
U_old = np.zeros((N1, N1))
U_new = np.zeros((N1, N1))

U0 = np.array([[ua(xi, yi) for yi in y] for xi in x])

U_old = U0

A = np.zeros((N1, N1))
B = np.zeros((N1, N1))
C = np.zeros((N1, N1))
D = np.zeros((N1, N1))
for i in range(N1):
    for j in range(N1):
        A[i, j] = -1/(2*h2)
        B[i, j] = 1/tau + 1/h2
        C[i, j] = -1/(2*h2)
            
alpha = np.zeros(N1)
beta = np.zeros(N1)

start = t.time()

for k in range(m):
    for i in range(N1):
        U_old[0, i] = phi3(y[i])
        U_new[0, i] = phi3(y[i])
        U_old[N, i] = phi4(y[i])
        U_new[N, i] = phi4(y[i])

    for i in range(1, N):
        for j in range(1, N):
            D[i, j] = U_old[i, j]/tau + (U_old[i, j-1] - 2*U_old[i, j] + U_old[i, j+1]) / (2*h2)

    for j in range(1, N):
        alpha[1] = 0
        beta[1] = phi3(x[j])
        for i in range(1, N):
            alpha[i+1] = -C[i, j] / (B[i, j] + A[i, j] * alpha[i])
            beta[i+1] = (D[i, j] - A[i, j]*beta[i])/(B[i, j] + A[i, j]*alpha[i])
        U_old[N, j] = phi4(x[j])
        for i in range(N-1, 0, -1):
            U_old[i, j] = alpha[i+1] * U_old[i+1, j] + beta[i+1]
    
    for i in range(N1):
        U_old[0, i] = phi1(x[i])
        U_new[0, i] = phi1(x[i])
        U_old[N, i] = phi2(x[i])
        U_new[N, i] = phi2(x[i])


    for i in range(1, N):
        for j in range(1, N):
            D[i, j] = U_old[i, j]/tau + (U_old[i, j+1] - 2*U_old[i, j] + U_old[i, j-1]) / (2*h2)

    for i in range(1, N):
        alpha[1] = 0
        beta[1] = 0 
        for j in range(1, N):
            alpha[j+1] = -C[i, j] / (B[i, j] + A[i, j] * alpha[i])
            beta[j+1] = (D[i, j] - A[i, j]*beta[j])/(B[i, j] + A[i, j]*alpha[j])
        U_new[i, N] = phi2(y[i])
        for j in range(N-1, 0, -1):
            U_new[i, j] = alpha[j+1] * U_new[i, j+1] + beta[j+1]

    U_old[:, :] = U_new[:, :]
    
end = t.time()
print('time = ', end-start, sep='')

Ua = np.array([[ua(xi, yi) for yi in y] for xi in x])

av_err = np.max(np.abs(Ua-U_new))
print(f"|Ua-U| = {av_err}")

mx = U_new.max()
mn = U_new.min()
X, Y = np.meshgrid(x, x)
fig = plt.figure(figsize=(12,5.5))
cmap = mpl.cm.get_cmap('RdBu_r')
ax =  fig.add_subplot(1,2,1)
c = ax.pcolor(X, Y, U_new, vmin=mn, vmax=mx, cmap=cmap)
ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)

ax =  fig.add_subplot(1,2,2, projection='3d')
p = ax.plot_surface(X, Y, U_new, vmin=mn, vmax=mx, rstride=3, cstride=3, linewidth=0, cmap=cmap)
ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)

cb = plt.colorbar(p, ax=ax, shrink=0.75)
cb.set_label(r"$u(x, y)$", fontsize=14)

fig1, ax1 = plt.subplots()
cs = plt.imshow(U_new, cmap='inferno')
fig1.colorbar(cs)
plt.show()

