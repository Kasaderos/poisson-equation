import numpy as np
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

def norm(a):
    return np.max(np.abs(a))

def mu(x, y):
    return x*x*x*y + x*y*y*y

def f(x, y):
    return -12 * x * y

N = 64 
N1 = N+1
L = 1.0 
h = L/N
h2 = h*h
x = np.linspace(0.0, L, N1)
y = np.linspace(0.0, L, N1)

#ua = np.array([[mu(x[i], y[j]) for j in range(N1)] for i in range(N1)])

ua = np.zeros((N1,N1))
for i in range(N1):
    for j in range(N1):
        ua[i, j] = mu(x[i], y[j])

u = np.zeros((N1, N1))

u0 = u.copy()
u_old = u.copy()

delta = 8/(h*h) * np.sin(np.pi*h/2)**2
delta2 = 8/(h*h) * np.cos(np.pi*h/2)**2
xi = delta/delta2

roH = (1-xi)/(1+xi)
print('roH = ', roH)

err = norm(u-ua)
eps = 1e-5
m = round(np.log(2/eps)/(2*np.sqrt(xi))) + 4
if m & 1:
    m += 1
print('optimal m = ', m)
print('h^2 = ', h*h)
print('eps = ', eps)

for i in range(N1):
    u[i, 0] = mu(x[i], 0.0)
    u[i, N] = mu(x[i], 1.0)
    u[0, i] = mu(0.0, y[i]) 
    u[N, i] = mu(1.0, y[i])

def get_teta(n):
    n = pow_two(n)

    teta = { 1:[1] }

    i = 2
    while i <= n:
        new_teta = []
        j = 1
        for j in range(1, i+1):
            if j & 1:   # odd
                new_teta.append(teta[ i//2 ][ (j+1)//2 -1 ])
            else:       # even
                new_teta.append(2*i - new_teta[j-2])
        teta[i] = new_teta
        i *= 2
    return teta[n]

def pow_two(n):
    i = 1
    while i < n:
        i *= 2
    return i

teta = get_teta(m+1)
teta = np.array([0] + teta) 
tau = np.zeros(len(teta))
for k in range(1, len(teta)):
    tau[k] = 2 /  (delta2 + delta + (delta2-delta)*np.cos(teta[k] * np.pi / (2 * len(teta)))) 

for k in range(1, m+1):
    for i in range(1, N):
        for j in range(1, N):
            u[i, j] = u_old[i, j] + tau[k]*((u_old[i+1, j] - u_old[i, j]) / h2 - (u_old[i, j] - u_old[i-1, j]) / h2 + 
                    (u_old[i, j+1] - u_old[i, j]) / h2 - (u_old[i, j] - u_old[i, j-1]) / h2 + f(x[i], y[j])) 
    
    u_old = u.copy()
    err = norm(u - ua)
    print(err, err / norm(u0-ua), norm(u-u))
    #time.sleep(0.01)    
print('max|u-u*| =', err)

# mx = u.max()
# mn = u.min()
# X, Y = np.meshgrid(x, x)
# fig = plt.figure(figsize=(12,5.5))
# cmap = mpl.cm.get_cmap('RdBu_r')
# ax =  fig.add_subplot(1,2,1)
# c = ax.pcolor(X, Y, u, vmin=mn, vmax=mx, cmap=cmap)
# ax.set_xlabel(r"$x_1$", fontsize=14)
# ax.set_ylabel(r"$x_2$", fontsize=14)

# ax =  fig.add_subplot(1,2,2, projection='3d')
# p = ax.plot_surface(X, Y, u, vmin=mn, vmax=mx, rstride=3, cstride=3, linewidth=0, cmap=cmap)
# ax.set_xlabel(r"$x_1$", fontsize=14)
# ax.set_ylabel(r"$x_2$", fontsize=14)

# cb = plt.colorbar(p, ax=ax, shrink=0.75)
# cb.set_label(r"$u(x, y)$", fontsize=14)

# fig1, ax1 = plt.subplots()
# cs = plt.imshow(u, cmap='inferno')
# fig1.colorbar(cs)
# plt.show()
