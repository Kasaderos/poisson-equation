import numpy as np
import math
import sys
import scipy.linalg as sl
import time as t
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.axes3d import Axes3D


def f(x, y):
    return 5*x + 2*x*y

def left(y):
    return 0.0

def right(y):
    return y+y*y

def bottom(x):
    return 0.0

def top(x):
    return x*x+x

def sweep(m, C, F):
    Alfa = [np.zeros((m,m))] * m
    beta = np.zeros((m, m))

    Alfa[0] = sl.inv(D)
    beta[0] = np.dot(Alfa[0], F[0])

    for i in range(m-1):
        tmp = sl.inv(D-Alfa[i])
        Alfa[i+1] = tmp
        beta[i+1] = np.dot(tmp, F[i+1] + beta[i])
    
    X = beta

    for i in range(m-2, -1, -1):
        X[i] = np.dot(Alfa[i], X[i+1]) + beta[i]

    return X

if len(sys.argv) < 2:
    print("N missing")
    exit(1)

n = int(sys.argv[1])
nm1 = n-1
nm2 = n-2
np1 = n+1

h = 1.0/n
h2 = h*h

x = np.linspace(0.0, 1.0, np1)
y = np.linspace(0.0, 1.0, np1)

F = np.zeros((nm1, nm1))
for i in range(nm1):
    F[0, i] += left(y[i+1]) / h2
    F[nm2, i] += right(y[i+1]) / h2
    F[i, 0] += bottom(x[i+1]) / h2
    F[i, nm2] += top(x[i+1]) / h2
    for j in range(nm2):
        F[i, j] += f(x[i+1], y[i+1])

F *= h2

D = np.ones((nm1, nm1))
D = 5 * np.eye(nm1) - sl.triu(sl.tril(D, 1), -1)

start = t.time()
U = sweep(nm1, D, F).transpose()
end = t.time()
print('time = ', end-start, sep='')

def ua(x, y):
    return x*x*y + y*y*x

Ua = np.zeros((np1, np1))
for i in range(np1):
    for j in range(np1):
        Ua[i, j] = ua(x[i], y[j])

ans = np.max(np.abs(Ua[:nm1, :nm1]-U))

print(f"|Ua-U| = {ans}")

x_i = x[1:n]
y_i = y[1:n]

mx = U.max()
mn = U.min()
print(mx, mn)
X, Y = np.meshgrid(x_i, x_i)

fig = plt.figure(figsize=(12, 5.5))
cmap = mpl.cm.get_cmap('RdBu_r')
ax = fig.add_subplot(1,2,1)
c = ax.pcolor(X, Y, U, vmin=mn, vmax=mx, cmap=cmap)
ax.set_xlabel(r"$x$", fontsize=14)
ax.set_ylabel(r"$y$", fontsize=14)

ax = fig.add_subplot(1, 2, 2, projection='3d')
p = ax.plot_surface(X, Y, U, vmin=mn, vmax=mx, rstride=3, cstride=3, linewidth=0, cmap=cmap)
ax.set_xlabel(r"$x$", fontsize=14)
ax.set_ylabel(r"$y$", fontsize=14)

cb = plt.colorbar(p, ax=ax, shrink=0.75)
cb.set_label(r"$u(x, y)$", fontsize=14)

fig1, ax1 = plt.subplots()

cs = plt.imshow(U, cmap='inferno')
fig1.colorbar(cs)
plt.show()
