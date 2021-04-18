# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 17:57:15 2021

@author: kasaderos
"""

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

N = 32 
N1 = N+1
h = 1.0/N
x = np.linspace(0.0, 1.0, N1)
y = np.linspace(0.0, 1.0, N1)


ua = np.zeros((N1,N1))
for i in range(N1):
    for j in range(N1):
        ua[i, j] = mu(x[i], y[j])

u = np.zeros((N1, N1))
u0 = np.zeros((N1, N1))
u_old = np.zeros((N1, N1))

err = norm(u_old-ua)
eps = 1e-5
m = int((np.log(1/eps))/(np.pi*h)**2+1)
print('optimal m = ', m)
print('h^2 = ', h*h)

for i in range(N1):
    u_old[i, 0] = mu(x[i], 0.0)
    u_old[i, N] = mu(x[i], 1.0)
    u_old[0, i] = mu(0.0, y[i]) 
    u_old[N, i] = mu(1.0, y[i])

u = u_old.copy()
while m > 0:
    for i in range(1, N):
        for j in range(1, N):
            u[i, j] = (u[i-1, j] + u_old[i+1, j] + u[i, j-1] + u_old[i, j+1] + h*h*f(x[i], y[j])) / 4 

    u_old = u.copy()
    err = norm(u - ua)
    m -= 1
    # print(err, err / norm(u0-ua), norm(u_old-u))

print('max|u-u*| =', err)

'''
mx = u.max()
mn = u.min()
X, Y = np.meshgrid(x, x)
fig = plt.figure(figsize=(12,5.5))
cmap = mpl.cm.get_cmap('RdBu_r')
ax =  fig.add_subplot(1,2,1)
c = ax.pcolor(X, Y, u, vmin=mn, vmax=mx, cmap=cmap)
ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)

ax =  fig.add_subplot(1,2,2, projection='3d')
p = ax.plot_surface(X, Y, u, vmin=mn, vmax=mx, rstride=3, cstride=3, linewidth=0, cmap=cmap)
ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)

cb = plt.colorbar(p, ax=ax, shrink=0.75)
cb.set_label(r"$u(x, y)$", fontsize=14)

fig1, ax1 = plt.subplots()
cs = plt.imshow(u, cmap='inferno')
fig1.colorbar(cs)
plt.show()
'''
