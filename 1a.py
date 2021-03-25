import numpy as np
import sys
import time as t

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

def ua(x, y):
    return x*x*y + y*y*x 

T = 1.0
L = 1.0
N = 32 
N1 = N+1
x = np.linspace(0.0, L, N1)
y = np.linspace(0.0, L, N1)

Ua = np.array([[ua(xi, yi) for yi in y] for xi in x])
mx = Ua.max()
mn = Ua.min()
X, Y = np.meshgrid(x, x)
fig = plt.figure(figsize=(12,5.5))
cmap = mpl.cm.get_cmap('RdBu_r')
ax =  fig.add_subplot(1,2,1)
c = ax.pcolor(X, Y, Ua, vmin=mn, vmax=mx, cmap=cmap)
ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)

ax =  fig.add_subplot(1,2,2, projection='3d')
p = ax.plot_surface(X, Y, Ua, vmin=mn, vmax=mx, rstride=3, cstride=3, linewidth=0, cmap=cmap)
ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)

cb = plt.colorbar(p, ax=ax, shrink=0.75)
cb.set_label(r"$u(x, y)$", fontsize=14)

fig1, ax1 = plt.subplots()
cs = plt.imshow(Ua, cmap='inferno')
fig1.colorbar(cs)
plt.show()
