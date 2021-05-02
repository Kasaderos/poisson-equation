import numpy as np
import time

def norm(a):
    return np.max(np.abs(a))

def norm_p2(a):
    return np.sqrt(np.dot(a, a))

def mu(x, y):
    return x*x*x*y + x*y*y*y

def f(x, y):
    return 12 * x * y

# def mu(x, y):
#     return np.cos(x) + np.sin(y)

# def f(x, y):
#     return -np.cos(x) - np.sin(y)

N = 4 
N1 = N + 1 
N_1 = N - 1
L = 1.0 
h = L/N
h2 = h*h
x = np.linspace(0.0, L, N1)
y = np.linspace(0.0, L, N1)
# print(x)
# print(y)

u = np.zeros((N1, N1))

ua = np.zeros((N1,N1))
for i in range(N1):
    for j in range(N1):
        ua[i, j] = mu(x[i], y[j])

# a, b, c = 1/h2 * np.array([-1.0, 4.0, -1.0])
a, b, c = np.array([-1.0, 4.0, -1.0])
# e = -1/h2 
e = -1.0
F = np.zeros((N, N))

for i in range(1, N):
    for j in range(1, N):
        F[i, j] = -f(x[i], y[j]) 
        if i-1 < 1:
            F[i, j] += mu(0.0, y[j]) / h2
        if i+1 >= N:
            F[i, j] += mu(1.0, y[j]) / h2
        if j-1 < 1:
            F[i, j] += mu(x[i], 0.0) / h2
        if j+1 >= N:
            F[i, j] += mu(x[i], 1.0) / h2

f = F[1:N, 1:N].ravel()

# eigens = np.linalg.eigvals(A)
# lambda_max = np.max(eigens)
# lambda_min = np.min(eigens)
# kappa = lambda_max / lambda_min
# print("lamba_min = ", lambda_min, "lambda_max = ", lambda_max, "kappa = ", kappa)
k = 0
x = np.zeros(N_1 * N_1)

r_old = f.copy()
r = f.copy() 
p = f.copy()

eps = 1e-3
N_2 = N_1 - 1
z = np.zeros(N_1 * N_1)
print(f)
#  4 -1  0 -1  0  0  0  0  0 -0.75             
# -1  4 -1  0 -1  0  0  0  0 -1.5
#  0 -1  4  0  0 -1  0  0  0  2.
# -1  0  0  4 -1  0 -1  0  0 -1.5
#  0 -1  0  1  4 -1  0 -1  0 -3.
#  0  0 -1  0 -1  4  0  0 -1  5.5
#  0  0  0 -1  0  0  4 -1  0  2. 
# -0  0  0  0 -1  0 -1  4 -1  5.5
#  0  0  0  0  0 -1  0 -1  4  30.75
print(a, b, c, e)
while norm_p2(r) > eps: 
    # z1 = b*b1+c*b2+e*b4
    # z2 = a*b1+b*b2+c*b3+e*b5
    # z3 = a*b2+b*b3+c*b4+e*b6
    # z4 = e*b1 + a*b3+b*b4+c*b5+e*b7
    # ...
    # zn_1 = e*b_n_4 + a*b_n_2 + b*b_n_1 + c*b_n
    # zn = e*b_n_3 + a*b_n_1 + b*b_n 
    z[0] = b*p[0] + c*p[1] + e*p[1+N_2]
    j = 0
    for i in range(1, N_2):
        z[i] = a*p[j] + b*p[j+1] + c*p[j+2] + e*p[j+2+N_2]
        j += 1
    z[N_2] = a*p[j] + b*p[j+1] + e*p[j+1+N_1]
    print(z)

    j = 0
    for i in range(N_1, 2*N_1):
        z[i] = e*p[j] + a*p[j+N_1] + b*p[j+N_1+1] + c*p[j+N_1+2] + e*p[j+N_1+2+N_1]
        j += 1
    print(z)
    
    j = N_1
    z[i] = e*p[j] + a*p[j+N_1] + b*p[j+N_1+1] + c*p[j+N_1+2]
    for i in range(N_1*(N_1-1), (N_1*N_1-N_2+2)):
        z[i] = e*p[j] + a*p[j+N_2] + b*p[j+N_2+1] + c*p[j+N_2+2]
        j += 1
    z[N_1*N_1-1] = e*p[N_1*N_1-N_1-1] + a*p[N_1*N_1-N_1+N_2] + b*p[N_1*N_1-1]
    print(z)
    # z = np.dot(A, p)
    nu = np.dot(r_old.T, r_old) / np.dot(p.T, z)
    x = x + nu * p
    r = r_old - nu * z
    mu = np.dot(r.T, r) / np.dot(r_old.T, r_old)
    p = r + mu * p

    r_old, r = r, r_old
    
    u = x.reshape((N_1, N_1))
    err = norm(u - ua[1:N,1:N])
    print('max|u-u*| =', err)
    print('||r_k||_2 =', norm_p2(r))
    time.sleep(100)


