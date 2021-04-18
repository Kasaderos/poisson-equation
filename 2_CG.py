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

N = 32 
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

a, b, c = -1, 4, -1
C = np.zeros((N_1, N_1))
C[0, 0], C[0, 1] = b, c 
for i in range(1, N_1-1):
    C[i, i-1] = a
    C[i, i] = b
    C[i, i+1] = c
C[N_1-1, N_1-2], C[N_1-1, N_1-1] = a, b 

C = 1/h2 * C
I = -1/h2 * np.eye(N_1)

F = np.zeros((N, N))

for i in range(1, N):
    for j in range(1, N):
        F[i, j] = f(x[i], y[j]) 
        if i-1 < 1 and 1 <= j <= N_1:
            F[i, j] += mu(0.0, y[j])
        if i+1 >= N and 1 <= j <= N_1:
            F[i, j] += mu(1.0, y[j])
        if j-1 < 1 and 1 <= i <= N_1:
            F[i, j] += mu(x[i], 0.0)
        if j+1 >= N and 1 <= i <= N_1:
            F[i, j] += mu(x[i], 1.0)

b = F[1:N, 1:N].ravel()
# print(C)
# print(I)
# print(b)
A = np.zeros((N_1*N_1, N_1*N_1)) 

A[:N_1, :N_1] = C.copy()
A[:N_1, N_1:2*N_1] = I.copy()
for i in range(1, N_1-1):
    A[i*N_1:(i+1)*N_1, (i-1)*N_1:(i)*N_1] = I.copy() 
    A[i*N_1:(i+1)*N_1, (i)*N_1:(i+1)*N_1] = C.copy() 
    A[i*N_1:(i+1)*N_1, (i+1)*N_1:(i+2)*N_1] = I.copy() 
A[(N_1-1)*N_1:N_1*N_1, (N_1-2)*N_1:(N_1-1)*N_1] = I.copy()
A[(N_1-1)*N_1:N_1*N_1, (N_1-1)*N_1:N_1*N_1] = C.copy()


M = np.diag(np.diag(A))

M_inv = np.linalg.inv(M)
# M = np.zeros((N_1*N_1, N_1*N_1)) 
# for i in range(N_1):
#     M[i*N_1:(i+1)*N_1, (i)*N_1:(i+1)*N_1] = C.copy() 

# print(A)
# print(M)
# print(b)
eigens = np.linalg.eigvals(A)
lambda_max = np.max(eigens)
lambda_min = np.min(eigens)
kappa = lambda_max / lambda_min
print("kappa=", kappa)
k = 0
x = np.zeros(N_1 * N_1)
r_old = b.copy() - np.dot(A, x) 
r = r_old.copy()


p = np.dot(M_inv, b.copy())

y_old = np.dot(M_inv, r)
y = y_old.copy()

eps = 1e-6

while norm_p2(r) > eps: 
    z = np.dot(A, p)
    nu = np.dot(y_old.T, r_old) / np.dot(p.T, z)
    x = x + nu * p
    r = r_old - nu * z
    y = np.dot(M_inv, r)
    mu = np.dot(y.T, r) / np.dot(y_old.T, r_old)
    p = y + mu * p

    r_old = r.copy()
    y_old = y.copy()
    
    u = x.reshape((N_1, N_1))
    err = norm(u - ua[1:N,1:N])
    print('max|u-u*| =', err)
    # print('||r_k||_2 =', norm_p2(r))
    time.sleep(0.1)

# print(x)
u = x.reshape((N_1, N_1))
# print(u)
# print(ua)
err = norm(u - ua[1:N, 1:N])
print('max|u-u*| =', err)

