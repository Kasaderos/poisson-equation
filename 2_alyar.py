import numpy as np
import sys

import scipy.linalg as sl
from math import pi, log

def f(x,y):
    return -6*x*y-2*x

def left(y):
    return 0.0

def right(y):
    return y*y+y

def bottom(x):
    return 0

def top(x):
    return x*x*x+x

def count_norm(U1, U2):
    ans = 0
    k = 0
    l = 0
    for i in range(np1):
        for j in range(np1):
            tmp = abs(U2[i, j] - U1[i, j])
            if tmp > ans:
                k = i
                l = j
                ans = tmp
    return ans

import signal as s

def handler(signum, frame):
    print(f"cur_norm = {cur_norm}")
    exit(1)

s.signal(s.SIGINT, handler)

if len(sys.argv) < 2:
    print("Not enough args")
    exit(1)

eps = 1e-6
n = int(sys.argv[1])
np1 = n+1
h = 1.0/n
h2 = h*h

m_optimal = int((2*log(1/eps)) / ((pi*h)**2)) + 1
m_step = m_optimal // 19
print(f"m_optimal = {m_optimal}")

x = np.linspace(0.0, 1.0, np1)
y = np.linspace(0.0, 1.0, np1)

def ua(x, y):
    return x*y * (x*x + y)

Ua = np.zeros((np1, np1))
for i in range(np1):
    for j in range(np1):
        Ua[i, j] = ua(x[i], y[i])

def write_data_to(filename, numb, Ua, U0a, Uk, Ukm1):
    rect = 0
    ans54 = count_norm(Uk, Ua)
    ans55 = ans54 / U0a
    ans56 = count_norm(Uk, Ukm1)

    with open(filename, 'a') as file:
        print(f'{str(numb).ljust(5)} {str(ans54).ljust(25)} {str(ans55).ljust(25)} {str(ans56).ljust(25)}', file=file)

U0 = np.zeros((np1,np1))
for i in range(np1):
    U0[i, 0] = bottom(x[i])
    U0[i, n] = top(x[i])
    U0[0, i] = left(y[i])
    U0[n, i] = right(y[i])

U0a = count_norm(U0, Ua)
U = U0.copy()

import time as t

start = t.time()

print(f"h = {h}")
cur_norm = 1
prev_norm = 2
prev_U = U.copy()

with open(f"table_data/optimal_{n}.txt", 'w') as file:
    print(f'{"k".ljust(5)} {"||Uk - u*||".ljust(25)} {"||Uk - u*||/||U0 - u*||".ljust(25)} {"||Uk - U(k-1)||".ljust(25)}', file=file)

p = 1
m = 0
while cur_norm > eps:
    cur_norm = count_norm(U, Ua)
    if m > m_optimal:
        print("Couldn't reach desired precision!")
        break

    for i in range(1, n):
        for j in range(1, n):
            U[i, j] = (prev_U[i-1, j] + prev_U[i+1, j] + prev_U[i, j-1] + prev_U[i, j+1] + h*h*f(x[i], y[j]))/4
    m += 1

    if m == p:
        write_data_to(f"table_data/optimal_{n}.txt", p, Ua, U0a, U, prev_U)
        p += m_step

    prev_U = U.copy()
    prev_norm = cur_norm

U = prev_U
end = t.time()

print('time = ', end-start)

ans = count_norm(U, Ua)

print(f"|Ua-U| = {ans}")
print(f"m_optimal = {m_optimal}")
print(f"m = {m}")









