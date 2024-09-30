import math
import numpy as np
from scipy.special import gamma

from jsonargparse import CLI

def main(J: int = 16,
         K: int = 2,
         trials: int = 2048,
         uniform: bool = False,
         hscale: bool = False,
         ):
    L = K**J
    D = trials
    h_scale = np.sqrt ((gamma(K/2) / gamma(J + K/2))**(1/J) / 2) if hscale else 1
    if uniform:
        x = np.ones((L,D)) / L
    else:
        x = np.random.normal (size=(L,D))
    h = np.random.normal (size=(K,D)) * h_scale
    a = x
    for i in range(J):
        b = np.empty((a.shape[0]//K,D))
        print(a.shape,"->",b.shape)
        for j in range(0,b.shape[0]):
            b[j,:] = sum ([a[2*j+k,:]*h[k,:] for k in range(K)])
        a = b
    var_x = np.var(x)
    var_h = np.var(h)
    H = np.sum(h**2,axis=0)
    print("h.shape:",h.shape)
    print("H.shape:",H.shape)
#    J_div_k = J // K
#    K_prime = J // J_div_k
    print("J:",J)
    print("K:",K)
    print("D:",D)
    print("h_scale:",h_scale)
    print("mean[h]:",np.mean(h))
    print("std[h]:",np.std(h))
    print("var[h]:",np.var(h))
    print("mean[x]:",np.mean(x))
    print("std[x]:",np.std(x))
    print("var[x]:",var_x)
    print("mean[H]:",np.mean(H),"should roughly equal var[h]*K:",var_h*K)
    print("var[H]:",np.var(H),"should roughly equal (var[h]^2)*2K:",(var_h**2)*2*K)
    print("mean[H^2]:",np.mean(H**2),"should roughly equal (var[h]^2) * 2^2 * gamma(3) / gamma(1):",var_h**2 * 2**2 * gamma(3) / gamma(1))
    for j in range(0,J+1):
        print(f" mean[H^{j}]:",np.mean(H**j),f"should roughly equal (var[h]^{j}) * 2^{j} * gamma({j/2 + K}) / gamma({j/2}):",(var_h**j) * (2**j) * gamma(j + K/2) / gamma(K/2))
    print("mean[H^J]:",np.mean(H**J),"should roughly equal var[h]^J * 2^J * gamma(J/2 + K) / gamma(J/2):",(var_h**J) * (2**J) * gamma(J + K/2) / gamma(K/2))
    print("V[x] * (2*V[h])^J * gamma(J + K/2) / gamma(K/2):",var_x * (2*var_h)**J * gamma(J + K/2) / gamma(K/2))
    print("mean[a0]:",np.mean(a))
    print("std[a0]:",np.std(a))
    print("var[a0]:",np.var(a))

def approx_dbl_fact(N):
    if N % 2 == 0:  # N = 2k
        k = N / 2
        return (2*k)**k * np.exp(-k)
    else:  # N = 2k-1
        k = (N+1) / 2
        return 2**(1-k) * np.exp(-k) * (2*k-1)**(2*k-1) * (k-1)**(1-k)

def fact(n):
    f = 1
    for i in range(1,n+1):
        f = f * i
    return f

def dblfact(n):
    f = 1
    for i in range(n%2+2,n+1,2):
        f = f * i
    return f

def v(n):
    return dblfact(2*n-1) - ((dblfact(n-1))**2 if n%2==0 else 0)

def approx_v(n):
    return approx_dbl_fact(2*n-1) - ((approx_dbl_fact(n-1))**2 if n%2==0 else 0)

def F(n):
    return G(2*n) - G(n)**2

def G(n):
    return (1 + (-1)**n) * 2**(n/2-1) * gamma((n+1)/2) / np.sqrt(np.pi)

CLI(main)