import numpy as np
import matplotlib.pyplot as plt
import cmath 
import math 
from numpy import random
from scipy.integrate import quad
# QUESTION 1
def f(x):
    return 3*x + 3*x**3 - 6*x**2

population_mean, error = quad(f, 0, 1)

def F(x):
    if x <= 0:
        return 0
    elif 0 < x < 1:
        return 1 - (1 - x)**3
    else:
        return 1

def F_inv(u):
    return 1 - (1 - u)**(1/3)

def lcg(seed, a, c, m, n):
    u = np.zeros(n)
    x = seed
    for i in range(n):
        x = (a * x + c) % m
        u[i] = x / m
    return u

def generate_samples(N, seed, a, c, m):
    u = lcg(seed, a, c, m, N)
    x = F_inv(u)
    return x

def plot_cdf(x, N):
    x.sort()
    p = np.arange(1, N+1) / N
    plt.plot(x, p, label=f'Empirical CDF (N={N})')
    x_theory = np.linspace(0, 1, 1000)
    plt.plot(x_theory, [F(xi) for xi in x_theory], label=f'Theoretical CDF')
    plt.axvline(x=population_mean, color='k', linestyle='--', label='Population Mean')
    plt.title(f'Empirical and Theoretical CDFs (N={N})')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.legend()
    plt.show()

def calculate_moments(x):
    mean = np.mean(x)
    var = np.var(x)
    return mean, var

Ns = [10, 100, 1000, 10000, 100000]
seed = 12345
a = 1664525
c = 1013904223
m = 2**32

population_var = 0.0375

print(f'Population mean: {population_mean:.4f}, Population variance: {population_var:.4f}')

for N in Ns:
    x = generate_samples(N, seed, a, c, m)
    plot_cdf(x, N)
    mean, var = calculate_moments(x)
    print(f'N={N}: mean={mean:.4f}, var={var:.4f}')

# QUESTION 2
def cdf(x):
    if x <= 0:
        return 0
    elif 0 < x <= 1:
        return 1 - np.exp(-x)
    else:
        return 1 - np.exp(-(2*x-1))

def lcg(seed, a, c, m, n):
    u = np.zeros(n)
    x = seed
    for i in range(n):
        x = (a * x + c) % m
        u[i] = x / m
    return u

def generate_samples(n):
    seed = 12345
    a = 1664525
    c = 1013904223
    m = 244944
    u = lcg(seed, a, c, m, n)
    x = np.zeros(n)
    for i in range(n):
        if u[i] <0:
            x[i] = 0
        elif u[i] < 1:
            x[i] = -np.log(1 - u[i])
        else:
            x[i] = (1 + np.log(1 - u[i])) / 2
    return x

Ns = [10, 100, 1000, 10000, 100000]

for N in Ns:
    x = generate_samples(N)
    x.sort()
    p = np.arange(1, N+1) / N
    plt.plot(x, p, label=f'Empirical CDF (N={N})')
    x_theory = np.linspace(0, 5, 1000)
    plt.plot(x_theory, [cdf(xi) for xi in x_theory], label=f'Theoretical CDF')
    plt.title(f'Empirical and Theoretical CDFs (N={N})')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.legend()
    plt.show()

    mean = np.mean(x)
    var = np.var(x)
    print(f'N={N}: mean={mean:.4f}, var={var:.4f}')

# QUESTION 3
def binary_search(q, ui):
    start = 0
    end = len(q) - 1

    while start < end:
        mid = math.floor((start + end)/2)
        if(ui <= q[mid]):
            end = mid
        else:
            start = mid + 1
  
    return start

def main(): 
    C = np.arange(1, 10000, 2)
    q = [i/5000 for i in range(1, 5001)]

    sizes = [50000, 100000, 500000, 1000000, 5000000]
    for sz in sizes:
        np.random.seed(42)
        U = np.random.uniform(size=sz) 

        X = []
        for ui in U:
            idx = binary_search(q, ui)
            X.append(C[idx - 1])

        hist, bins = np.histogram(X, bins=50, density=True)
        w = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=w)
        plt.xlabel('Random Numbers (X)    (divided in intervals)')
        plt.ylabel('Probability')
        plt.title('Probability Distribution of X - total count = {}'.format(sz))
        plt.show()

        freq = {}
        for i in C:
            freq[i] = 0
        for i in X:
            freq[i] += 1

        for key, value in freq.items():
            freq[key] = value/len(X)

        plt.bar(C, list(freq.values()), align='center', label='Actual PDF')
        plt.xlabel('Random Numbers (X)')
        plt.ylabel('Probability')
        plt.title('Probability Mass Function of X - total count = {}'.format(sz))
        plt.show()
  
if __name__ == "__main__": 
    main()
