
import numpy as np


def f(x):
    return 2*x


a = 0
b = 10
n = 1000
X = np.linspace(a, b, n)

s = np.sum(f(X)) * (a + b) / n
print(f"Integral av funksjon f fra {a} til {b} er {s}")
