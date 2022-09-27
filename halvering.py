
from pylab import *


def f(x):
    return 0.5 * x**3 - 2*x**2 + 1

def zero(func, space, accuracy=10**-5, max_iterations = 1000):
    a, b = space

    i = 0
    while abs(func((m := (a + b) / 2))) >= accuracy and i < max_iterations:
        if func(a) * func(m) < 0:
            b = m
        else:
            a = m
        i += 1

    if i >= max_iterations:
        return None

    return m

s = zero(f, (1, 5))
print(s)

x = linspace(-3, 4, 1001)
y = f(x)

plot(x, y)
grid()
show()


