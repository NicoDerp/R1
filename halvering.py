
from pylab import *


def f(x):
    return 2*log(x**4 + 4) - 0.5*x


def zero(func, space, accuracy=10**-5, max_iterations = 100):
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

s = zero(f, (-10, 80))

x = linspace(-10, 80, 1001)
y = f(x)

plot(x, y)
grid()
show()


