
import math
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - math.e**x - 10

def df(x):
    dx = 10 ** -6
    return (f(x+dx) - f(x-dx)) / (2*dx)

#def df(x):
#    dx = 10 ** -8
#    return (f(x+dx) - f(x)) / dx


nIterations = 15

xn = 4  # Et tall i nærheten av nullpunkt

for i in range(nIterations):
    print(f"Nullpunkt på {xn:.10f}, hvor f(x) = {f(xn):.10f}")
    xn = xn - f(xn) / df(xn)

print(f"Nullpunkt på {xn:.10f}, hvor f(x) = {f(xn):.10f}")


