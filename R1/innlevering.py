
from regresjon import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D


with open("skraakast.txt", "r") as f:
    lines = f.readlines()
    lines = [line.split("     ") for line in lines]
    X = [float(line[0]) for line in lines]
    Y = [float(line[1]) for line in lines]
    Z = [float(line[2]) for line in lines]


# Bruker regresjon til å finne passende funksjoner for y og z
Yfunc = linear(X, Y)
Zfunc = polynomial(2, X, Z)

def kast(x):
    return np.array((Yfunc(x), Zfunc(x)))

def dkast(x):
    dx = 10**-7
    return (kast(x+dx) - kast(x-dx)) / (2*dx)

print()
print()
print(np.sqrt(sum(dkast(0)**2)))

plotx = np.linspace(0, 1)

print()
print(f"y(x) = {Yfunc.prettify()}")
print(f"z(x) = {Zfunc.prettify()}")
print("\n")

x = 0
for i in range(100):
    x -= kast(x)[0]/dkast(x)[0]
print(f"x = {x}, kast(x) = {kast(x)}")


ax = plt.axes(projection='3d')
plt.plot(X, Y, Z, "o", label="Skraakast")
#plt.plot(X, Y, label="Y")
#plt.plot(X, Z, label="Z")
#plotFunction(Yfunc, (0, 1))
#plotFunction(Zfunc, (0, 1))
plt.plot(plotx, Yfunc(plotx), Zfunc(plotx), label="3D regression")
plt.legend()
plt.grid()
plt.show()
