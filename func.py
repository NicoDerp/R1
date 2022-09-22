
from pylab import *


class Function():
    def __init__(self, func, space=(-10, 10, 1001), dx=0.001):
        self.func = func
        self.space = space

        if len(space) != 3:
            raise ValueError("len(space) isn't 3")

        self.lin = linspace(*space)
        self.dx = dx

    def __call__(self, x):
        return self.func(x)

    def __ddx(self):
        f = lambda x: (self.func(x + self.dx) - self.func(x)) / self.dx
        return self.copy().setFunc(f)

    def ddx(self, n=1):
        f = self
        for i in range(n):
            f = f.__ddx()
        return f

    def copy(self):
        return Function(func=self.func, space=self.space, dx=self.dx)

    def setFunc(self, func):
        self.func = func
        return self

    def plot(self):
        plot(self.lin, self.func(self.lin))

def f(x):
    return 2*x**3 - 6*x**2 - 2*x + 6

def ddx(x, func):
    dx = 0.001
    return (func(x+dx) - func(x)) / dx


myFunc = Function(lambda x: x**2)

X = np.linspace(-10, 10, 1001)
Y = f(X)

myFunc.plot()
myFunc.ddx().plot()
myFunc.ddx(2).plot()

grid()
show()

