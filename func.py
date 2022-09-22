
from pylab import *
import types


__functions = {}

class Function():
    def __init__(self, func, name=None, space=(-10, 10, 1001), dx=0.001):
        if isinstance(func, str):
            self.name = ""
            for c in func:
                if c == "("
            self.func = eval("lambda x:" + func, np.__dict__)
            self.funcs = func

        elif isinstance(func, types.FunctionType):
            if not name:
                raise ValueError("Func is function but no name supplied!")

            self.func = func
            self.funcs = None
            self.name = name

        else:
            raise ValueError("Func isn't string or lambda!")

        if len(space) != 3:
            raise ValueError("len(space) isn't 3")

        self.space = space

        self.lin = linspace(*space)
        self.dx = dx
    
        __functions.append(self)
        
    def __call__(self, x):
        return self.func(x)

    def __ddx(self):
        f = lambda x: (self.func(x + self.dx) - self.func(x)) / self.dx
        func = self.copy().setFunc(f)
        func.name += "'"
        return func

    def ddx(self, n=1):
        self.__getname()
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
        plot(self.lin, self.func(self.lin), label=self.name)


f = Function("f(x) = 2*x**3 - 6*x**2 - 2*x + 6")
g = Function("g(x) = x**2")

f.plot()
f.ddx().plot()
f.ddx(2).plot()

g.plot()

legend()
igrid()
show()

