
from pylab import *
import types


__functions = {}

class Function():
    def __init__(self, func, space=(-10, 10, 1001), dx=0.001, label=None):
        if isinstance(func, str):
            self.func = eval("lambda x:" + func, np.__dict__)
            self.funcs = func
        elif isinstance(func, types.FunctionType):
            self.func = func
            self.funcs = None
        else:
            raise ValueError("Func isn't string or lambda!")

        self.space = space

        if len(space) != 3:
            raise ValueError("len(space) isn't 3")

        self.lin = linspace(*space)
        self.dx = dx
    
        self.label = label

        __functions.append(self)
        
    def __call__(self, x):
        return self.func(x)

    def __ddx(self):
        f = lambda x: (self.func(x + self.dx) - self.func(x)) / self.dx
        func = self.copy().setFunc(f)
        if func.label:
            func.label += "'"
        return func

    def __getname(self):
        if self.label != None:
            return self.label

        l = [k for k,v in globals().items() if v is self]
        self.label = l[0] if len(l) != 0 else ""
        return self.label

    def ddx(self, n=1):
        self.__getname()
        f = self
        for i in range(n):
            f = f.__ddx()
        return f

    def copy(self):
        return Function(func=self.func, space=self.space, dx=self.dx, label=self.label)

    def setFunc(self, func):
        self.func = func
        return self

    def plot(self):
        l = self.__getname()
        if l:
            plot(self.lin, self.func(self.lin), label=l)
        else:
            plot(self.lin, self.func(self.lin))


f = Function("f(x) = 2*x**3 - 6*x**2 - 2*x + 6")
g = Function("g(x) = x**2")

f.plot()
f.ddx().plot()
f.ddx(2).plot()

g.plot()

legend()
igrid()
show()

