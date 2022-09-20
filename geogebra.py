
from pylab import *

dx = 0.0001


class Function:
    def __init__(self):
        pass

    def func(self, x):
        return x

    def d(self, x):
        return (self.func(x+dx) - self.func(x)) / dx


def parse(s):
    pairs = []
    for c in s:
        if c == " ":
            continue


a = "x * 4"

print(parse(a))
