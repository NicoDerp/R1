
import numpy as np


class Expr:
    def __init__(self, name="", *args, is_int=False):
        self.classname = name
        self.args = args
        self.is_int = is_int

    def __str__(self):
        args = ", ".join(str(arg) for arg in self.args)
        return f"{self.classname}({args})"


class Mul(Expr):
    def __init__(self, *args):
        super().__init__(self, "Mul")

        self.args = []
        for arg in args:
            if not isinstance(arg, Expr):
                self.args.append(Expr(arg, is_int=True))
            else:
                self.args.append(arg)

        # 1. Flattening

print(Mul(1, 2, Mul(4, 5)))

def sqrt(n):
    return n


