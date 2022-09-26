
import numpy as np


class Expr:
    def __init__(self, name="", *args, is_int=False, is_symbol=False):
        self.classname = name
        self.args = args
        self.is_int = is_int
        self.is_symbol = is_symbol
        self._is_number = False

    def format(self):
        args = ", ".join((str(arg) for arg in self.args))
        return f"{self.classname}({args})"

    @property
    def is_number(self):
        return all(arg.is_number for arg in self.args)

    def __str__(self):
        return self.format()

    def __repr__(self):
        return self.format()

    def __pow__(self, other):
        other = Integer.tryInt(other)
        if not issubclass(type(other), Expr):
            return NotImplemented
        return Pow(self, other)

    def __mul__(self, other):
        other = Integer.tryInt(other)
        if not issubclass(type(other), Expr):
            return NotImplemented
        return Mul(self, other)

    def __rmul__(self, other):
        other = Integer.tryInt(other)
        if not issubclass(type(other), Expr):
            return NotImplemented
        return Mul(self, other)

    def __add__(self, other):
        other = Integer.tryInt(other)
        if not issubclass(type(other), Expr):
            return NotImplemented
        return Add(self, other)

    def __radd__(self, other):
        other = Integer.tryInt(other)
        if not issubclass(type(other), Expr):
            return NotImplemented
        return Add(self, other)

    def __sub__(self, other):
        other = Integer.tryInt(other)
        if not issubclass(type(other), Expr):
            return NotImplemented
        return Add(self, Mul(other, -1))

    def __rsub__(self, other):
        other = Integer.tryInt(other)
        if not issubclass(type(other), Expr):
            return NotImplemented
        return Add(Mul(self, -1), other)

class Integer(Expr):
    def __init__(self, n):
        super().__init__("Integer", is_int=True)
        self.n = n

    def tryInt(n):
        if isinstance(n, int):
            return Integer(n)

        return n

    def format(self):
        return f"{self.n}"

    def __hash(self):
        return hash(self.n)

    def __eq__(self, other):
        if isinstance(other, Integer):
            return self.n == other.n
        if isinstance(other, int):
            return self.n == other
        return NotImplemented

# Symbols like a, b, x, y
class Symbol(Expr):
    def __init__(self, s):
        super().__init__("Symbol", is_symbol=True)
        self.s = s

    def format(self):
        return f"{self.s}"

    def __hash__(self):
        return hash(self.s)

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return NotImplemented
        return self.s == other.s

# a**b
class Pow(Expr):
    def __new__(cls, base, exp, evaluate=True):
        base = Integer.tryInt(base)
        exp = Integer.tryInt(exp)

        # If exponent is 1 then set object to base instead of Pow(base, 1)
        if exp.is_int and exp.n == 1:
            return base

        # If exponent is 0 then the answer is 1, always
        if exp.is_int and exp.n == 0:
            return Integer(1)

        # If we evaluate
        if evaluate:
            # If both base and exponent is integer then we can return Integer(base ** exp)
            if base.is_int and exp.is_int:
                return Integer(base.n ** exp.n)

        # Nothing more can be done just return normal Pow(base, exp)
        obj = super(Expr, cls).__new__(cls)
        return obj

    def __init__(self, base, exp, evaluate=True):
        super().__init__("Pow", base, exp)
        self.base = Integer.tryInt(base)
        self.exp = Integer.tryInt(exp)

        self.result = None

        if not evaluate:
            return

        if self.base.is_int and self.exp.is_int:
            self.result = self.base.n ** self.exp.n

    def format(self):
        if self.result:
            return f"{self.result}"

        return f"{self.base}**{self.exp}"

class Mul(Expr):
    def __init__(self, *args, evaluate=True):
        super().__init__("Mul", *args)

        if not evaluate:
            return

        # 1. Flattening
        self.args = self.flatten()

        # 2. Identity Removing
        self.args = [Integer.tryInt(arg) for arg in self.args if arg != 1]

        # 3. Exponent Collecting
        self.args = self.as_exp()

    def as_exp(self):
        d = {}
        for arg in self.args:
            if isinstance(arg, Pow):
                if any(arg.base == k for k in d.keys()):
                    d[arg.base] += arg.exp
                else:
                    d[arg.base] = arg.exp

            elif isinstance(arg, Integer):
                if any(arg == k for k in d.keys()):
                    d[arg.n] += 1
                else:
                    d[arg.n] = 1

            elif isinstance(arg, Symbol):
                if arg in d.keys():
                    d[arg] += 1
                else:
                    d[arg] = 1

            else:
                if any(arg == k for k in d.keys()):
                    d[arg] += 1
                else:
                    d[arg] = 1

        new_args = []
        for base in d:
            exp = d[base]
            new_args.append(Pow(base, exp))

        return tuple(new_args)

    def format(self):
        return "*".join(f"{arg}" if arg.is_int or arg.is_symbol else f"({arg})" for arg in self.args)

    def flatten(self, obj=None):
        if obj == None:
            obj = self

        new_args = []
        for arg in obj.args:
            if isinstance(arg, Mul):
                new_args += self.flatten(arg)
            else:
                new_args.append(arg)
        return new_args

class Add(Expr):
    def __new__(cls, *args, evaluate=True):

        # If there are no arguments it is zero
        if len(args) == 0:
            obj = super(Expr, cls).__new__(Integer)
            obj.__init__(0)
            return obj

        obj = super(Expr, cls).__new__(cls)
        obj.args = args

        # 1. Flattening
        obj.args = obj.flatten()

        # 2. Identity Removing
        obj.args = [arg for arg in obj.args if arg != 0]

        # 3. Coefficient Collecting
        obj.args = obj.as_coeff()

        # If there is a single result return Integer(result)
        if len(obj.args) == 1:
            return Integer(*obj.args)

        # Nothing more can be done just return normal Add(*args)
        return obj

    def __init__(self, *args, evaluate=True):
        pass
    #    super().__init__("Add", *args)

    def format(self):
        return " + ".join(f"{arg}" for arg in self.args)

    def as_coeff(self):
        d = {}
        consts = 0
        for arg in self.args:
            arg = Integer.tryInt(arg)

            if arg.is_int:
                consts += arg.n

            else:
                if any(arg == k for k in d.keys()):
                    d[arg] += 1
                else:
                    d[arg] = 1

        new_args = [Integer(consts)]
        for expr in d:
            coeff = d[expr]
            new_args.append(Mul(expr, coeff))

        return tuple(new_args)

    def flatten(self, obj=None):
        if obj == None:
            obj = self

        new_args = []
        for arg in self.args:
            if isinstance(arg, Add):
                new_args += self.flatten(arg)
            else:
                new_args.append(arg)
        return new_args

x = Symbol("x")
a = Add(x, x, 2, 3)
print(a)

