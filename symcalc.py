

class Expr:
    def __init__(self, name="", *args, skip=False, is_int=False, is_symbol=False, is_negative=False):
        self.classname = name
        if not skip:
            self.args = args
            self.is_negative = False
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

    def __neg__(self):
        return Mul(self, -1)

class Integer(Expr):
    def __init__(self, n):
        super().__init__("Integer", is_int=True, is_negative=(n<0))
        self.n = n

    def tryInt(n):
        if isinstance(n, int):
            return Integer(n)

        return n

    def format(self):
        return f"{self.n}"

    def __hash__(self):
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
        obj.base = base
        obj.exp = exp

        return obj

    def __init__(self, base, exp, evaluate=True):
        super().__init__("Pow", self.base, self.exp)

    def __hash__(self):
        return hash(tuple(self.args))

    def __eq__(self, other):
        if not isinstance(other, Pow):
            return NotImplemented
        return self.base == other.base and self.exp == self.exp

    def format(self):
        if self.base.is_int or self.base.is_symbol:
            return f"{self.base}**{self.exp}"

        return f"({self.base})**{self.exp}"

class Mul(Expr):
    def __new__(cls, *args, evaluate=True):

        # If there are no arguments it is zero
        #if len(args) == 0:
        #    obj = super(Expr, cls).__new__(Integer)
        #    obj.__init__(0)
        #    return obj

        obj = super(Expr, cls).__new__(cls)
        obj.args = args
        obj.is_negative = False

        if not evaluate:
            return obj

        # 1. Flattening
        obj.args = obj.flatten()

        # 2. Identity Removing
        obj.args = [Integer.tryInt(arg) for arg in obj.args if arg != 1]

        # 3. Exponent Collecting
        obj.args = obj.as_exp()

        # 4. Term Sorting
        obj.args = obj.as_ordered()

        # If there is a single result return result (already either Integer or another Expr)
        if len(obj.args) == 1:
            return obj.args[0]

        if obj.args[0] == -1:
            obj.is_negative = True

        # Nothing more can be done just return normal Mul(*args)
        return obj

    def __init__(self, *args, evaluate=True):
        super().__init__("Mul", skip=True)

    def __hash__(self):
        return hash(tuple(self.args))

    def __eq__(self, other):
        if not isinstance(other, Mul):
            return NotImplemented
        return self.args == other.args

    def as_two_terms(self):
        if len(self.args) == 1:
            first = Integer(1)
            last = self.args[0]
        elif len(self.args) == 2:
            first = self.args[0]
            last = self.args[1]
        else:
            first = self.args[0]
            last = Mul(*[arg for arg in self.args[1:]])

        return first, last

    def as_ordered(self):
        new_args = []

        # First constants
        for arg in self.args:
            if arg.is_int:
                new_args.append(arg)

        sorted_args = {}
        other_args = []

        # Then symbols
        for arg in self.args:
            if arg.is_int:
                continue

            if isinstance(arg, Pow):
                if arg.base.is_symbol:
                    sorted_args[arg.base.s] = arg
                else:
                    other_args.append(arg)

            elif arg.is_symbol:
                sorted_args[arg.s] = arg
            else:
                other_args.append(arg)

        sorted_args = dict(sorted(sorted_args.items()))
        for s in sorted_args:
            arg = sorted_args[s]
            new_args.append(arg)

        new_args += other_args

        return new_args

    def as_exp(self):
        consts = 1
        d = {}
        for arg in self.args:
            if isinstance(arg, Pow):
                if any(arg.base == k for k in d.keys()):
                    d[arg.base] += arg.exp
                else:
                    d[arg.base] = arg.exp

            elif isinstance(arg, Integer):
                consts *= arg.n

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

        if consts == 1:
            new_args = []
        else:
            new_args = [Integer(consts)]
        for base in d:
            exp = d[base]
            if exp == 0:
                pass # Nothing 1*x = x
            elif exp == 1:
                new_args.append(base)
            else:
                new_args.append(Pow(base, exp))

        return tuple(new_args)

    def format(self):
        npar = lambda a: a.is_int or a.is_symbol
        if self.is_negative:
            return "-"+"*".join(f"{arg}" if npar(arg) else f"({arg})" for arg in self.args[1:])
        return "*".join(f"{arg}" if npar(arg) else f"({arg})" for arg in self.args)

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

        if not evaluate:
            return obj

        # 1. Flattening
        obj.args = obj.flatten()

        # 2. Identity Removing
        obj.args = [arg for arg in obj.args if arg != 0]

        # 3. Coefficient Collecting
        obj.args = obj.as_coeff()

        # Result is 0
        if len(obj.args) == 0:
            return Integer(0)

        # If there is a single result return result (already either Integer or another Expr)
        if len(obj.args) == 1:
            return obj.args[0]

        # Nothing more can be done just return normal Add(*args)
        return obj

    def __init__(self, *args, evaluate=True):
        super().__init__("Add", skip=True)

    def format(self):
        string = str(self.args[0])
        for i, arg in enumerate(self.args[1:]):
            if arg.is_negative:
                #sign = " - "
                sign = ""
                s = str(arg)
            else:
                sign = " + "
                s = str(arg)

            string += sign + s
        return string

    def as_coeff(self):
        d = {}
        consts = 0
        for arg in self.args:
            arg = Integer.tryInt(arg)

            if arg.is_int:
                consts += arg.n

            elif isinstance(arg, Mul):
                first, last = arg.as_two_terms()

                #if any(last == k for k in d.keys()):
                if last in d:
                    d[last] += first
                else:
                    d[last] = first

            else:
                #if any(arg == k for k in d.keys()):
                if arg in d:
                    d[arg] += 1
                else:
                    d[arg] = 1

        if consts == 0:
            new_args = []
        else:
            new_args = [Integer(consts)]

        for expr in d:
            coeff = d[expr]
            if coeff == 0:
                pass # 0*expr = 0
            elif coeff == 1:
                new_args.append(expr) # 1*expr = expr
            else:
                new_args.append(Mul(expr, coeff))

        return tuple(new_args)

    def flatten(self, obj=None):
        if obj == None:
            obj = self

        new_args = []
        for arg in obj.args:
            if isinstance(arg, Add):
                new_args += self.flatten(obj=arg)
            else:
                new_args.append(arg)

        return new_args

x = Symbol("x")
y = Symbol("y")
p = y*x*2
print(p)

