
import matplotlib.pyplot as plt
import numpy as np
import random
import re


class Function:
    def __init__(self, f, x=None, y=None, multiVariable=False):
        self.f = f
        self.x = x
        self.y = y
        self.multiVariable = multiVariable
    
    def prettify(self, decimals=2):
        return "ooga booga"
    
    def prettifyLatex(self, decimals=2):
        return f"${self.prettify(decimals)}$"
    
    def getY(self, r):
        x = np.linspace(r[0], r[1], 51)
        y = self.f(x)
        return y
    
    def __call__(self, x):
        return self.f(x)

class Polynomial(Function):
    def __init__(self, coeffs, degree, x=None, y=None):
        self.f = lambda x: sum([coeffs[i]*x**i for i in range(degree+1)])
        self.coeffs = coeffs
        self.consts = [[] for i in range(degree+1)]
        self.degree = degree
        super().__init__(self.f, x, y)
    
    def prettify(self, decimals=2):
        s = ""
        for i in range(self.degree, -1, -1):
            c = self.coeffs[i]
            
            if c == 0:
                continue
            
            if c < 0:
                c = -c
                if i == self.degree:
                    s += "-"
                else:
                    s += " - "
            elif i != self.degree:
                s += " + "
            
            if i == 0:
                s += f"{c:.{decimals}f}"
                for n in self.consts[i]:
                    s += f"{n}"
            elif i == 1:
                if round(c, 2) == 1.0:
                    for n in self.consts[i]:
                        s += f"{n}"
                    s += "x"
                else:
                    s += f"{c:.{decimals}f}"
                    for n in self.consts[i]:
                        s += f"{n}"
                    s += "x"
            elif round(c, 2) == 1.0:
                for n in self.consts[i]:
                    s += f"{n}"
                s += f"x^{i}"
            else:
                s += f"{c:.{decimals}f}"
                for n in self.consts[i]:
                    s += f"{n}"
                s += f"x^{i}"

        return s

class Linear(Polynomial):
    def __init__(self, a, b, x=None, y=None):
        self.a = a
        self.b = b
        super().__init__([b, a], 1, x, y)

class Exponential(Function):
    def __init__(self, a, b, x=None, y=None):
        self.f = lambda x: a*b**x
        self.coeffs = [a, b]
        self.consts = [[], []]
        self.degree = 1
        super().__init__(self.f, x, y)
    
    def prettify(self, decimals=2):
        s = ""
        
        c = self.coeffs[0]
        if c < 0:
            c = -c
            s += "-"
        
        if round(c, 2) != 1.0:
            s += f"{c:.{decimals}f}*"
 
        c = self.coeffs[1]
        s += f"{c:.{decimals}f}^x"

        return s

class Logarithmic(Function):
    def __init__(self, a, b, x=None, y=None):
        self.f = lambda x: a+b*np.log(x)
        self.coeffs = [a, b]
        self.consts = [[], []]
        self.degree = 1
        super().__init__(self.f, x, y)
    
    def prettify(self):
        s = ""
        
        if round(self.coeffs[0], 2) != 0.0:
            s += f"{self.coeffs[0]:.2f}"
 
        c = self.coeffs[1]
        if c < 0:
            c = -c
            s += " - "
        else:
            s += " + "
        s += f"{c:.2f}*ln(x)"

        return s

class Equation:
    def __init__(self, *args):
        if not isinstance(args[0], Function):
            l = args[0].split("=")
            if len(l) != 2:
                print("[ERROR] Cannot parse equation because it doesn't contain '='")
                exit()
            
            f, g = l
            self.f = parsePolynomial(f)
            self.g = parsePolynomial(g)
        else:
            self.f = args[0]
            self.g = args[1]
    
    def solve(self, maxIterations=100):
        dx = 10**-6
        func = lambda x: self.f(x) - self.g(x)
        dfunc = lambda x: (func(x + dx) - func(x - dx)) / (2*dx)
        xn = 0
        for i in range(maxIterations):
            xn = xn - func(xn) / dfunc(xn)
        return xn
    
    def prettify(self):
        return f"{self.f.prettify()} = {self.g.prettify()}"
    
    def prettifyLatex(self):
        return f"${self.f.prettify()} = {self.g.prettify()}$"

def parsePolynomial(s):
    
    # 2ab^5
    matches = re.findall((
            # Luft
            r"[\s]*"
            
            # Optional + eller -
            r"([-+])?"
            
            # Luft
            "[\s]*"
            
            r"(?:"
                # Optional desimaltall eller et tall
                r"(\d+\.\d+|\d+)?"
                
                # (En bokstav. luft. optional *) minst en
                r"(?:"
                    # Luft
                    r"[\s]*"
                    
                    # Optional gangetegn
                    r"\*?"
                    
                    # En bokstav
                    r"([a-zA-Z])"
                    
                    # Optional (^|**) og et tall
                    r"(?:"
                        r"(?:"
                            r"\^"
                            
                            r"|"
                            
                            r"\*\*"
                        r")"
                        
                        r"(\d+)"
                    r")?"
                r")+"
                
                r"|"
                
                # Desimaltall eller et tall
                r"(\d+\.\d+|\d+)"
                
                # Luft
                r"[\s]*"

                # Optional gangetegn
                r"\*?"
                
                # Luft
                r"[\s]*"
                
                # (En bokstav. luft. optional *) 0 eller mer
                r"(?:"
                    # Optional gangetegn
                    r"\*?"
                    
                    # En bokstav
                    r"([a-zA-Z])"
                    
                    # Optional (^|**) og et tall
                    r"(?:"
                        r"(?:"
                            r"\^"
                            
                            r"|"
                            
                            r"\*\*"
                        r")"
                        
                        r"(\d+)"
                    r")?"
                    
                    # Luft
                    r"[\s]*"
                r")*"
            r")"), s)
    
    coeffs = {}
    consts = []
    
    print(matches)
    
    # ax^b
    for match in matches:
        sign = match[0]
        a = "1"
        csts = []
        if match[1]:
            a = match[1]
            b = match[2]
        if match[3]:
            a = match[3]
            b = "0"
        elif not match[2]:
            b = "1"
        a = float(sign + a)
        b = float(b)
        print(match, f"{a}*x^{b}")
        
        if b in coeffs:
            coeffs[b] += a
        else:
            coeffs[b] = a
    
    print(coeffs)
    exit()
    degree = len(coeffs)-1
    if degree == 1:
        return Linear(coeffs[1], coeffs[0])
    return Polynomial([coeffs[i] for i in range(degree+1)], degree)
                     
def parseEquation(s):
    l = s.split("=")
    if len(l) != 2:
        print("[ERROR] Cannot parse equation because it doesn't contain '='")
        exit()
    
    f, g = l
    f = parsePolynomial(f)
    g = parsePolynomial(g)
    return Equation(f, g)

def linear(X, Y, maxIterations=100000, rate=0.001):
    X = np.array(X)
    Y = np.array(Y)
    a, b = [0, 0]
    learning_rate = rate
    m = float(len(Y))
    for i in range(maxIterations):
        predicted = [a*x + b for x in X]
        
        Da = -2 * sum(X * (Y - predicted)) / m
        Db = -2 * sum(Y - predicted) / m
        
        error = sum((Y - predicted) ** 2) / m
        
        a -= learning_rate * Da
        b -= learning_rate * Db
        
        if i % 1000 == 0:
            print(error, i, a, b)
        
        if error < 0.000001:
            break
    
    return Linear(a, b, X, Y)


def polynomialAdaDelta(degree, X, Y, maxIterations=100000):
    coeffs = [0 for i in range(degree+1)]
    #learning_rate = 0.00001
    learning_rate = 0.001
    m = float(len(Y))
    Ms = [0 for i in range(degree+1)]
    epsilon = 10 ** -8
    EG2t = np.zeros(degree+1)
    EdO2 = np.zeros(degree+1)
    lastError = 0
    for i in range(maxIterations):
        predicted = [sum([coeffs[i]*x**i for i in range(degree+1)]) for x in X]
        
        Ds = np.array([-2 * sum(X**i * (Y - predicted)) / m for i in range(degree+1)])
        
        error = sum((Y - predicted) ** 2) / m
        
        EG2t = 0.9*EG2t + (1 - 0.9)*Ds**2
        
        # Momentum
        #Ms = [0.9*Ms[i] + learning_rate * Ds[i] for i in range(degree+1)]
        
        # AdaDelta
        #Ms = np.array([(np.sqrt(0.9*EdO2[i] + epsilon) / np.sqrt(EG2t[i] + epsilon)) * Ds[i] for i in range(degree+1)])
        
        # AdaDelta numpizized
        Ms = (np.sqrt(0.9*EdO2 + epsilon) / np.sqrt(EG2t + epsilon)) * Ds
        
        EdO2 = 0.9*EdO2 + (1 - 0.9)*Ms**2
        
        coeffs = [coeffs[i] - Ms[i] for i in range(degree+1)]
        #print(error, coeffs)
        if i % 100 == 0:
            print(error)
        #print(Y)
        
        if error < 0.0001:
            break
        
        if any(np.isnan(coeffs)):
            print("[ERROR] Got NaN, try another function or supply more points")
            exit()
        
        if error == lastError:
            break
            
        lastError = error
    
    return Polynomial(coeffs, degree, X, Y)

def polynomial(degree, X, Y, maxIterations=1000000, rate=0.001):
    X = np.array(X)
    Y = np.array(Y)
    coeffs = [0 for i in range(degree+1)]
    #learning_rate = 0.00001
    learning_rate = rate
    B1 = 0.9
    B2 = 0.999
    epsilon = 10 ** -8
    
    m = float(len(Y))
    Mt = np.zeros(degree+1)
    Vt = np.zeros(degree+1)
    lastError = 0
    for i in range(maxIterations):
        predicted = [sum([coeffs[i]*x**i for i in range(degree+1)]) for x in X]
        
        Ds = np.array([-2 * sum(X**i * (Y - predicted)) / m for i in range(degree+1)])
        error = sum((Y - predicted) ** 2) / m
        
        Mt = B1*Mt + (1 - B1) * Ds
        Vt = B2*Vt + (1 - B2) * Ds**2
        
        Mht = Mt / (1 - B1)
        Vht = Vt / (1 - B2)
        
        # Adam numpizized
        Ms = (learning_rate / (np.sqrt(Vht) + epsilon)) * Mht
        
        coeffs = [coeffs[i] - Ms[i] for i in range(degree+1)]
        #print(error, coeffs)
        #if i % 1000 == 0:
        print(i, error)
        #print(Y)
        
        if error < 0.0000001:
            break
        
        if any(np.isnan(coeffs)):
            print("[ERROR] Got NaN, try another function or supply more points")
            exit()
        
        if error == lastError:
            break
            
        lastError = error

    return Polynomial([c for c in coeffs], degree, X, Y)

def exponential(X, Y, maxIterations=100000, rate=0.001):
    #func = linear(X, np.log(Y))
    func = polynomial(1, X, np.log(Y), rate=rate)
    return Exponential(np.exp(func.coeffs[0]), np.exp(func.coeffs[1]), X, Y)

def logarithmic(X, Y, maxIterations=100000):
    #func = linear(X, np.log(Y))
    #print(np.exp(Y/max(Y)))
    nY = Y / max(Y)
    nY = Y - min(nY)
    func = polynomial(1, X, np.exp(nY), rate=0.001)
    return Logarithmic(np.log(func.coeffs[0])*max(Y), -np.log(func.coeffs[1]), X, Y)

def solveEquations(equations, maxIterations=100000):
    coeffs = [0 for i in range(degree+1)]
    #learning_rate = 0.00001
    learning_rate = 0.001
    B1 = 0.9
    B2 = 0.999
    epsilon = 10 ** -8
    
    m = float(len(Y))
    Mt = np.zeros(degree+1)
    Vt = np.zeros(degree+1)
    lastError = 0
    for i in range(maxIterations):
        predicted = [sum([coeffs[i]*x**i for i in range(degree+1)]) for x in X]
        
        Ds = np.array([-2 * sum(X**i * (Y - predicted)) / m for i in range(degree+1)])
        error = sum((Y - predicted) ** 2) / m
        
        Mt = B1*Mt + (1 - B1) * Ds
        Vt = B2*Vt + (1 - B2) * Ds**2
        
        Mht = Mt / (1 - B1)
        Vht = Vt / (1 - B2)
        
        # Adam numpizized
        Ms = (learning_rate / np.sqrt(Vht) + epsilon) * Mht
        
        coeffs = [coeffs[i] - Ms[i] for i in range(degree+1)]
        #print(error, coeffs)
        if i % 100 == 0:
            print(error)
        #print(Y)
        
        if error < 0.000001:
            break
        
        if any(np.isnan(coeffs)):
            print("[ERROR] Got NaN, try another function or supply more points")
            exit()
        
        if error == lastError:
            break
            
        lastError = error

    return Polynomial(coeffs, degree, X, Y)


def plotFunction(func, r=(0, 12), points=True):
    x = np.linspace(r[0], r[1], 51)
    y = func(x)
    
    plt.plot(x, y, label=func.prettifyLatex())
    
    if func.x != None and func.y != None:
        plt.plot(func.x, func.y, 'o')
    
    plt.grid()
    plt.legend()



def f(x):
    return 2*x + np.sin(x*10**16)*10

def f2(x):
    return 2*x**2 - 5*x + 2

def f3(x):
    return 4*x**3 - 2*x**2 - 5*x + 2

def f4(x):
    return 2*5**x

def f5(x):
    return 6*x**3 + 2*x**2 + 5*x + 2

if __name__ == "__main__":
    testx = np.linspace(-10, 10, 15)
    #testx = list(range(12))
    #testy = [42572, 47474, 51315, 53043, 54112, 60528, 70585, 75781, 76281, 78861, 83304, 92213]
    testy = f5(testx)
    
    #a = parsePolynomial("x^2 - 2ab")
    #print(a.prettify())
    
    #a = Equation("2ab^5 + 2a - b = 16")
    #b = Equation("a - b = 2")
    #print("Solving", eq.prettify())
    #answer = solveEquations([a, b])
    #print(answer)
    
    plotFunction(polynomial(3, testx, testy, rate=10000000000))
    #plotFunction(polynomial(3, testx, testy))
    plt.show()

