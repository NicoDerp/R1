
import matplotlib.pyplot as plt
import numpy as np
import random


class Function:
    def __init__(self, f, coeffs, x, y):
        self.f = f
        self.coeffs = coeffs
        self.x = x
        self.y = y
    
    def prettify(self):
        return "ooga booga"
    
    def __call__(self, x):
        return self.f(x)

class Polynomial(Function):
    def __init__(self, coeffs, degree, x, y):
        self.f = lambda x: sum([coeffs[i]*x**i for i in range(degree+1)])
        super().__init__(self.f, coeffs, x, y)
        self.degree = degree
    
    def prettify(self):
        s = "$"
        for i in range(self.degree, -1, -1):
            c = self.coeffs[i]
            if c < 0:
                c = -c
                if i == self.degree:
                    s += "-"
                else:
                    s += " - "
            elif i != self.degree:
                s += " + "
    
            if i == 0:
                s += f"{c:.2f}"
            elif i == 1:
                s += f"{c:.2f}x"
            else:
                s += f"{c:.2f}x^{i}"
        s += "$"
        return s


def linear(X, Y, maxIterations=10000):
    a, b = [0, 0]
    learning_rate = 0.001
    m = float(len(Y))
    for i in range(maxIterations):
        predicted = [a*x + b for x in X]
        
        Da = -2 * sum(X * (Y - predicted)) / m
        Db = -2 * sum(Y - predicted) / m
        
        error = sum((Y - predicted) ** 2) / m
        
        a -= learning_rate * Da
        b -= learning_rate * Db
        
        if i % 100 == 0:
            print(error, i)
        #print(Y)
        
        if error < 0.000001:
            break
    
    return Polynomial([b, a], 1, X, Y)


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

def polynomial(degree, X, Y, maxIterations=100000):
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
        
        if error < 0.0001:
            break
        
        if any(np.isnan(coeffs)):
            print("[ERROR] Got NaN, try another function or supply more points")
            exit()
        
        if error == lastError:
            break
            
        lastError = error

    return Polynomial(coeffs, degree, X, Y)


def plotFunction(func, points=True):
    x = np.linspace(-20, 20, 51)
    y = func(x)
    
    plt.plot(x, y, label=func.prettify())
    plt.plot(func.x, func.y, 'o')
    
    plt.grid()
    plt.legend()



def f(x):
    return 2*x + 1

def f2(x):
    return 2*x**2 - 5*x + 2

def f3(x):
    return 6*x**3 - 2*x**2 - 5*x + 2

testx = np.linspace(-10, 10, 11)
testy = f(testx)

#plotFunction(polynomial(3, testx, testy))
plotFunction(linear(testx, testy))
plt.show()

