
import numpy as np


def NoneActivation(x):
    return x

def dNoneActivation(x):
    return 1

def ReLU(x):
    return max(0, x)

def dReLU(x):
    return max(0, 1)

class Layer:
    def __init__(self, size, next, activation):
        self.size = size
        self.next = next
        self.neurons = np.zeros(size)
        self.oldNeurons = np.zeros(size)
        self.biases = np.zeros(size)
        
        if next:
            self.weights = np.random.rand(next.size, size)
            #self.weights = np.ones((next.size, size))
        
        if activation == "none":
            self.activation = NoneActivation
            self.dActivation = dNoneActivation
        elif activation == "ReLU":
            self.activation = ReLU
            self.dActivation = dReLU

    def feedForward(self):
        if not self.next:
            return
        
        self.next.neurons = self.activation(self.weights.dot(self.neurons) + self.next.biases)
        self.next.oldNeurons = self.next.neurons


b = Layer(4, None, "none")
a = Layer(6, b, "none")

a.neurons = np.ones(a.size)

a.feedForward()
print(b.neurons)

#print(a.neurons)
#print(a.weights.dot(a.neurons))

