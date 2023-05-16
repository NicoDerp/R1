
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
    def __init__(self, size, prev, next, activation):
        self.size = size
        self.prev = prev
        self.next = next
        self.neurons = np.zeros(size)
        self.zNeurons = np.zeros(size)
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
        
        self.next.zNeurons = self.weights.dot(self.neurons) + self.next.biases
        self.next.neurons = self.next.activation(self.next.zNeurons)
        self.next.neurons = self.next.activation(self.prev.weights*)
        self.next.oldNeurons = self.next.neurons


class StackedRNN:
    def __init__(self):
        self.layers = []
        self.layers.insert(0, Layer(2, None, "none"))
        self.layers.insert(0, Layer(4, self.layers[0], "none"))
        self.layers.insert(0, Layer(2, self.layers[0], "none"))
        self.layerCount = len(self.layers)

        self.layers[0].neurons = np.ones(self.layers[0].size)

        self.layers[0].feedForward()
        print(self.layers[1].neurons)

    def gradientDescent(self, actual):
        lossDerivative = (2.0/self.layers[-1].size) * (self.layers[-1].neurons - actual)
        errorL = lossDerivative * self.layers[-1].dActivation(self.layers[-1].zNeurons)

        # L-1 .. 0
        for i in range(self.layerCount-2, -1, -1):
            layer = self.layers[i]
            errorl = layer.dActivation()
        #self.layers[-2].weights -= 0.001 * errorL * self.layers[-2].neurons


ai = StackedRNN()
ai.gradientDescent(np.array([1, 2, 3, 4]))

# print(a.neurons)
# print(a.weights.dot(a.neurons))

