
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
    def __init__(self, size, nextSize, activation):
        self.size = size
        self.neurons = np.zeros(size)
        self.biases = np.zeros(size)
        
        if nextSize != -1:
            # This to next
            self.xWeights = np.random.rand(nextSize, size)

            # This to this
            self.hWeights = np.random.rand(nextSize, nextSize)
            #self.weights = np.ones((next.size, size))
        
        if activation == "none":
            self.activation = NoneActivation
            self.dActivation = dNoneActivation
        elif activation == "ReLU":
            self.activation = ReLU
            self.dActivation = dReLU


class StackedRNN:
    def __init__(self):
        self.layers = [Layer(2,  4, "none"),
                       Layer(4,  2, "none"),
                       Layer(2, -1, "none")]
        self.layerCount = len(self.layers)

        self.layers[0].neurons = np.ones(self.layers[0].size)

        self.feedForward()
        print(self.layers[1].neurons)

    def feedForward(self):
        # 0 .. L-1
        for i in range(0, self.layerCount-1):
            curL = self.layers[i]
            nextL = self.layers[i+1]
            # nextL.zNeurons = curL.weights.dot(curL.neurons) + nextL.biases
            nextL.neurons = nextL.activation(curL.xWeights.dot(curL.neurons) + curL.hWeights.dot(nextL.neurons))

        # Calculate last layer which doesn't have hidden state
        self.layers[-1].neurons = nextL.activation(curL.xWeights.dot(curL.neurons) + curL.hWeights.dot(nextL.neurons))

    def gradientDescent(self, actual):
        lossDerivative = (2.0/self.layers[-1].size) * (self.layers[-1].neurons - actual)
        errorL = lossDerivative * self.layers[-1].dActivation(self.layers[-1].zNeurons)

        # L-1 .. 0
        for i in range(self.layerCount-2, -1, -1):
            layer = self.layers[i]
            errorl = layer.dActivation()
        #self.layers[-2].weights -= 0.001 * errorL * self.layers[-2].neurons


ai = StackedRNN()
#ai.gradientDescent(np.array([1, 2, 3, 4]))

# print(a.neurons)
# print(a.weights.dot(a.neurons))

