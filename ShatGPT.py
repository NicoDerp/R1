import numpy
import numpy as np


def NoneActivation(x):
    return x


def dNoneActivation(x):
    return 1


def ReLU(x):
    return max(0, x)


def dReLU(x):
    return max(0, 1)


def Softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dSigmoid(x):
    s = Sigmoid(x)
    return s * (1 - s)


class Layer:
    def __init__(self, size, prevSize, activation):
        self.size = size
        self.prevSize = prevSize
        self.neurons = np.zeros(size)
        self.states = np.zeros(size)

        self.fWeights = np.random.rand(size, size+prevSize)
        self.iWeights = np.random.rand(size, size+prevSize)
        self.cWeights = np.random.rand(size, size+prevSize)
        self.oWeights = np.random.rand(size, size+prevSize)

        self.fBiases = np.zeros(size)
        self.iBiases = np.zeros(size)
        self.cBiases = np.zeros(size)
        self.oBiases = np.zeros(size)

        if activation == "none":
            self.activation = NoneActivation
            self.dActivation = dNoneActivation
        elif activation == "ReLU":
            self.activation = ReLU
            self.dActivation = dReLU


class LSTM:
    def __init__(self):
        self.inputSize = 2
        self.hiddenLayers = [Layer(4, self.inputSize, "none"),
                             Layer(2, 4, "none")]

    def feedForward(self, inputState):
        if inputState.shape != (self.inputSize,):
            print(f"[ERROR] Feed-forward input's shape is not ({self.inputSize}, 0) but {inputState.shape}")
            return

        layer = self.hiddenLayers[0]
        vt = np.concatenate((layer.states, inputState))
        # for state in layer.states:
        ft = Sigmoid(layer.fWeights.dot(vt) + layer.fBiases)
        it = Sigmoid(layer.iWeights.dot(vt) + layer.iBiases)
        ct = np.tanh(layer.cWeights.dot(vt) + layer.cBiases)
        ot = Sigmoid(layer.oWeights.dot(vt) + layer.oBiases)
        layer.neurons = ft * layer.neurons + it * ct
        layer.states = ot * np.tanh(layer.neurons)

        for i in range(1, len(self.hiddenLayers)):
            layer = self.hiddenLayers[i]
            vt = np.concatenate((layer.states, self.hiddenLayers[i-1].neurons))
            # for state in layer.states:
            ft = Sigmoid(layer.fWeights.dot(vt) + layer.fBiases)
            it = Sigmoid(layer.iWeights.dot(vt) + layer.iBiases)
            ct = np.tanh(layer.cWeights.dot(vt) + layer.cBiases)
            ot = Sigmoid(layer.oWeights.dot(vt) + layer.oBiases)
            layer.neurons = ft * layer.neurons + it * ct
            layer.states = ot * np.tanh(layer.neurons)

    def gradientDescent(self, actual):
        lossDerivative = (2.0/self.layers[-1].size) * (self.layers[-1].neurons - actual)
        errorL = lossDerivative * self.layers[-1].dActivation(self.layers[-1].zNeurons)

        # L-1 .. 0
        for i in range(self.layerCount-2, -1, -1):
            layer = self.layers[i]
            errorl = layer.dActivation()
        #self.layers[-2].weights -= 0.001 * errorL * self.layers[-2].neurons


ai = LSTM()

ai.feedForward(np.ones(ai.inputSize))
for layer in ai.hiddenLayers:
    print(layer.states)
    print(layer.neurons)
    print()
#ai.gradientDescent(np.array([1, 2, 3, 4]))

# print(a.neurons)
# print(a.weights.dot(a.neurons))

