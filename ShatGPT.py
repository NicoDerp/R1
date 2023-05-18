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
    def __init__(self, layerType, size, activation):
        self.layerType = layerType
        self.size = size
        self.output = np.zeros(size)
        self.prev = None

        if activation == "none":
            self.activation = NoneActivation
            self.dActivation = dNoneActivation
        elif activation == "ReLU":
            self.activation = ReLU
            self.dActivation = dReLU

    def feedForward(self):
        pass

    def setup_(self, prev):
        self.prev = prev


class InputLayer(Layer):
    def __init__(self, size):
        super().__init__("InputLayer", size, "")


class FFLayer(Layer):
    def __init__(self, size, activation):
        super().__init__("FFLayer", size, activation)

        self.weights = None
        self.biases = np.zeros(self.size)

    def feedForward(self):
        self.output = self.activation(self.weights.dot(self.prev.output) + self.biases)

    def setup_(self, prev):
        super().setup_(prev)

        self.weights = np.random.rand(self.size, self.prev.size)


class LSTMLayer(Layer):
    def __init__(self, size, activation):
        super().__init__("FFLayer", size, activation)
        self.fWeights = None
        self.iWeights = None
        self.cWeights = None
        self.oWeights = None

        self.fBiases = np.zeros(self.size)
        self.iBiases = np.zeros(self.size)
        self.cBiases = np.zeros(self.size)
        self.oBiases = np.zeros(self.size)
        self.states = np.zeros(size)

    def feedForward(self):
        vt = np.concatenate((self.states, self.prev.output))

        ft = Sigmoid(self.fWeights.dot(vt) + self.fBiases)
        it = Sigmoid(self.iWeights.dot(vt) + self.iBiases)
        ct = np.tanh(self.cWeights.dot(vt) + self.cBiases)
        ot = Sigmoid(self.oWeights.dot(vt) + self.oBiases)

        self.output = ft * self.output + it * ct
        self.states = ot * np.tanh(self.output)

    def setup_(self, prev):
        super().setup_(prev)

        self.fWeights = np.random.rand(self.size, self.size + self.prev.size)
        self.iWeights = np.random.rand(self.size, self.size + self.prev.size)
        self.cWeights = np.random.rand(self.size, self.size + self.prev.size)
        self.oWeights = np.random.rand(self.size, self.size + self.prev.size)


class AI:
    def __init__(self):
        self.layers = [InputLayer(3),
                       LSTMLayer(4, "none"),
                       FFLayer(2, "none")]

        self.setupLayers()

    def setupLayers(self):
        if len(self.layers) < 3:
            print(f"[ERROR] At least 3 layers are required")
            return

        if self.layers[0].layerType != "InputLayer":
            print(f"[ERROR] First layer isn't InputLayer")
            return

        self.layers[0].setup_(None)
        for i in range(1, len(self.layers)):
            self.layers[i].setup_(self.layers[i - 1])

    def feedForward(self, inputState):
        if inputState.shape != self.layers[0].output.shape:
            print(f"[ERROR] Feed-forward input's shape is not {self.layers[0].output.shape} but {inputState.shape}")
            return

        self.layers[0].output = inputState
        for i in range(1, len(self.layers)):
            self.layers[i].feedForward()

    def gradientDescent(self, actual):
        lossDerivative = (2.0/self.layers[-1].size) * (self.layers[-1].neurons - actual)
        errorL = lossDerivative * self.layers[-1].dActivation(self.layers[-1].zNeurons)

        # L-1 .. 0
        for i in range(self.layerCount-2, -1, -1):
            layer = self.layers[i]
            errorl = layer.dActivation()
        #self.layers[-2].weights -= 0.001 * errorL * self.layers[-2].neurons


ai = AI()

ai.feedForward(np.ones(ai.layers[0].size))
for layer in ai.layers:
    if layer.layerType == "LSTMLayer":
        print(layer.states)
        print(layer.output)
    else:
        print(layer.output)
    print()
#ai.gradientDescent(np.array([1, 2, 3, 4]))

# print(a.neurons)
# print(a.weights.dot(a.neurons))

