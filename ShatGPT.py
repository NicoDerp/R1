
import numpy as np


def NoneActivation(x):
    return x


def dNoneActivation(x):
    return 1


def ReLU(x):
    return max(0, x)


def dReLU(x):
    return max(0, 1)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


class Layer:
    def __init__(self, size, prevSize, activation):
        self.size = size
        self.prevSize = prevSize
        self.state = np.zeros(size)
        self.biases = np.zeros(size)

        # Prev to this
        self.xWeights = np.random.rand(size, prevSize)

        # This to this
        self.hWeights = np.random.rand(size, prevSize)
        #self.weights = np.ones((next.size, size))
        
        if activation == "none":
            self.activation = NoneActivation
            self.dActivation = dNoneActivation
        elif activation == "ReLU":
            self.activation = ReLU
            self.dActivation = dReLU


class LSTM:
    def __init__(self):
        self.inputSize = 2
        self.layers = [Layer(4, self.inputSize, "none"),
                       Layer(5, 4, "none"),
                       Layer(2, 5, "none")]
        self.layerCount = len(self.layers)

    def feedForward(self, inputState):
        if inputState.shape != (self.inputSize,):
            print(f"[ERROR] Feed-forward input's shape is not ({self.inputSize}, 0) but {inputState.shape}")
            return

        curL = self.layers[0]
        print(curL.hWeights.dot(inputState))
        curL.state = curL.activation(curL.xWeights.dot(inputState) + curL.hWeights.dot(inputState) + curL.biases)

        # 1 .. L-1
        for i in range(1, self.layerCount-1):
            prevL = self.layers[i-1]
            curL = self.layers[i]
            # nextL.zNeurons = curL.weights.dot(curL.neurons) + nextL.biases
            curL.state = curL.activation(curL.xWeights.dot(prevL.state) + curL.hWeights.dot(prevL.state) + curL.biases)

        # Calculate last layer which doesn't have hidden state
        #self.layers[-1].state = nextL.activation(curL.xWeights.dot(curL.state) + curL.hWeights.dot(nextL.neurons))

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
print(ai.layers[0].state)
#ai.gradientDescent(np.array([1, 2, 3, 4]))

# print(a.neurons)
# print(a.weights.dot(a.neurons))

