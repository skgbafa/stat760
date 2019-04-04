# Samuel Gbafa
# Homework 7
# April 3rd, 2019

import pandas as pd
import numpy as np
import math



# import data
def getData(trainFile):
    with open(trainFile) as csvfile:
        data = pd.read_csv(csvfile, '\t')
        # remove extra columns
        data.drop('train', axis=1, inplace=True)
        data.drop('Unnamed: 0', axis=1, inplace=True)
    return data

def getX(data, numFeatures):
    x = np.copy(data)
    # remove first column
    x[::, 0] = 1
    # keep n cols
    return x[::, :numFeatures + 1]

def getY(data, numFeatures):
    y = np.copy(data)
    # get last column
    last = y.shape[1]
    # keep n cols
    return y[::, last - 1:]

# normalize
def normalize(y):
    def std(x, mean):
        return (x + 0.4) / mean
    std = np.vectorize(std)
    return std(y, np.mean(y))

# sigmoid function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x): 
    return x * (1 - x)

# Xavier initializer
def xavierInit(inputCount, outputCount, bias):
    initialized = np.random.normal(0, 1/math.sqrt(inputCount), (inputCount, outputCount))
    if (bias):
        initialized = np.vstack([initialized, np.ones((1, outputCount))])
    return initialized

# feed forward
def feedForward(oHat, w1, w2, y):
    o1 = sigmoid(np.dot(oHat, w1))
    # add another node to o1
    o1 = np.concatenate((o1, np.ones((1,))))
    o2 = sigmoid(np.dot(o1, w2))

    error = y - o2
    D2 = sigmoidDerivative(o2)
    D1 = sigmoidDerivative(o1)
    return error, D1, D2

# back propgation
def backPropagate(D1, D2):
    dW1 = []
    dW2 = []
    return dW1, dW2

# train network
def trainNN(trainingData, testingData, initW1, initW2, numEpochs, learningRate):
    # for each epoch
    w1 = initW1
    w2 = initW2
    for epoch in range(numEpochs):
        rowIndex = 0
        # get a training example
        for row in trainingData:
            # feed forward => error, derivatives
            error, D1, D2 = feedForward(np.transpose(row), w1, w2, testingData)
            # back-propgate => gradient shift
            dW1, dW2 = backPropagate(D1, D2)
            # get correction
            rowIndex += 1
    # calculate epoch correction
    # update weights
    return w1, w2


if __name__ == "__main__":
    # import data
    dataFile = "./data/example.data"
    # dataFile = "./data/prostate.data"

    # import data
    data = getData(dataFile)

    # initialize hyperparameters
    epochs = 10
    numFeatures = 8
    hiddenNodes = 8
    learningRate = 5e-4
    bias = True

    # convert data to matrix
    matrixData = data.values

    # divide into training and test set
    testCount = int(matrixData.shape[0] * 0.2)
    trainingData = matrixData[testCount:]
    testingData = matrixData[:testCount]

    # get training data
    x = getX(trainingData, numFeatures)
    y = getY(trainingData, numFeatures)
    y = normalize(y)

    # initialize weights via Xavier initialization
    w1 = xavierInit(numFeatures, hiddenNodes, bias)
    w2 = xavierInit(hiddenNodes, 1, bias)

    print(w1.shape)
    print(w2.shape)

    # train network via backprop
    trainedW1, trainedW2 = trainNN(x, y, w1, w2, epochs, learningRate)

    # determine test error
