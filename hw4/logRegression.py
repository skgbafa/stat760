# Samuel Gbafa
# Homework 4
# 02/28/19

import csv
import json
import pandas as pd
import numpy as np

# import data
def getData(trainFile):
    with open(trainFile) as csvfile:
        data = pd.read_csv(csvfile)
        data['famhist'] = data['famhist'].map({'Present': 1, 'Absent': 0})
        data.describe()
    return data


def initialBeta(numFeatures):
    # return matrix with 1 column and n + 1 features
    return np.zeros((numFeatures + 1, 1), dtype=np.int)


def getX(data, numFeatures):
    x = np.copy(data)
    # replace first column with 1
    x[::, 0] = 1
    # keep n cols
    return x[::, :numFeatures + 1]


def getY(data, numFeatures):
    y = np.copy(data)
    # get last column
    last = y.shape[1]
    # keep n cols
    return y[::, last - 1:]

def gradientLogBeta(x, y, beta):
    # 
    # return sum
    return None

def maximizeBeta(x, y, seedBeta, alpha, interations):
    # initialize beta
    currentBeta = seedBeta
    newBeta = currentBeta
    currentIteration = 0
    
    # while not max iterations or gradient diff is small
    while currentIteration < interations:
        # get current gradient
        gradientStep = alpha * gradientLogBeta(x, y, currentBeta)
        # update beta
        newBeta = currentBeta + gradientStep
        # increment
        currentBeta = newBeta
        currentIteration += 1
        
    # return initial beta
    return newBeta

def testModel(beta, x):
    return None

if __name__ == "__main__":
    dataFile = "./data/example.data"
    # dataFile = "./data/SAheart.data"

    # import data
    data = getData(dataFile)
    
    # convert data to matrix
    matrixData = data.values
    # get test and training data
    testCount = int(matrixData.shape[0] * 0)
    trainingData = matrixData[testCount:]
    testingData = matrixData[:testCount]

    # program params
    numFeatures = 1
    interations = 10

    # regression params
    alpha = 1e-7
    
    # initialize beta
    seedBeta = initialBeta(numFeatures)

    # get training data
    x = getX(trainingData, numFeatures)
    y = getY(trainingData, numFeatures)

    # gradient decent
    maxBeta = maximizeBeta(x, y, seedBeta, alpha, interations)

    # # test model
    # testX = getX(testingData, numFeatures)
    # error = testModel(maxBeta, testX)

    # # plot error
    # # todo... in loop





