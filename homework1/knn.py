from tqdm import tqdm
import json

# convert file to data
def lineToData(line):
    return tuple(list(map(float, filter(lambda x: x != '\n', line.split(' ')))))

# import data
def getData(trainFile, testFile):
    trainingData = list(map(lineToData, open(trainFile, 'r').readlines()))
    testingData = list(map(lineToData, open(testFile, 'r').readlines()))
    return { "training": trainingData, 'testing': testingData }

# get distances
def getDistances(trial, features):
    # get distance for each element in test set
    trialData = trial[1:]   # skip first element
    distances = []
    for feature in features:
        squaredSum = 0
        featureData = feature[1:]  # skip first element
        for index in range(len(trialData)):
            squaredSum += (trialData[index] - featureData[index]) ** 2  # get distance by sum of squares
        distances.append((feature[0], squaredSum))
    # sort distances
    distances.sort(key=lambda tup: tup[1])
    return distances
                   

# get knn
def getKNN(k, distances):
    return distances[:k] # sorted in previous step, return first k

# get majority of neighbors
def getMajority(neighbors):
    counts = dict()
    for neighbor in neighbors:
        index = neighbor[0]
        counts[index] = counts.get(index, 0) + 1
    return max(counts, key=counts.get)

# calulate results
def results(resultingData):
    total = len(resultingData)
    hit = 0
    miss = 0
    for result in resultingData:
        if int(result[0]) == int(result[1]):
            hit += 1
        else:
            miss += 1
    # print output
    output = {
        "total": total,
        "hit": hit,
        "miss": miss,
        "error rate": "%.3f%%" % float(miss/total * 100),
        "accuracy": "%.3f%%" % float(hit/total * 100)
    }
    print(json.dumps(output))

if __name__ == "__main__":
    k = 9
    # trainFile = "./data/example.train"
    # testFile = "./data/example.test"
    trainFile = "./data/zip.train"
    testFile = "./data/zip.test"
    resultingData = []

    # get data
    data = getData(trainFile, testFile)
    # print(data["testing"][0])  # distance example

    # get distances
    for trial in tqdm(data["testing"]):
        # get distances for trial
        distances = getDistances(trial, data["training"])
        # get k nearest neighbors
        neighbors = getKNN(k, distances)
        # identify by neighbors
        hypothesis = getMajority(neighbors)
        # store actual and guess
        resultingData.append((trial[0], hypothesis))
    
    # calulate results
    results(resultingData)
        


