# Samuel Gbafa
# Homework 2
# 02/14/19

import csv
import json

# import data
def getData(trainFile):
    features = []
    with open(trainFile) as csvfile:
        reader = csv.DictReader(csvfile,  delimiter='	')
        for row in reader:
            features.append(row)
    return features

if __name__ == "__main__":
    trainFile = "./data/example.data"
    # import data
    data = getData(trainFile)
    print(json.dumps(data))
