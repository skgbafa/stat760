# Samuel Gbafa
# Homework 2
# 02/14/19

import csv
import json
import pandas as pd

# import data
def getData(trainFile):
    features = []
    with open(trainFile) as csvfile:
        reader = csv.DictReader(csvfile,  delimiter='	')
        for row in reader:
            # data cleanup
            row.pop("", None)
            for key in row.keys():
                row[key] = float(row[key]) if (row[key] != 'T' and row[key] != 'F') else row[key]
            features.append(row)
        # convert to pandas dataframe 
        keys = list(features[0].keys())
        data = pd.DataFrame(features, columns=keys)
    return data

def getStats(data):
    # calc stats
    print(data.describe())
    return {}

if __name__ == "__main__":
    # trainFile = "./data/example.data"
    trainFile = "./data/prostate.data"

    # import data
    data = getData(trainFile)
    getStats(data)


# {
#     "": "1",
#     "lcavol": "-0.579818495",
#     "lweight": "2.769459",
#     "age": "50",
#     "lbph": "-1.38629436",
#     "svi": "0",
#     "lcp": "-1.38629436",
#     "gleason": "6",
#     "pgg45": "  0",
#     "lpsa": "-0.4307829",
#     "train": "T"
# }
