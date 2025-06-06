import numpy as np

#implementation of K-NN classifier (k=1 ; 1-NN)
class NNClassifier:
    def __init__(self):
        self.trainData = None #storing training features
        self.trainLabels = None #corresponding labels

    #training to memorize the data for NN algo to use
    def train(self, data, labels): 
        self.trainData = data
        self.trainLabels = labels

    #gets label of nearest point by calculating Euclidean distance
    def getLabel(self, instance):
        dist = np.sqrt(((self.trainData - instance) ** 2).sum(axis=1))
        nearest_point = np.argmin(dist) #get shortest distance
        return self.trainLabels[nearest_point]


#normalize data/features 
def normalize(features):
    min = np.min(features, axis=0) #minimum per column
    max = np.max(features, axis=0) #max per column
    features = (features - min) / (max - min) #scale range[0,1]
    return features 

#read dataset file from input and return features & labels
def read_data(filename, normal = True):
    data = np.loadtxt(filename) #load text file as np.array

    #separate label and features in data array 
    labels = data[:, 0]
    features = data[:, 1:]

    #normalize features before returning
    if (normal): return normalize(features), labels
    return features, labels