import time
import numpy as np

### Perform Leave one out cross validation (LOOCV)
def validation(features, clf, dataset, output_trace = False): #input a set of features and classifier
    #feature_set = [x-1 for x in features] #convert feature list to 0-index
    feature_set = features #feature list
    correct = 0  #correct predictions counter
    total = len(dataset[0]) #instances in dataset

    instances = dataset[0]  #feature data
    labels = dataset[1]     #labels data

    #select features only from validation
    instances = instances[:, feature_set] 

    #time variables
    inference_time = 0
    train_time = 0

    for i in range(total): #perform LOOCV- for each instance hold out to use as test data
        #split into training data except i-th instance
        train_data = np.concatenate((instances[:i], instances[i+1:]), axis = 0)
        train_labels = np.concatenate((labels[:i], labels[i+1:]), axis = 0)

        test_point = instances[i]
        test_label = labels[i]

        #measure time and for KNN training
        start = time.time()
        clf.train(train_data, train_labels)
        end = time.time()
        train_time += end - start

        start = time.time()
        pred_label = clf.getLabel(test_point)
        end = time.time()
        inference_time += end - start

        correct += (pred_label == test_label) #check if prediciton is correct

    if (output_trace):
        print(f" Average Training Time: {str(train_time/total)[:6]} seconds.")
        print(f"Average Inference Time: {str(inference_time/total)[:6]} seconds.")

    #return classification accuracy 
    return correct/total 