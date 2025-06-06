from Validation import validation
from Classifier import NNClassifier

class Node:
    def __init__(self, parent = None, score = None, current_features= [], features= []):
        self.parent = parent
        self.children = []

        # set of features that are currently being used & remaining features to use
        self.current_features = current_features
        self.features = features

    def evaluate(self, dataset):
        return validation(features = self.current_features,
                          clf = NNClassifier(),
                          dataset = dataset,
                          output_trace = False)

    #expand this node by creating a child node for each feature not already included
    def expand(self):
        for feature in self.features:
            if feature not in self.current_features:
                #create a new node with the added feature and add it to the children list
                new_child = Node(current_features=self.current_features + [feature], features=self.features)
                self.children.append(new_child)

    #contract this node by creating a child node for each feature removed
    def contract(self):
      for i in range(len(self.current_features)):
        #create new list of current features minus the current index feature
        new_features = self.current_features[:i] + self.current_features[i+1:]
        new_child = Node(parent=self, current_features=new_features, features=self.features)
        self.children += [new_child]