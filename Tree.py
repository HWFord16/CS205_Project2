from Node import Node
from Classifier  import read_data


class Tree:
    # need to pass in features (and later dataset too) to tree class
    def __init__(self, data_file, method = "forward"):

      self.dataset = read_data(data_file, normal = True)
      self.num_instances = self.dataset[0].shape[0]
      self.num_features = self.dataset[0].shape[1]
      feature_set = [i for i in range(self.num_features)]
      self.accuracyLog= []

      if method == "forward":
        self.root = Node(current_features= [], features=feature_set)
      else:
        self.root = Node(current_features= feature_set, features=feature_set)

      self.best_feature_set = None  #tstore the best feature set found
      self.best_score = -1  #keep track of the best score found
      
    #use current_features & dataset to evaluate later on
    def evaluate(self, tree_node): 
        return tree_node.evaluate(dataset = self.dataset)

    def backward_elimination(self):
        n = self.root
        initial_score = self.evaluate(n)  #predicting an initial accuracy
        self.accuracyLog.append((n.current_features, initial_score))
        print(f"\nThis dataset has {self.num_features} features (not including class attribute), with {self.num_instances} instances.")
        print(f"\nRunning Nearest Neighbor with all features and using LOOCV gets an accuracy of {initial_score*100:.1f}%")
        print("\n====Beginning search====\n")

        #continue removing features until only one is left
        while len(n.current_features) > 1:
            n.contract()
            best_child = None

            for child in n.children:
                child.score = self.evaluate(child)  #evaluate each new configuration after contraction
                print(f"Using feature {child.current_features} accuracy is {child.score * 100:.1f}%")

                if best_child is None or child.score > best_child.score:
                    best_child = child

            #update the best feature set if the new configuration is better
            if best_child:
                self.accuracyLog.append((best_child.current_features, best_child.score))
                if best_child.score > self.best_score:
                  self.best_score = best_child.score
                  self.best_feature_set = best_child.current_features
                  print(f"\nFeature set {best_child.current_features} was best, accuracy is {best_child.score *100 :.1f}%\n")

                n = best_child  #move to the best child node for further reduction
            else:
                break
        return self.best_feature_set

    def forward_selection(self):
        current_node = self.root
        initial_score = self.evaluate(current_node)  #prediciting an initial accuracy
        self.accuracyLog.append((current_node.current_features, initial_score))
        print(f"\nThis dataset has {self.num_features} features (not including class attribute), with {self.num_instances} instances.")
        print(f"\nRunning Nearest Neighbor with no features and using LOOCV gets an accuracy of {initial_score*100:.1f}%")
        print("\n====Beginning search====\n")

        #start with no features and expand by adding features
        while len(current_node.current_features) < len(current_node.features):
            current_node.expand()
            best_child = None

            for child in current_node.children:
                child.score = self.evaluate(child)  #evaluate the addition of each new feature
                print(f"Using feature(s) {child.current_features} accuracy is {child.score * 100 :.1f}%")

                if best_child is None or child.score > best_child.score:
                    best_child = child
            
            if best_child: #log the best child found at the depth
                self.accuracyLog.append((best_child.current_features, best_child.score))
                if best_child.score > self.best_score:
                    self.best_score = best_child.score
                    self.best_feature_set = best_child.current_features
                    print(f"\nFeature set {best_child.current_features} was best, accuracy is {best_child.score * 100:.1f}%\n")

            #move to the best child for further expansion
            if best_child:
                current_node = best_child
            else:
                break
        return self.best_feature_set