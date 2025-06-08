import time
import csv
import json
from Tree import Tree


def main():
    print("\n\nWelcome to Harrison Feature's Selection Algorithm.\n")
    print("\nChoose the Dataset:\n1: \tSmall Dataset #18\n2: \tLarge Dataset #10\n3: \tDiabetes Dataset\n")
    choice = input("Enter your choice (1 or 2 or 3): ")


    if   (choice == '1'):
        file = 'CS205_small_Data__18.txt'
    elif (choice == '2'):
        file = 'CS205_large_Data__10.txt'
    elif (choice == '3'):
        file = 'diabetes_data.txt'
    else:
        print("Invalid choice.")
        return

    print("\nChoose the feature selection method:\n1: \tForward Selection\n2: \tBackward Elimination")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        print("\nForward Selection chosen.")
        tree = Tree(data_file = file, method='forward')
        start_time = time.perf_counter()
        tree.forward_selection()
        end_time  = time.perf_counter()
    elif choice == '2':
        print("\nBackward Elimination chosen.")
        tree = Tree(data_file = file, method='backward')
        start_time = time.perf_counter()
        tree.backward_elimination()
        end_time  = time.perf_counter()
    else:
        print("\nInvalid choice. Exiting.")
        return

    print(f"\nFinished Search! The best feature subset is {tree.best_feature_set}, with an accuracy of {tree.best_score * 100:.1f}%.\n")
    print(f"Feature selection took {end_time - start_time:.2f} seconds.\n")

    #log best feature subsets and accurarcies for plotting
    log_filename = "accuracy_log.csv"
    with open(log_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Feature_Set", "Accuracy"])
        for feature_set, acc in tree.accuracyLog:
            writer.writerow([json.dumps(feature_set), acc])

if __name__ == "__main__":
    main()