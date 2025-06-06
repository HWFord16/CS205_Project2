from Tree import Tree


def main():
    print("\n\nWelcome to Harrison Feature's Selection Algorithm.\n")
    print("\nChoose the Dataset:\n1: \tSmall Dataset #18\n2: \tLarge Dataset #10")
    choice = input("Enter your choice (1 or 2): ")


    if   (choice == '1'):
        file = 'CS205_small_Data__18.txt'
    elif (choice == '2'):
        file = 'CS205_large_Data__10.txt'
    else:
        print("Invalid choice.")
        return

    print("\nChoose the feature selection method:\n1: \tForward Selection\n2: \tBackward Elimination")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        print("\nForward Selection chosen.")
        tree = Tree(data_file = file, method='forward')
        tree.forward_selection()
    elif choice == '2':
        print("\nBackward Elimination chosen.")
        tree = Tree(data_file = file, method='backward')
        tree.backward_elimination()
    else:
        print("\nInvalid choice. Exiting.")
        return

    print(f"\nFinished Search! The best feature subset is {tree.best_feature_set}, with an accuracy of {tree.best_score * 100:.1f}%.\n\n")

if __name__ == "__main__":
    main()