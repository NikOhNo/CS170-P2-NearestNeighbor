import random
import numpy as np
import math

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    # features we care about are label, current_set, and feature_to_add
    feature_indices = [0] + current_set + [feature_to_add]
    subsetData = data[:, feature_indices]
    subsetAccuracy = accuracy(subsetData)
    print(f"\tUsing feature(s) {current_set + [feature_to_add]} accuracy is {subsetAccuracy:.1f}%")
    return subsetAccuracy

def forward_feature_search(data):
    print_data_info(data)

    current_set_of_features = [] # initialize an empty set

    best_overall_feature_set = []
    best_overall_accuracy = 0

    # from column 2 (start of features) to end
    for i in range(1, data.shape[1]):
        #print(f"On the {i}th level of the search tree")
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0

        # from column 2 to end
        for k in range(1, data.shape[1]):
            if (current_set_of_features.__contains__(k) == False):  # only add unconsidered features
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)

                if (accuracy > best_so_far_accuracy):   # found a better accuracy feature
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        current_set_of_features.append(feature_to_add_at_this_level)

        print("")
        if (best_so_far_accuracy < best_overall_accuracy):
            print(f"(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        print(f"Feature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy:.1f}%\n")
        # check if its our new overall best
        if (best_so_far_accuracy > best_overall_accuracy):
            best_overall_accuracy = best_so_far_accuracy
            best_overall_feature_set = current_set_of_features.copy()
    
    print(f"Finished search!! The best feature subset is {best_overall_feature_set}, which has an accuracy of {best_overall_accuracy:.1f}%\n")

# TODO: implement backward feature search
def backward_feature_search(data):
    return None

def accuracy(data):
    number_correctly_classified = 0

    # for all rows (data points)
    for i in range(0, data.shape[0]):
        object_to_classify = data[i, 1:]        # gets data from 2nd column to end
        label_object_to_classify = data[i, 0]   # gets the label from 1st column

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        # for all rows
        for k in range(0, data.shape[0]):
            if (k != i):
                # calculate distance
                difference = object_to_classify - data[k, 1:]
                distance = math.sqrt(np.sum(difference ** 2))
                # if better distance, update nearest neighbor
                if (distance < nearest_neighbor_distance):
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location, 0]
        if (label_object_to_classify == nearest_neighbor_label):
            number_correctly_classified += 1
        
    # accuracy = # correct / # data points
    accuracy = number_correctly_classified / data.shape[0]
    return accuracy * 100

def print_data_info(data):
    rows, cols = data.shape
    print(f"This dataset has {cols - 1} features (not including the class attribute), with {rows} instances.\n")
    allAccuracy = accuracy(data)
    print(f"Running nearest neighbor with all {cols - 1} features, using \"leaving-one-out\" evaluation, I get an accuracy of {allAccuracy:.1f}%\n")


def select_data():
    testFile = input(f"Type in the name of the file to test : ")

    try:
        data = np.loadtxt(testFile) 
        return data

    except FileNotFoundError:
        print(f"Error: The file '{testFile}' was not found. Please check the file name and try again.")
        return select_data()
    except ValueError:
        print(f"Error: The data in '{testFile}' has an incorrect format. Please check the file.")
        return select_data()
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return select_data()
    
def select_algorithm(data):
    print(f"Type the number of the algorithm you want to run.")
    print(f"\t1) Forward Selection")
    print(f"\t2) Backward Elimination")
    algorithm = input()
    print("")

    if (algorithm == "1"):
        forward_feature_search(data)
    elif (algorithm == "2"):
        backward_feature_search(data)
    else:
        print(f"Unrecognized algorithm entered")
        select_algorithm(data)

def main():
    print(f"Welcome to Niko Udria's Feature Selection Algorithm.")
    # comments for easy filename copy paste
    # CS170_Large_Data__62.txt
    # CS170_Small_Data__12.txt
    data = select_data()
    select_algorithm(data)

if __name__ == "__main__":
    main()