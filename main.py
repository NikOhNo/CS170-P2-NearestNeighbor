import random
import numpy as np
import time
import math

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    # features we care about are label, current_set, and feature_to_add
    if current_set is None:
        feature_indices = [0]
        feature_display = []
    elif feature_to_add is None:
        feature_indices = [0] + current_set
        feature_display = current_set
    else:
        feature_indices = [0] + current_set + [feature_to_add]
        feature_display = current_set + [feature_to_add]

    subsetData = data[:, feature_indices]
    subsetAccuracy = accuracy(subsetData)
    print(f"\tUsing feature(s) {curly_list(feature_display)} accuracy is {subsetAccuracy:.1f}%")
    return subsetAccuracy

def calculate_default_rate(data):
    labels = data[:, 0]     # Extract the labels from the 0th column
    number_of_defaults = np.sum(labels == 2)
    total_points = data.shape[0]

    # Compute the default rate as a percentage
    default_rate = (number_of_defaults / total_points) * 100
    return default_rate

def forward_feature_search(data):
    print_data_info(data)

    startTime = time.time()

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
        print(f"Feature set {curly_list(current_set_of_features)} was best, accuracy is {best_so_far_accuracy:.1f}%\n")
        # check if its our new overall best
        if (best_so_far_accuracy > best_overall_accuracy):
            best_overall_accuracy = best_so_far_accuracy
            best_overall_feature_set = current_set_of_features.copy()

    endTime = time.time()
    print(f"Finished search!! The best feature subset is {curly_list(best_overall_feature_set)}, which has an accuracy of {best_overall_accuracy:.1f}%")
    elapsed_time = endTime - startTime
    print_time(elapsed_time)

# TODO: implement backward feature search
def backward_feature_search(data):
    print_data_info(data)

    startTime = time.time()

    current_set_of_features = [] # initialize a set with all features
    for i in range(1, data.shape[1]):
        current_set_of_features.append(i)

    best_overall_feature_set = []
    best_overall_accuracy = 0

    # from column 2 (start of features) to end
    for i in range(1, data.shape[1]):
        #print(f"On the {i}th level of the search tree")
        feature_to_exclude = None
        best_so_far_accuracy = 0

        # from column 2 to end
        for k in range(1, data.shape[1]):
            if (current_set_of_features.__contains__(k) == True):  # only add unconsidered features
                copy_current_set = current_set_of_features.copy()
                # remove the feature from copy and test
                copy_current_set.remove(k)  
                accuracy = leave_one_out_cross_validation(data, copy_current_set, None)

                if (accuracy > best_so_far_accuracy):   # found a better accuracy feature
                    best_so_far_accuracy = accuracy
                    feature_to_exclude = k
        
        current_set_of_features.remove(feature_to_exclude)

        print("")
        if (best_so_far_accuracy < best_overall_accuracy):
            print(f"(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        print(f"Feature set {curly_list(current_set_of_features)} was best, accuracy is {best_so_far_accuracy:.1f}%\n")
        # check if its our new overall best
        if (best_so_far_accuracy > best_overall_accuracy):
            best_overall_accuracy = best_so_far_accuracy
            best_overall_feature_set = current_set_of_features.copy()
    
    endTime = time.time()
    print(f"Finished search!! The best feature subset is {curly_list(best_overall_feature_set)}, which has an accuracy of {best_overall_accuracy:.1f}%")
    elapsed_time = endTime - startTime
    print_time(elapsed_time)
    print("")

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
    print(f"Running nearest neighbor with all {cols - 1} features, using \"leaving-one-out\" evaluation, I get an accuracy of {allAccuracy:.1f}%")
    default_rate = calculate_default_rate(data)
    print(f"Default rate of dataset is {default_rate:.1f}%\n")
    print(f"Beginning search. \n")

def print_time(seconds):
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    time_parts = []
    if days > 0:
        time_parts.append(f"{int(days)} days")
    if hours > 0:
        time_parts.append(f"{int(hours)} hours")
    if minutes > 0:
        time_parts.append(f"{int(minutes)} minutes")
    if seconds > 0 or (days == 0 and hours == 0 and minutes == 0):
        time_parts.append(f"{int(seconds)} seconds")

    print(f"Elapsed Time: {', '.join(time_parts)}\n")

def curly_list(lst):
    return f"{{{', '.join(map(str, lst))}}}"

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