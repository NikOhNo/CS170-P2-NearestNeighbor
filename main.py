import random
import numpy as np
import math

data = np.array([
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5]
])

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    return random.uniform(0, 100)

def forward_feature_search(data):

    current_set_of_features = [] # initialize an empty set

    # from column 2 (features) to end
    for i in range(1, data.shape[1]):
        print(f"On the {i}th level of the search tree")
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0

        # from column 2 to end
        for k in range(1, data.shape[1]):
            if (current_set_of_features.__contains__(k) == False):  # only add unconsidered features
                print(f"--Considering adding the {k} feature")
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k + 1)

                if (accuracy > best_so_far_accuracy):   # found a better accuracy feature
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        current_set_of_features.append(feature_to_add_at_this_level)
        print(f"On level {i} i added feature {feature_to_add_at_this_level} to current set")

# TODO: implement backward feature search
def backward_feature_search(data):
    return None

def accuracy():
    # load the data from filepath
    file_path = r"D:\Program Files\GitHub\CS170-P2-NearestNeighbor\CS170_Small_Data__12.txt"
    data = np.loadtxt(file_path)
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
    return accuracy

def main():
    accuracy()

if __name__ == "__main__":
    main()