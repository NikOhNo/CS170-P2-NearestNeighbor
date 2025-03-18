import random
import numpy as np

data = np.array([
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5]
])

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    return random.uniform(0, 100)

def forward_feature_search(data):

    current_set_of_features = [] # initialize an empty set

    for i in range(1, data.shape[1]):
        print(f"On the {i}th level of the search tree")
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0

        for k in range(1, data.shape[1]):
            if (current_set_of_features.__contains__(k) == False):
                print(f"--Considering adding the {k} feature")
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k + 1)

                if (accuracy > best_so_far_accuracy):
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        current_set_of_features.append(feature_to_add_at_this_level)
        print(f"On level {i} i added feature {feature_to_add_at_this_level} to current set")

def main():
    forward_feature_search(data)

if __name__ == "__main__":
    main()