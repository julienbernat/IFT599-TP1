import pandas as pd

labels_df = pd.read_csv('../labels.csv')

data_df = pd.read_csv('../data.csv')

# Create a function that calculates the euclidean distance between two vectors
def euclidean_distance(vector1, vector2):
    distance = 0
    for i in range(1,len(vector1)):
        distance += (vector1[i] - vector2[i])**2
    return distance**0.5

# Create a function that calculates the Mahalanobis distance between two vectors
def mahalanobis_distance(vector1, vector2, covariance_matrix):
    distance = 0
    for i in range(1,len(vector1)):
        distance += (vector1[i] - vector2[i])**2 / covariance_matrix[i][i]
    return distance**0.5

# Create a function that calculates the cosine distance between two vectors
def cosine_distance(vector1, vector2):
    distance = 0
    for i in range(1,len(vector1)):
        distance += vector1[i] * vector2[i]
    return distance

# Create a function that calculate the distance intra class
# The intra class distance is the maximum distance between two points of the same class
def intra_class_distance_euclidean(separated_data):
    intra_class_distance = {}
    for class_label in separated_data:
        max_distance = 0
        for i in range(1, len(separated_data[class_label])):
            for j in range(i+1, len(separated_data[class_label])):
                distance = euclidean_distance(separated_data[class_label][i], separated_data[class_label][j])
                if distance > max_distance:
                    max_distance = distance
        intra_class_distance[class_label] = max_distance
    return intra_class_distance
        
# Create a function that separates the data by class
def separate_data_by_class(data, labels):
    separated_data = {}
    for i in range(1, len(data)):
        class_label = labels[i][1]
        if class_label not in separated_data:
            separated_data[class_label] = []  # Initialize an empty list for each class label if it doesn't exist
        separated_data[class_label].append(data[i])
    return separated_data

separatedData = separate_data_by_class(data_df.values, labels_df.values)
intraClassEuclidean = intra_class_distance_euclidean(separatedData)
print(intraClassEuclidean)

