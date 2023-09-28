import pandas as pd

labels_df = pd.read_csv('labels.csv')

data_df = pd.read_csv('data.csv')

# Create a function that calculates the euclidean distance between two points
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i])**2
    return distance**0.5

# Create a function that calculates the Mahalanobis distance between two points
def mahalanobis_distance(point1, point2, covariance_matrix):
    distance = 0
    for i in range(len(point1)):
        for j in range(len(point1)):
            distance += (point1[i] - point2[i]) * (point1[j] - point2[j]) * covariance_matrix[i][j]
    return distance**0.5

# Create a function that calculates the cosine distance between two points
def cosine_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += point1[i] * point2[i]
    return distance

# Create a function that calculate the distance intra class
def intra_class_distance(data, labels):
    distance = 0
    return distance
        
# Create a function that separates the data by class
def separate_data_by_class(data, labels):
    separated_data = {}
    print(labels[0][0])
    print(labels[1])
    print(labels[2])
    for i in range(len(data)):
        if labels[i] not in separated_data:
            separated_data[labels[i]] = []
        separated_data[labels[i]].append(data[i])
    return separated_data

separatedData = separate_data_by_class(data_df.values, labels_df.values)
print(separatedData)
