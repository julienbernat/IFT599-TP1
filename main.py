import pandas as pd
import scipy
import numpy as np

labels_df = pd.read_csv('../labels.csv')

data_df = pd.read_csv('../data.csv')

# Create a function that calculate the distance intra class
# The intra class distance is the maximum distance between every vector of the class and its centroid
def intra_class_distance(separated_data,distance_function,mahalonobis=False):
    intra_class_distance = {}
    for class_label in separated_data:
        intra_class_distance[class_label] = 0
        centroid_class = np.mean(separated_data[class_label], axis=0)
        for i in range(1, len(separated_data[class_label])):
            if(mahalonobis):
                cov_matrix = np.cov(separated_data[class_label][i]) #TODO ca marche pas...
                distance = distance_function(separated_data[class_label][i], centroid_class, cov_matrix)
            else:
                distance = distance_function(separated_data[class_label][i], centroid_class)
            if distance > intra_class_distance[class_label]:
                intra_class_distance[class_label] = distance
    return intra_class_distance
        
# Create a function that separates the data by class
def separate_data_by_class(data, labels):
    separated_data = {}
    for i in range(2, len(data)):
        class_label = labels[i][1]
        if class_label not in separated_data:
            separated_data[class_label] = []  # Initialize an empty list for each class label if it doesn't exist
        separated_data[class_label].append(data[i][1:])
    return separated_data

# Create the function that calclates the distance beetween two classes
# the distance beetween two classes is the minimum distance beetween a vector of the first class and the centroid of the second class
def class_distance(class1, class2, distance_function):
    class_distance = 0
    centroid_class2 = np.mean(class2, axis=0)
    for i in range(1, len(class1)):
        distance = distance_function(class1[i], centroid_class2)
        if distance < class_distance:
            class_distance = distance
    return class_distance

# Create a function that calculates the distance inter class
# The inter class distance is the minimum distance between class_distance(class1, class2) and class_distance(class2, class1)
def inter_class_distance(class1,class2,distance_function):
    distance1 = class_distance(class1, class2, distance_function)
    distance2 = class_distance(class2, class1, distance_function)
    if distance1 < distance2:
        return distance1
    else:
        return distance2
    
# Create a function that calculates the overlap beetween two classes
# The overlap (intra_class_distance(class1) + intra_class_distance(class2))/2*inter_class_distance(class1,class2)
def overlap(class1,class2,distance_function):
    overlap = (intra_class_distance(class1,distance_function) + intra_class_distance(class2,distance_function))/(2*inter_class_distance(class1,class2,distance_function))
    return overlap



separatedData = separate_data_by_class(data_df.values, labels_df.values)
#intraClassEuclidean = intra_class_distance(separatedData, scipy.spatial.distance.euclidean)
#print(intraClassEuclidean)
# intraClassMahalanobis = intra_class_distance(separatedData, scipy.spatial.distance.mahalanobis, True)
# print(intraClassMahalanobis)
#intraClassCosine = intra_class_distance(separatedData, scipy.spatial.distance.cosine)
#print(intraClassCosine)


