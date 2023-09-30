import pandas as pd
import scipy
import numpy as np

def remove_ids(data):
    data = data.drop(data.columns[0], axis=1)
    return data

# Create a function that separates the data by class
def separate_data_by_class(data, labels):
    separated_data = {}
    for i in range(1, len(data)):
        class_label = labels[i][1]
        if class_label not in separated_data:
            separated_data[class_label] = []  # Initialize an empty list for each class label if it doesn't exist
        separated_data[class_label].append(data[i])
    return separated_data

#Create a function to calcuate the covariance matrx
def covariance_matrix(data):
    cov_matrix = np.cov(data.T.values)
    # cov_matrix = np.linalg.inv(cov_matrix.T)
    return cov_matrix

# Create a function that calculate the distance intra class
# The intra class distance is the maximum distance between every vector of the class and its centroid
def intra_class_distance(data,distance_function,mahalonobis=False,cov_matrix=None):
    intra_distance = 0
    centroid_class = np.mean(data, axis=0)
    for i in range(1, len(data)):
        if(mahalonobis):
            distance = distance_function(data[i], centroid_class, cov_matrix)
        else:
            distance = distance_function(data[i], centroid_class)
        if distance > intra_distance:
            intra_distance = distance
    return intra_distance

# Create the function that calclates the distance beetween two classes
# the distance beetween two classes is the minimum distance beetween a vector of the first class and the centroid of the second class
def class_distance(class1, class2, distance_function,mahalonobis=False,cov_matrix=None):
    class_distance = 1000000000
    centroid_class2 = np.mean(class2, axis=0)
    for i in range(1, len(class1)):
        if(mahalonobis):
            distance = distance_function(class1[i], centroid_class2, cov_matrix)
        else:
            distance = distance_function(class1[i], centroid_class2)
        if distance < class_distance:
            class_distance = distance
    return class_distance

# Create a function that calculates the distance inter class
# The inter class distance is the minimum distance between class_distance(class1, class2) and class_distance(class2, class1)
def inter_class_distance(class1,class2,distance_function,mahalonobis=False,cov_matrix=None):
    distance1 = class_distance(class1, class2, distance_function,mahalonobis,cov_matrix)
    distance2 = class_distance(class2, class1, distance_function, mahalonobis,cov_matrix)
    if distance1 < distance2:
        return distance1
    else:
        return distance2
    
# Create a function that calculates the overlap beetween two classes
# The overlap (intra_class_distance(class1) + intra_class_distance(class2))/2*inter_class_distance(class1,class2)
def overlap(class1,class2,distance_function,mahalonobis=False,cov_matrix=None):
    overlap = (intra_class_distance(class1,distance_function,mahalonobis,cov_matrix) + intra_class_distance(class2,distance_function,mahalonobis,cov_matrix))/(2*inter_class_distance(class1,class2,distance_function,mahalonobis,cov_matrix))
    return overlap

#Extract the labels
labels_df = pd.read_csv('labels.csv')

#Extract the data
data_df = pd.read_csv('data.csv')
#print the shape of the data
print(data_df.shape)

#remove ids
data_df = remove_ids(data_df)
print(data_df.shape)

#calculate the covariance matrix
cov_matrix = covariance_matrix(data_df)
print(len(cov_matrix),len(cov_matrix[0]))

#separate the data by class
separatedData = separate_data_by_class(data_df.values, labels_df.values)
print(len(separatedData['PRAD']),len(separatedData['BRCA']))

#Calculate the overlap beetween two classes using euclidean distance
# overlapPRADBRCA = overlap(separatedData['PRAD'],separatedData['BRCA'],scipy.spatial.distance.euclidean)
# overlapPRADKIRC = overlap(separatedData['PRAD'],separatedData['KIRC'],scipy.spatial.distance.euclidean)
# overlapPRADLUAD = overlap(separatedData['PRAD'],separatedData['LUAD'],scipy.spatial.distance.euclidean)
# overlapPRADCOAD = overlap(separatedData['PRAD'],separatedData['COAD'],scipy.spatial.distance.euclidean)
# overlapBRCAKIRC = overlap(separatedData['BRCA'],separatedData['KIRC'],scipy.spatial.distance.euclidean)
# overlapBRCALUAD = overlap(separatedData['BRCA'],separatedData['LUAD'],scipy.spatial.distance.euclidean)
# overlapBRCACOAD = overlap(separatedData['BRCA'],separatedData['COAD'],scipy.spatial.distance.euclidean)
# overlapKIRCLUAD = overlap(separatedData['KIRC'],separatedData['LUAD'],scipy.spatial.distance.euclidean)
# overlapKIRCCOAD = overlap(separatedData['KIRC'],separatedData['COAD'],scipy.spatial.distance.euclidean)
# overlapLUADCOAD = overlap(separatedData['LUAD'],separatedData['COAD'],scipy.spatial.distance.euclidean)

# print('PRAD BRCA Euclidean', overlapPRADBRCA)
# print('PRAD KIRC Euclidean', overlapPRADKIRC)
# print('PRAD LUAD Euclidean', overlapPRADLUAD)
# print('PRAD COAD Euclidean', overlapPRADCOAD)
# print('BRCA KIRC Euclidean', overlapBRCAKIRC)
# print('BRCA LUAD Euclidean', overlapBRCALUAD)
# print('BRCA COAD Euclidean', overlapBRCACOAD)
# print('KIRC LUAD Euclidean', overlapKIRCLUAD)
# print('KIRC COAD Euclidean', overlapKIRCCOAD)
# print('LUAD COAD Euclidean', overlapLUADCOAD)

# #Calculate the overlap beetween two classes using cosine distance
# overlapPRADBRCA = overlap(separatedData['PRAD'],separatedData['BRCA'],scipy.spatial.distance.cosine)
# overlapPRADKIRC = overlap(separatedData['PRAD'],separatedData['KIRC'],scipy.spatial.distance.cosine)
# overlapPRADLUAD = overlap(separatedData['PRAD'],separatedData['LUAD'],scipy.spatial.distance.cosine)
# overlapPRADCOAD = overlap(separatedData['PRAD'],separatedData['COAD'],scipy.spatial.distance.cosine)
# overlapBRCAKIRC = overlap(separatedData['BRCA'],separatedData['KIRC'],scipy.spatial.distance.cosine)
# overlapBRCALUAD = overlap(separatedData['BRCA'],separatedData['LUAD'],scipy.spatial.distance.cosine)
# overlapBRCACOAD = overlap(separatedData['BRCA'],separatedData['COAD'],scipy.spatial.distance.cosine)
# overlapKIRCLUAD = overlap(separatedData['KIRC'],separatedData['LUAD'],scipy.spatial.distance.cosine)
# overlapKIRCCOAD = overlap(separatedData['KIRC'],separatedData['COAD'],scipy.spatial.distance.cosine)
# overlapLUADCOAD = overlap(separatedData['LUAD'],separatedData['COAD'],scipy.spatial.distance.cosine)

# print('PRAD BRCA Cosine', overlapPRADBRCA)
# print('PRAD KIRC Cosine', overlapPRADKIRC)
# print('PRAD LUAD Cosine', overlapPRADLUAD)
# print('PRAD COAD Cosine', overlapPRADCOAD)
# print('BRCA KIRC Cosine', overlapBRCAKIRC)
# print('BRCA LUAD Cosine', overlapBRCALUAD)
# print('BRCA COAD Cosine', overlapBRCACOAD)
# print('KIRC LUAD Cosine', overlapKIRCLUAD)
# print('KIRC COAD Cosine', overlapKIRCCOAD)
# print('LUAD COAD Cosine', overlapLUADCOAD)

#Calculate the overlap beetween two classes using mahalanobis distance
overlapPRADBRCA = overlap(separatedData['PRAD'],separatedData['BRCA'],scipy.spatial.distance.mahalanobis,True,cov_matrix)
print('PRAD BRCA Mahalanobis', overlapPRADBRCA)
overlapPRADKIRC = overlap(separatedData['PRAD'],separatedData['KIRC'],scipy.spatial.distance.mahalanobis,True,cov_matrix)
print('PRAD KIRC Mahalanobis', overlapPRADKIRC)
overlapPRADLUAD = overlap(separatedData['PRAD'],separatedData['LUAD'],scipy.spatial.distance.mahalanobis,True,cov_matrix)
print('PRAD LUAD Mahalanobis', overlapPRADLUAD)
overlapPRADCOAD = overlap(separatedData['PRAD'],separatedData['COAD'],scipy.spatial.distance.mahalanobis,True,cov_matrix)
print('PRAD COAD Mahalanobis', overlapPRADCOAD)
overlapBRCAKIRC = overlap(separatedData['BRCA'],separatedData['KIRC'],scipy.spatial.distance.mahalanobis,True,cov_matrix)
print('BRCA KIRC Mahalanobis', overlapBRCAKIRC)
overlapBRCALUAD = overlap(separatedData['BRCA'],separatedData['LUAD'],scipy.spatial.distance.mahalanobis,True,cov_matrix)
print('BRCA LUAD Mahalanobis', overlapBRCALUAD)
overlapBRCACOAD = overlap(separatedData['BRCA'],separatedData['COAD'],scipy.spatial.distance.mahalanobis,True,cov_matrix)
print('BRCA COAD Mahalanobis', overlapBRCACOAD)
overlapKIRCLUAD = overlap(separatedData['KIRC'],separatedData['LUAD'],scipy.spatial.distance.mahalanobis,True,cov_matrix)
print('KIRC LUAD Mahalanobis', overlapKIRCLUAD)
overlapKIRCCOAD = overlap(separatedData['KIRC'],separatedData['COAD'],scipy.spatial.distance.mahalanobis,True,cov_matrix)
print('KIRC COAD Mahalanobis', overlapKIRCCOAD)
overlapLUADCOAD = overlap(separatedData['LUAD'],separatedData['COAD'],scipy.spatial.distance.mahalanobis,True,cov_matrix)
print('LUAD COAD Mahalanobis', overlapLUADCOAD)

