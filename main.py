# Based off task 2 of Introduction to Machine Learning with Python: A Guide for Data Scientists - Andreas C. Müller and Sarah Guido

from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import mglearn

# Task is to learn to predict whether a tumour is malignant based on the measurements of the tissue
# This task uses realworld data from the Wisconsin Breast Cancer dataset

# First task is to load in the dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
# Datasets from scikit-learn are usually stored as Bunch objects, which contain some information about the dataset as well as the
# actual data. Bunch objects essentially just act as dictionaries. Any values can be accessed using a dot (bunch.key)
print("cancer.keys(): \n{}".format(cancer.keys()))

# The data contains 569 data points, 212 are labelled as malignant and 357 as benign: cancer.date just gets the data
# value from the dictionary
print("Shape of cancer data: \n{}".format(cancer.data.shape))

print("Sample counts per class \n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))


# Investigate how to number of neighbors affects the generalization of the model by analysing the accuracy 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# First thing to perform with the data set is to split the data into a training and test set so the accuracy of the model
# can be calculated at the end

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state = 60)

training_accuracy = []
test_accuracy = []

neighbors_count = range(1,11)

for i in neighbors_count:
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))

plt.plot(neighbors_count,training_accuracy,label="training accuracy")
plt.plot(neighbors_count,test_accuracy, label="test accuracy")
plt.ylabel("Acuracy")
plt.xlabel("n_neighbors")
plt.legend(
