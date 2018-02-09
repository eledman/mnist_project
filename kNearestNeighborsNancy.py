# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:42:35 2018

@author: Nancy
"""

from read_mnist import load_data 
# make sure read_mnist is commented/uncommented to provide correct data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

#loading MNIST digits dataset
train_set, valid_set, test_set = load_data()

best_k = 0
max_accuracy = 0

# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 30, 2):
	# train the k-Nearest Neighbor classifier with the current value of `k`
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(train_set[0], train_set[1])
 
	# evaluate the model and update the best accuracy and corresponding k
	score = model.score(valid_set[0], valid_set[1])
	if (score>max_accuracy):
		max_accuracy = score
		best_k = k
 
# find the value of k that has the largest accuracy
print(best_k, "achieved highest accuracy of ", max_accuracy)


# re-train our classifier using the best k value and predict the labels of the
# test data
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(train_set[0], train_set[1])
predictions = model.predict(test_set[0])
 
# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(test_set[1], predictions))
