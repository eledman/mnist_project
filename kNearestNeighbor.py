from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np

#loading MNIST digits dataset
mnist = datasets.load_digits()

# this process of splitting results in a 60-20-20 train-validate-test split

# take the MNIST data and construct the training and testing split, using 80% of the
# data for training and 20% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
	mnist.target, test_size=0.2, random_state=42)

# take 25% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
	test_size=0.25, random_state=84)

best_k = 0
max_accuracy = 0

# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in xrange(1, 30, 2):
	# train the k-Nearest Neighbor classifier with the current value of `k`
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(trainData, trainLabels)
 
	# evaluate the model and update the best accuracy and corresponding k
	score = model.score(valData, valLabels)
	if (score>max_accuracy):
		max_accuracy = score
		best_k = k
 
# find the value of k that has the largest accuracy
print(best_k, "achieved highest accuracy of ", max_accuracy)


# re-train our classifier using the best k value and predict the labels of the
# test data
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(trainData, trainLabels)
predictions = model.predict(testData)
 
# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))
