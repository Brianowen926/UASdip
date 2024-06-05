from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.cluster import KMeans
import numpy as np
import cv2

train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')

from sklearn.preprocessing import LabelEncoder

# ...

# Encode the labels
le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_labels)
test_labels_encoded = le.transform(test_labels)

# Create the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=100, metric='manhattan')

# Train the KNN classifier
knn_classifier.fit(train_data, train_labels_encoded)

# Make predictions on the test data
knn_prediction = knn_classifier.predict(test_data)
knn_acc = accuracy_score(test_labels_encoded, knn_prediction)
print('Accuracy: ', knn_acc*100)

image_path = r'C:\Users\brian\Documents\KULIAH\SEMESTER 8\Data Image Processing\UAS_DIP\Sport_class\test\bowling\1.jpg'
image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img_hsv)
image = (0.7 * s + 0.3 * v)
image = cv2.resize(image, (180,180), interpolation = cv2.INTER_AREA)
image_flat = image.flatten()

predicted_class = knn_classifier.predict([image_flat])
predicted_class_name = le.inverse_transform(predicted_class)[0]
print('Predicted Class: ', predicted_class_name)