import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')

# #CLASSIFICATION KNN
knn_classifier = KNeighborsClassifier(n_neighbors=100, metric = 'manhattan') #karena class nya cuman 2. CATATAN di ppt hlm 8
knn_classifier.fit(train_data, train_labels)

knn_prediction = knn_classifier.predict(test_data)
knn_acc = accuracy_score(test_labels, knn_prediction)
print('Accuracy: ' , knn_acc*100)

image_path = r'C:\Users\brian\Documents\KULIAH\SEMESTER 8\Data Image Processing\UAS_DIP\Sport_class\test\air hockey\1.jpg'
image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img_hsv)
image = (0.7 * s + 0.3 * v)
image = cv2.resize(image, (100,100), interpolation = cv2.INTER_AREA)
image_flat = image.flatten()

predicted_class = knn_classifier.predict([image_flat])
print('Predicted Class: ', predicted_class)
