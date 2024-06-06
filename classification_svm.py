import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from sklearn import svm

train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')


#CLASSIFICATION SVM
svm_classifier = svm.SVC(kernel='sigmoid')
svm_classifier.fit(train_data, train_labels)

svm_prediction = svm_classifier.predict(test_data)
svm_acc = accuracy_score(test_labels, svm_prediction)
print('Accuracy of SVM: ', svm_acc)

image_path = r'C:\Users\brian\Documents\KULIAH\SEMESTER 8\Data Image Processing\UAS_DIP\Sport_class\test\air hockey\1.jpg'
image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img_hsv)
image = (0.7 * s + 0.3 * v)
image = cv2.resize(image, (100,100), interpolation = cv2.INTER_AREA)
image_flat = image.flatten()

predicted_class = svm_classifier.predict([image_flat])
print('Predicted Class: ', predicted_class)
