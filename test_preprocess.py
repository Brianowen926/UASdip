import os
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu

dataset_dir = r'C:\Users\brian\Documents\KULIAH\SEMESTER 8\Data Image Processing\UAS_DIP\Sport_class'

train_data = []  # list untuk nyimpen data
train_labels = []

train_dir = os.path.join(dataset_dir, 'train')
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (180, 180), interpolation=cv2.INTER_AREA)

        # Convert to HSV color space
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        img_gray = (0.7 * s + 0.3 * v)

        # Apply Otsu's thresholding
        otsu = threshold_otsu(img_gray)
        img_otsu = img_gray > otsu
        img_otsu = (img_otsu * 255).astype(np.uint8)

        # Apply median blurring
        median = cv2.medianBlur(img_otsu, 3)

        # Apply histogram equalization
        img_eq = cv2.equalizeHist(median)

        # Normalize image data
        scaler = StandardScaler()
        img_eq_normalized = scaler.fit_transform(img_eq.reshape(-1, 1)).reshape(img_eq.shape)

        # Flatten image data
        train_data.append(img_eq_normalized.flatten())
        train_labels.append(class_name)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
np.save('train_data.npy', train_data)
np.save('train_labels.npy', train_labels)

# LOAD TESTING DATA
test_data = []  # list untuk nyimpen data
test_labels = []

test_dir = os.path.join(dataset_dir, 'test')
for class_name in os.listdir(test_dir):
    class_dir = os.path.join(test_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (180, 180), interpolation=cv2.INTER_AREA)

        # Convert to HSV color space
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        img_gray = (0.7 * s + 0.3 * v)

        # Apply Otsu's thresholding
        otsu = threshold_otsu(img_gray)
        img_otsu = img_gray > otsu
        img_otsu = (img_otsu * 255).astype(np.uint8)

        # Apply median blurring
        median = cv2.medianBlur(img_otsu, 3)

        # Apply histogram equalization
        img_eq = cv2.equalizeHist(median)

        # Normalize image data
        scaler = StandardScaler()
        img_eq_normalized = scaler.fit_transform(img_eq.reshape(-1, 1)).reshape(img_eq.shape)

        # Flatten image data
        test_data.append(img_eq_normalized.flatten())
        test_labels.append(class_name)

test_data = np.array(test_data)
test_labels = np.array(test_labels)
np.save('test_data.npy', test_data)
np.save('test_labels.npy', test_labels)