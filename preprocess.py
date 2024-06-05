import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu

dataset_dir = r'C:\Users\brian\Documents\KULIAH\SEMESTER 8\Data Image Processing\UAS_DIP\Sport_class'


train_data = [] #list untuk nyimpen data
train_labels = []

train_dir = os.path.join(dataset_dir, 'train')
for class_name in os.listdir(train_dir):
  class_dir = os.path.join(train_dir, class_name)
  for image_name in os.listdir(class_dir):
    image_path = os.path.join(class_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100,100), interpolation=cv2.INTER_AREA) #pantes gak bisa di folder high jump/159.ink hapus aja
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # median = cv2.medianBlur(image, 1)
    # #mean threshold
    # img_hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    # h,s,v = cv2.split(img_hsv)
    # img_gray = (0.7 * s + 0.3 * v)
    # thresh = img_gray.mean()
    # img_th = img_gray > thresh
    # img_th = (img_th * 255).astype(np.uint8) #normalisasi agar bisa di proces waktu median blur
    # img_eq = cv2.equalizeHist(img_th)
    
    # otsu threhold
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img_hsv)
    img_gray = (0.7 * s + 0.3 * v)
    otsu = threshold_otsu(img_gray)
    img_otsu = img_gray > otsu
    img_otsu = (img_otsu * 255).astype(np.uint8) #normalisasi agar bisa di proces waktu median blur
    
    
  #  #mean threshold
  #   img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  #   h,s,v = cv2.split(img_hsv)
  #   img_gray = (0.7 * s + 0.3 * v)
  #   thresh = img_gray.mean()
  #   img_th = img_gray > thresh
  #   img_th = (img_th * 255).astype(np.uint8) #normalisasi agar bisa di proces waktu median blur
    
    #median and histoequal
    median = cv2.medianBlur(img_otsu, 3)
    img_eq = cv2.equalizeHist(median)

    # r,g,b = cv2.split(median)
    # # img_gray = (r+g+b)/3
    # x = img_th.flatten().reshape(-1,1)
    

    # kmeans = KMeans(n_clusters = 2) #cluster=2 karena dalam kasus ini kita ingin memisahkan 1 objek apel dengan bg
    # kmeans.fit(x)
    # label = kmeans.predict(x)
    # bin = label.reshape(img_th.shape) #mengembalikan ke ukuran img aslinya, bin = kernel

    # cluster_result = cv2.merge([r*bin, g*bin, b*bin])

    train_data.append(img_eq.flatten())
    train_labels.append(class_name)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
np.save('train_data.npy', train_data)
np.save('train_labels.npy', train_labels)

#LOAD TESTING DATA
test_data = [] #list untuk nyimpen data
test_labels = []

test_dir = os.path.join(dataset_dir, 'test')
for class_name in os.listdir(test_dir):
  class_dir = os.path.join(test_dir, class_name)
  for image_name in os.listdir(class_dir):
    image_path = os.path.join(class_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100,100), interpolation = cv2.INTER_AREA)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # median = cv2.medianBlur(image, 1) #COBA KALO MEDIAN DAN HISTOEQUAL DI PINDAH KE BAWAH
    # # #mean threshold
    # img_hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    # h,s,v = cv2.split(img_hsv)
    # img_gray = (0.7 * s + 0.3 * v)
    # thresh = img_gray.mean()
    # img_th = img_gray > thresh
    # img_th = (img_th * 255).astype(np.uint8) #
    # img_eq = cv2.equalizeHist(img_th)
    
    #otsu threhold
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img_hsv)
    img_gray = (0.7 * s + 0.3 * v)
    otsu = threshold_otsu(img_gray)
    img_otsu = img_gray > otsu
    img_otsu = (img_otsu * 255).astype(np.uint8) #normalisasi agar bisa di proces waktu median blur
    
    
    # #mean threshold
    # img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # h,s,v = cv2.split(img_hsv)
    # img_gray = (0.7 * s + 0.3 * v)
    # thresh = img_gray.mean()
    # img_th = img_gray > thresh
    # img_th = (img_th * 255).astype(np.uint8) #normalisasi agar bisa di proces waktu median blur
    
    #median and histoequal
    median = cv2.medianBlur(img_otsu, 3)
    img_eq = cv2.equalizeHist(median)


    # r,g,b = cv2.split(median)
    # # img_gray = (r+g+b)/3
    # x = img_th.flatten().reshape(-1,1)
    

    # kmeans = KMeans(n_clusters = 2) #cluster=2 karena dalam kasus ini kita ingin memisahkan 1 objek apel dengan bg
    # kmeans.fit(x)
    # label = kmeans.predict(x)
    # bin = label.reshape(img_th.shape) #mengembalikan ke ukuran img aslinya, bin = kernel

    # cluster_result = cv2.merge([r*bin, g*bin, b*bin])

    test_data.append(img_eq.flatten())
    test_labels.append(class_name)

test_data = np.array(test_data)
test_labels = np.array(test_labels)
np.save('test_data.npy', test_data)
np.save('test_labels.npy', test_labels)
