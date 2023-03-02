import time
from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
import numpy as np


#loading data from mnist
(train_X, train_y), (test_X, test_y)= mnist.load_data()

labels=int(train_X[0].size)#calculating the data labels size (this case 28*28=784)
train_X_size=int(train_X.size / labels)#calculation the train data size(this case 60000)
test_X_size=int(test_X.size / labels)#calculation the test data size (this case 10000)

#reshaping training input and test input into 2D tables
train_X_2d=np.reshape(train_X, (train_X_size, labels))#converting the labels from 2 dimentiosn to 1 (60000 , 28 , 28 -> 60000 , 784)
test_X_2d=np.reshape(test_X, (test_X_size, labels))#same as above with the test data table


#K NEAREST NEIGHBOR with 1 neighbor
k_neighbors=1
knn = KNeighborsClassifier(k_neighbors)

#fit needs a 2D table and a 1d table (data,lables and expected outcome)
knn.fit(train_X_2d, train_y)

#time and accuracy of knn-1

startknn1 = time.time() #time before running score

print("knn-1 Accuracy: ", knn.score(test_X_2d, test_y) * 100,"%")

endknn1 = time.time() #time after score

# timeafter - timebefore= running time
print("knn-1, seconds: ",endknn1 - startknn1)

print("-----------------------")

#K NEAREST NEIGHBOR with 3 neighbor
k_neighbors=3
knn = KNeighborsClassifier(k_neighbors)
knn.fit(train_X_2d, train_y)

#print time and accuracy for knn with 3 neighbors
startknn3 = time.time()
print("knn-3 Accuracy: ", knn.score(test_X_2d, test_y)* 100,"%")
endknn3 = time.time()
print("knn-3, seconds: ",endknn3 - startknn3)

print("-----------------------")


#nearestCentroid algorithm

nc=NearestCentroid()
nc.fit(train_X_2d, train_y)

#print time and accuracy for nearest centroid algorithm
startnc = time.time()
print("nc Accuracy: ", nc.score(test_X_2d, test_y)* 100,"%")
endnc = time.time()
print("nc, seconds: ",endnc - startnc)
