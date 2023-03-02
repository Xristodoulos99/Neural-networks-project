import numpy as np
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA

(train_X, train_y), (test_X, test_y)= mnist.load_data()


labels=int(train_X[0].size)#calculating the data labels size (this case 28*28=784)
train_X_size=int(train_X.size / labels)#calculation the train data size(this case 60000)
test_X_size=int(test_X.size / labels)#calculation the test data size (this case 10000)



#normilize data range from 0 -> 255 to 0 -> 1
train_X=np.reshape(train_X, (train_X_size, labels))
test_X=np.reshape(test_X, (test_X_size, labels))
train_X=train_X/255.0
test_X=test_X/255.0


#making the testing dataset smaller in some cases for time consuming issues
train_X=train_X[0:20000]
train_y=train_y[0:20000]



#Making the problem a 2 class problem instead of 10
j=0
for i in train_y:
    train_y[j]=i%2
    j+=1
j=0
for i in test_y:
    test_y[j]=i%2
    j+=1


#Lowering the "quality" with PCA
#pca1=PCA(.92)
#pca2=PCA(.923)
#train_X=pca1.fit_transform(train_X)
#test_X=pca2.fit_transform(test_X)
#print(train_X.shape)
#print(test_X.shape)



#plt.figure
#plt.imshow(approximation1[1].reshape(28,28), cmap='gray')
#plt.show()
#plt.figure
#plt.imshow(approximation2[1].reshape(28,28), cmap='gray')
#plt.show()





#random testing to get comfortable with the library

"""
from sklearn.svm import SVC
timestart=time.time()
classifier = SVC(gamma= 0.0001, C=100,kernel = 'rbf' )
classifier.fit(train_X, train_y)
y_pred = classifier.predict(test_X)
print("training time:" , time.time()-timestart , "s")
timetest=time.time()
print(accuracy_score(test_y,y_pred)*100.0)
print("Test time",time.time()-timetest ," s")
"""

#C testing

"""
tests for C

C_table=[0.01,0.1,1,10,100,1000,10000]
gamma=0.0
acc=[]
tt=time.time()
times=[]
for C in C_table:
    t=time.time()
    classifier = SVC(gamma= 0.0001, C=C,kernel = 'rbf' )
    classifier.fit(train_X, train_y)
    y_pred = classifier.predict(test_X)

    acc.append(accuracy_score(test_y,y_pred)*100.0)
    times.append(time.time()-t)
"""

"""
plt.subplots(figsize=(10, 5))
plt.semilogx(C_table, times,'-gD' ,color='red' , label="Testing time")
plt.grid(True)
plt.xlabel("Cost Parameter C")
plt.ylabel("Time in seconds")
plt.legend()
plt.title('Time for Cost')
plt.show()

plt.subplots(figsize=(10, 5))
plt.semilogx(C_table, acc,'-gD' ,color='red' , label="Testing Accuracy")
plt.grid(True)
plt.xlabel("Cost Parameter C")
plt.ylabel("Accuracy")
plt.legend()
plt.title('Accuracy for Cost')
plt.show()

print("time:",t)
print("acc",acc)
"""

#gamma testing

"""
for gamma values

from sklearn.svm import SVC
gamma_table=[0.0001,0.001,0.01,0.1,1,10,100,1000,"auto","scale"]
tr_acc=[]
acc=[]
tt=time.time()
times=[]
for g in gamma_table:
    t=time.time()
    svm = SVC(gamma=g,C=100)
    svm.fit(train_X, train_y)

    y_pred_t=svm.predict(train_X)
    y_pred = svm.predict(test_X)


    acc.append(accuracy_score(test_y,y_pred))
    tr_acc.append(accuracy_score(train_y,y_pred_t))

    times.append(time.time()-t)
print(acc)
print(tr_acc)
"""


#kernel testing
"""
for kernel comparing

from sklearn.svm import SVC
tr_acc=[]
acc=[]
tt=time.time()
times=[]
kernel=["linear", "poly", "rbf" ,"sigmoid"]
for j in kernel:
    t=time.time()
    svm=SVC(kernel=j)
    svm.fit(train_X,train_y)
    print("trained")

    y_pred_t=svm.predict(train_X)
    print("training test")
    y_pred=svm.predict(test_X)
    print("test test")

    tr_acc.append(accuracy_score(y_pred_t,train_y))
    acc.append(accuracy_score(y_pred,test_y))
    print(tr_acc)
    print(acc)
    times.append(time.time()-t)

plt.plot(kernel, times,'-gD' ,color='red' , label="Testing time")
plt.grid(True)
plt.xlabel("kernel")
plt.ylabel("Time in seconds")
plt.legend()
plt.title('Time for kernel')
plt.show()


plt.plot(kernel, acc,'-gD' ,color='green' , label="Testing Accuracy")
plt.grid(True)
plt.xlabel("kernel")
plt.ylabel("Testing Accuracy")
plt.legend()
plt.title('Accuracy for kernel')
plt.show()


plt.plot(kernel, tr_acc,'-gD' ,color='red' , label="Training Accuracy")
plt.grid(True)
plt.xlabel("kernel")
plt.ylabel("Training Accuracy")
plt.legend()
plt.title('Accuracy for kernel')
plt.show()
"""



#the best svm , to compare with knn and centroid
from sklearn.svm import SVC
gamma=0.01
C=0.1
tr_acc=0.0
kernel="poly"
acc=0.0
svm=SVC(C=C,gamma=gamma,kernel=kernel)
tt=time.time()
svm.fit(train_X,train_y)
print("training time = ", time.time()-tt , "s")

y_pred_t=svm.predict(train_X)
print("Training accuracy: ",accuracy_score(y_pred_t,train_y))
tt=time.time()
y_pred=svm.predict(test_X)
print("testing time = ", time.time()-tt , "s")

print("Test accuracy",accuracy_score(y_pred,test_y))


for i in range(100):
    if (y_pred[i] != test_y[i]):
        print("Correct value= ",test_y[i])
        print("predicted value was =", y_pred[i])
        fig = plt.figure
        plt.imshow((test_X[i]).reshape((28, 28)), cmap='gray')
        plt.show()
        plt.xlabel("Correct digit")
        plt.ylabel("Predicted digit was wrong ")

