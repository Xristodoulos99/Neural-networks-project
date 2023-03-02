import numpy as np
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA, IncrementalPCA

(train_X, train_y), (test_X, test_y)= mnist.load_data()


labels=int(train_X[0].size)#calculating the data labels size (this case 28*28=784)
train_X_size=int(train_X.size / labels)#calculation the train data size(this case 60000)
test_X_size=int(test_X.size / labels)#calculation the test data size (this case 10000)


#normilize data range from 0 -> 255 to 0 -> 1
train_X=np.reshape(train_X, (train_X_size, labels))
test_X=np.reshape(test_X, (test_X_size, labels))
train_X=train_X/255.0
test_X=test_X/255.0


noise_factor = 0.2
x_train_noisy = train_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_X.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)


#making the testing dataset smaller in some cases for time consuming issues
train_X=x_train_noisy[:]

temp=train_X[:]

times=time.time()

#Lowering the "quality" with PCA
pca1=PCA(n_components=49).fit(train_X)
train_X=pca1.transform(train_X)

new=pca1.inverse_transform(train_X)

print(time.time()-times," s")
plt.figure(figsize=(40, 4))
for i in range(10):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(temp[i].reshape(28, 28), cmap="binary")

    # display reconstructed (after noise removed) image
    ax = plt.subplot(3, 20, 40 + i + 1)
    plt.imshow(new[i].reshape(28, 28), cmap="binary")

plt.show()
