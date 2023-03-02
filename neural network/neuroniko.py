import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import time

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    max = np.max(x, axis=1, keepdims=True)
    exp = np.exp(x - max)
    sum = np.sum(exp, axis=1, keepdims=True)
    x = exp / sum
    return x

def dReLU(x):
    return 1 * (x > 0)

def calc_accuracy(o3,y):
    #if the expected outcome and the actual outcome are the same then the accuracy increases by 1
    #this is repeated for all the batch.
    tempacc=0
    outcome = [0] * len(o3)
    correct_outcome = [0] * len(o3)
    for tempi in range(len(o3)):
        outcome[tempi] = np.argmax(o3[tempi])
        correct_outcome[tempi] = np.argmax(y[tempi])
        if (correct_outcome[tempi] == outcome[tempi]):
            tempacc+=1
    return tempacc


def shuffle(train_X,train_y):
    #shuffling the data
    shuffled_x = train_X[:]
    shuffled_y = train_y[:]
    #this moves every data 10000 places , when it reaches the dataset size it loops back to the start of it.
    for data in range(train_X_size):
        newi = (data + 100000) % train_X_size
        shuffled_x[newi] = train_X[data]
        shuffled_y[newi] = train_y[data]
    return shuffled_x,shuffled_y

(train_X, train_y), (test_X, test_y)= mnist.load_data()

labels=int(train_X[0].size)#calculating the data labels size (this case 28*28=784)
train_X_size=int(train_X.size / labels)#calculation the train data size(this case 60000)
test_X_size=int(test_X.size / labels)#calculation the test data size (this case 10000)


#normilize data range from 0 -> 255 to 0 -> 1
train_X=np.reshape(train_X, (train_X_size, labels))
test_X=np.reshape(test_X, (test_X_size, labels))
train_X=train_X/255.0
test_X=test_X/255.0


#There are 10 classes so the expected output need to be an array of the 10 classes where the expected class is "1" and the rest is "0"
#Chanching both test and train y to this format
table = np.zeros((train_y.shape[0], 10))
for i in range(train_y.shape[0]):
    table[i][int(train_y[i])] = 1
train_y=table

table = np.zeros((test_y.shape[0], 10))
for i in range(test_y.shape[0]):
    table[i][int(test_y[i])] = 1
test_y = table

#Dataset is ready

#initialiazation
batch=64
#number of batches, has to be integer
number_of_batches=train_X_size//batch
if (train_X_size % batch != 0):
    number_of_batches=number_of_batches+1
epochs=10
EraAccuracy = [0.0] * epochs

#learning rate
lr=0.001
#loading the first batches

#loss = 0.0
accuracy = 0.0

#Setting the layers
#We set the layers by generating their weights and their biases

L1_neurons=512
L2_neurons=256
Loutcome_neurons=10
#layer 1 has a weight for each neuron from every input (784*L1_neurons)
W_L1 = np.random.randn(labels, L1_neurons)
b_L1 = np.random.randn(L1_neurons, )

#layer 2 has a weight for each neuron from every input neuron from the previous layer
W_L2 = np.random.randn(W_L1.shape[1], L2_neurons)
b_L2 = np.random.randn(L2_neurons, )

#layer 3 is the outcome layer and has wights for all 10 outcomes from all the neuron from the previous layer
W_Outcome = np.random.randn(W_L2.shape[1], Loutcome_neurons)
b_L3 = np.random.randn(Loutcome_neurons, )

#We now begin the training
#def train(self):
l=0
totalstartnc = time.time()
for epoch in range(epochs):
    #getting the pc time before any code is ran so we can time
    #each Era's training time
    startnc = time.time()
    print("Era: ", epoch+1)
    l = 0
    #accuracy
    accuracy = 0.0

    #We shaffle the data each era so the

    train_X,train_y=shuffle(train_X,train_y)



    for batch_number in range(number_of_batches):
        #we need to set the new batch of data everytime until complete the whole training dataset

        #checking whether we're on the last batch
        if ((batch_number + 1) <= number_of_batches):
            x = train_X[(batch_number * batch) : ((batch_number + 1) * batch)]
            y = train_y[(batch_number * batch) : ((batch_number + 1) * batch)]

        #feedforward process
        #X*W + bias
        #x and w are tables so we need to find the inner product
        #the dot function implemanted in numpy does that for us

        #Layer 1
        o1 = relu(x.dot(W_L1) + b_L1)
        #we use the activation function of relu for out layers

        #Layer 2
        o2 = relu(o1.dot(W_L2) + b_L2)

        #Layer 3
        #for class sorting we traditionally use a softmax activation function
        o3 = softmax(o2.dot(W_Outcome) + b_L3)


        #calculate the error for the batch
        #outcome - expected outcome for all the batch
        error = o3 - y

        #backprop



        #as we are working with batches we are
        #dividing the error with the batchsize to find the impact of each neuron
        error/=batch

        Delta_W3 = np.dot(error.T, o2).T
        Delta_W2 = np.dot((np.dot((error), W_Outcome.T) * dReLU(o1.dot(W_L2) + b_L2)).T, o1).T
        Delta_W1 = np.dot((np.dot(np.dot((error), W_Outcome.T) * dReLU(o1.dot(W_L2) + b_L2), W_L2.T) * dReLU(x.dot(W_L1) + b_L1)).T,
                          x).T

        db3 = np.sum(error, axis=0)
        db2 = np.sum(np.dot((error), W_Outcome.T) * dReLU(o1.dot(W_L2) + b_L2), axis=0)
        db1 = np.sum((np.dot(np.dot((error), W_Outcome.T) * dReLU(o1.dot(W_L2) + b_L2), W_L2.T) * dReLU(x.dot(W_L1) + b_L1)),
                     axis=0)


        W_Outcome = W_Outcome - lr * Delta_W3
        W_L2 = W_L2 - lr * Delta_W2
        W_L1 = W_L1 - lr * Delta_W1

        b_L3 = b_L3 - lr * db3
        b_L2 = b_L2 - lr * db2
        b_L1 = b_L1 - lr * db1


        #loss with mse
        l =l + np.mean(error * error)


        accuracy+=calc_accuracy(o3,y)

    era_accuracy=accuracy/train_X_size * 100
    print("era_accuracy= ",era_accuracy,"%")
    endnc = time.time()
    print("Training time for Era:",epoch + 1, " :", endnc - startnc)
    EraAccuracy[epoch]=era_accuracy





totalendnc = time.time()
print("Total training time: ", totalendnc-totalstartnc)
#Running the test data
x = test_X
y = test_y
startnc = time.time()
#passing the data though feedfordward only
#backprop is not needed for the
o1 = relu(x.dot(W_L1) + b_L1)
o2 = relu(o1.dot(W_L2) + b_L2)
o3 = softmax(o2.dot(W_Outcome) + b_L3)


accuracy=calc_accuracy(o3 , y)/test_X_size
endnc = time.time()
print("Testing time:",epoch + 1, " :", endnc - startnc)
print("testing accuracy:", 100 * accuracy, "%")


#plotting data
eras_list = list(range(1, epochs+1))
plt.plot(eras_list,EraAccuracy)
plt.ylabel('Accuracy')
plt.xlabel('Training era')
plt.title('Accuracy in each training era!')
plt.show()